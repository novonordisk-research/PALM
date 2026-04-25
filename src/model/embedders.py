import logging
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import regex as re
import torch
# from Bio.SeqUtils.ProtParam import ProteinAnalysis #this was for biophysics descriptors probably removing is best choice
from omegaconf import DictConfig
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from transformers import (  # move to the hugging face transformer implementation for consistency
    AutoModelForMaskedLM,
    AutoTokenizer,
    EsmModel,
)

from src.model.abstract_components import (
    AAFeaturizerModel,
    EmbedderModel,
    LLMEmbedderModel,
    #component_type, #I do not remember what this was calling 
)


logger = logging.getLogger(__name__)

# ----- Embedder implementations ---- #

class ESM(LLMEmbedderModel):
    """ESM (Embedder Model) class for extracting embeddings from protein sequences.

    Args:
        cfg (DictConfig): Configuration dictionary for the ESM model.

    Attributes:
        n_layers (int): Number of layers to extract embeddings from.
        model (EsmModel): The ESM model instance.
        tokenizer (AutoTokenizer): Tokenizer for sequence processing.
        device (torch.device): Device (CPU or GPU) to run the model on.
        chain_break_value (str): Chain break value for sequence duplication.
        chain_break_len (int): Length of the chain break value.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # Parse model name for hyperparameters
        split = str(self.cfg.embedder.model_name).split("_")
        self.n_layers = int(re.sub("\D", "", split[1]))
        logger.info(
            f"Loading model: {self.cfg.embedder.model_name}, will extract embeddings from {self.n_layers}-th layer"
        )

        self.factor_1, self.factor_2, self.factor_3 = self.batch_scale_factors[
            self.cfg.embedder.model_name
        ]

        # Setup cache directory
        hf_dir_path = Path(
            self.cfg.persistence.data_root_folder,
            self.cfg.persistence.pretrained_weights,
        )
        if not hf_dir_path.exists():
            logger.info(f"Creating cache directory: {hf_dir_path}")
            hf_dir_path.mkdir(parents=True)

        try:
            # Initialize model and tokenizer
            model_name = f"facebook/{self.cfg.embedder.model_name}"
            self.model = EsmModel.from_pretrained(model_name, cache_dir=hf_dir_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=hf_dir_path)

            # Setup device (prefer CUDA, avoid MPS which has issues with ESM)
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

            logger.info(f"Moving model to {self.device}")
            self.model = self.model.to(self.device).eval()

            # Initialize chain break parameters
            self.chain_break_len = 25
            self.chain_break_value = ""
            if self.cfg.embedder.n_copies and self.cfg.embedder.n_copies > 1:
                if self.cfg.embedder.chain_break == "poly-gly-linker":
                    self.chain_break_value = "G" * self.chain_break_len
                else:
                    raise ValueError(f"Unsupported chain break: {self.cfg.embedder.chain_break}")

            # Validate mask placeholder if configured
            self._validate_mask_placeholder()

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _validate_mask_placeholder(self) -> None:
        """Validate that the mask placeholder is not a valid amino acid."""
        placeholder = self.cfg.embedder.get("mask_placeholder")
        if placeholder is None:
            return
        
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        if placeholder in valid_aa:
            raise ValueError(
                f"Mask placeholder '{placeholder}' cannot be a valid amino acid. "
                f"Use a character like '#' or '*' instead."
            )
        logger.info(f"Mask placeholder '{placeholder}' will be replaced with '{self.tokenizer.mask_token}'")

    def replace_mask_placeholders(self, spaced_sequences: list[str]) -> list[str]:
        """Replace placeholder characters with the tokenizer's mask token.
        
        Args:
            spaced_sequences: List of sequences with spaces between residues.
                              e.g., ["M K T A Y # G C V S", ...]
        
        Returns:
            Sequences with placeholders replaced by the mask token.
                              e.g., ["M K T A Y <mask> G C V S", ...]
        """
        placeholder = self.cfg.embedder.get("mask_placeholder")
        if placeholder is None:
            return spaced_sequences
        
        mask_token = self.tokenizer.mask_token  # "<mask>" for ESM models
        return [seq.replace(placeholder, mask_token) for seq in spaced_sequences]

    def duplicate_sequences(self, sequences: list[str]) -> list[str]:
        """Duplicate sequences with chain breaks if configured."""
        if not (self.cfg.embedder.n_copies and self.cfg.embedder.n_copies > 1):
            return sequences

        duplicated = []
        for sequence in sequences:
            copies = [sequence] * self.cfg.embedder.n_copies
            duplicated.append(self.chain_break_value.join(copies))
        return duplicated

    def process_batch(self, seqs: list[str]) -> list[torch.Tensor]:
        """Process a batch of sequences and extract embeddings."""
        self.validate_layer_idx()
        seq_lens = [len(x) for x in seqs]
        seqs = [" ".join(list(x)) for x in seqs]

        # Replace mask placeholders with actual mask tokens
        seqs = self.replace_mask_placeholders(seqs)

        # Tokenize sequences
        token_encoding = self.tokenizer.batch_encode_plus(
            seqs, add_special_tokens=True, padding="longest"
        )

        input_ids = torch.tensor(token_encoding["input_ids"]).to(self.device)
        attention_mask = torch.tensor(token_encoding["attention_mask"]).to(self.device)

        with torch.inference_mode():
            if self.should_use_specific_layer():
                # print(f"Getting layer specific embedding from {self.cfg.embedder.layer_idx}")
                # Extract from specific layer
                embedding_repr = self.model(
                    input_ids, attention_mask=attention_mask, output_hidden_states=True
                )
                hidden_states = embedding_repr.hidden_states[self.cfg.embedder.layer_idx]
                residue_embeddings = [
                    hidden_states[batch_idx, :s_len].detach().cpu()
                    for batch_idx, s_len in enumerate(seq_lens)
                ]
            else:
                # print(f"Extracting embeddings from the final layer")
                # Extract from last layer
                embedding_repr = self.model(input_ids, attention_mask=attention_mask)
                residue_embeddings = [
                    embedding_repr.last_hidden_state[batch_idx, :s_len].detach().cpu()
                    for batch_idx, s_len in enumerate(seq_lens)
                ]

        return residue_embeddings

    def extract_embeddings(self, all_residue_embeddings: list, seq_lens: list) -> list:
        """Extract embeddings for the central sequence."""
        central_seq_idx = self.cfg.embedder.n_copies // 2
        extracted_embeddings = []

        for residue_embeddings, seq_len in zip(all_residue_embeddings, seq_lens, strict=True):
            start_idx = central_seq_idx * (seq_len + self.chain_break_len)
            stop_idx = start_idx + seq_len
            extracted_embeddings.append(residue_embeddings[start_idx:stop_idx])

        return extracted_embeddings

    def forward(self, sequences: list):
        """Forward pass to extract embeddings from sequences."""
        # Handle sequence duplication if configured
        if self.cfg.embedder.n_copies and self.cfg.embedder.n_copies > 1:
            seq_lens = [len(seq) for seq in sequences]
            sequences = self.duplicate_sequences(sequences)

        # Sort sequences by length for optimal batching
        original_order, sequences = zip(
            *sorted(enumerate(sequences), key=lambda x: len(x[1]), reverse=True),
            strict=False,
        )

        # Process batches and collect embeddings
        residue_embeddings = []
        for batch in self.get_batches(sequences):
            batch_embeddings = self.process_batch(batch)
            residue_embeddings.extend(batch_embeddings)

        # Extract central sequence embeddings if using copies
        if self.cfg.embedder.n_copies and self.cfg.embedder.n_copies > 1:
            residue_embeddings = self.extract_embeddings(residue_embeddings, seq_lens)

        # Restore original sequence order
        residue_embeddings = [
            emb
            for _, emb in sorted(
                zip(original_order, residue_embeddings, strict=False),
                key=lambda x: x[0],
            )
        ]

        # Return mean-pooled or per-residue embeddings
        if self.cfg.embedder.mean_pool:
            seq_embeddings = self.mean_pool_embeddings(residue_embeddings)
            logger.info(f"Mean-pooled embeddings shape: {seq_embeddings.shape}")
            return seq_embeddings

        logger.info(f"Per-residue embeddings count: {len(residue_embeddings)}")
        return residue_embeddings



class OneHot(AAFeaturizerModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        (f"Loading model: {self.cfg.embedder.model_name}")

    @property
    def aa_feature_mapping(self):
        # fmt: off
        aa_list = ['A','C','D','E','F',
                   'G','H','I','K','L',
                   'M','N','P','Q','R',
                   'S','T','V','W','Y']
        # fmt: on
        aa_feature_mapping = {
            aa: [(1 if aa_list[i] == aa else 0) for i in range(len(aa_list))] for aa in aa_list
        }
        return aa_feature_mapping

    def validate_sequences(self, sequences: list) -> None:
        aa_alphabet = set(self.aa_feature_mapping.keys())
        for i, seq in enumerate(sequences):
            for char in seq:
                if char not in aa_alphabet:
                    raise ValueError(
                       f"Sequence {i} contains a character {char} that is not in the alphabet: {aa_alphabet}"
                    )



class ZScale(AAFeaturizerModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        logger.info(f"Loading model: {self.cfg.embedder.model_name}")

    @property
    def aa_feature_mapping(self):
        # fmt: off
        aa_feature_mapping = { 
            'A': [0.24, -2.32, 0.6, -0.14, 1.3],
            'C': [0.84, -1.67, 3.71, 0.18, -2.65],
            'D': [3.98, 0.93, 1.93, -2.46, 0.75],
            'E': [3.11, 0.26, -0.11, -3.04, -0.25],
            'F': [-4.22, 1.94, 1.06, 0.54, -0.62],
            'G': [2.05, -4.06, 0.36, -0.82, -0.38],
            'H': [2.47, 1.95, 0.26, 3.9, 0.09],
            'I': [-3.89, -1.73, -1.71, -0.84, 0.26],
            'K': [2.29, 0.89, -2.49, 1.49, 0.31],
            'L': [-4.28, -1.3, -1.49, -0.72, 0.84],
            'M': [-2.85, -0.22, 0.47, 1.94, -0.98],
            'N': [3.05, 1.62, 1.04, -1.15, 1.61],
            'P': [-1.66, 0.27, 1.84, 0.7, 2.0],
            'Q': [1.75, 0.5, -1.44, -1.34, 0.66],
            'R': [3.52, 2.5, -3.5, 1.99, -0.17],
            'S': [2.39, -1.07, 1.15, -1.39, 0.67],
            'T': [0.75, -2.18, -1.12, -1.46, -0.4],
            'V': [-2.59, -2.64, -1.54, -0.85, -0.02],
            'W': [-4.36, 3.94, 0.59, 3.44, -1.59],
            'Y': [-2.54, 2.44, 0.43, 0.04, -1.47]
        }
        # fmt: on
        return aa_feature_mapping

    def validate_sequences(self, sequences: list) -> None:
        aa_alphabet = set(self.aa_feature_mapping.keys())
        for i, seq in enumerate(sequences):
            for char in seq:
                if char not in aa_alphabet:
                    raise ValueError(
                        f"Sequence {i} contains a character {char} that is not in the alphabet: {aa_alphabet}"
                    )