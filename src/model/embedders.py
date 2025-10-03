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

            # Setup device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device == "cpu":
                raise NotImplementedError("Batching is only implemented for GPUs")

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

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

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
        for _, seq in enumerate(sequences):
            for char in seq:
                if char not in aa_alphabet:
                    raise ValueError(
                        "Sequence {i} contains a character {char} that is not in the alphabet: {aa_alphabet}"
                    )
