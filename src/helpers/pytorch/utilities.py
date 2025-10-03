import logging

import numpy as np
import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class EmbeddingsDataset(Dataset):
    def __init__(
        self, embeddings: list[torch.tensor], labels: np.array, max_length: int = 1000
    ) -> None:
        if len(embeddings) != labels.shape[0]:
            raise ValueError(
                f"Mismatched length between embeddings and labels: {len(embeddings)}, {labels.shape[0]}"
            )
        self.embeddings = embeddings
        self.labels = torch.from_numpy(labels).to(dtype=torch.float)
        self.max_length = max_length

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        if len(self.embeddings[index]) > self.max_length:
            raise ValueError(
                f"Sequence with index {index} and length {len(self.embeddings[index])} is longer than the limit {self.max_length}"
            )
        return self.embeddings[index], self.labels[index]

    def __len__(self):
        return len(self.embeddings)

    def embedding_dim(self):
        return self.embeddings[0].shape[-1]  # assume the last dimension is the hidden dimension

    def get_labels(self):
        return self.labels


class EmbeddingsDatasetResidueLevel(Dataset):
    def __init__(
        self, embeddings: list[torch.tensor], labels: np.array, max_length: int = 1000
    ) -> None:
        if len(embeddings) != labels.shape[0]:
            raise ValueError(
                f"Mismatched length between embeddings and labels: {len(embeddings)}, {labels.shape[0]}"
            )
        self.embeddings = embeddings
        self.max_length = max_length
        self.labels = [torch.tensor([float(y) for y in list(x)], dtype=torch.float) for x in labels]

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        if len(self.embeddings[index]) > self.max_length:
            raise ValueError(
                f"Sequence with index {index} and length {len(self.embeddings[index])} is longer than the limit {self.max_length}"
            )
        return self.embeddings[index], self.labels[index]

    def __len__(self):
        return len(self.embeddings)

    def embedding_dim(self):
        return self.embeddings[0].shape[-1]  # assume the last dimension is the hidden dimension

    def get_labels(self):
        return torch.cat(self.labels)
