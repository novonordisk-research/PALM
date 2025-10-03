import logging

import lightning as L  # noqa: N812
import mlflow
import torch
from torch import nn


logger = logging.getLogger(__name__)


class LSTMModule(L.LightningModule):
    """Fake AggreProt roughly developed from https://www.biorxiv.org/content/10.1101/2024.03.06.583680v1, adapted by EHEC"""

    def __init__(
        self,
        embeddings_dim: int,
        output_dim: int = 1,
        dropout: float = 0.25,
        optimizer_type: str = "sgd",
        learning_rate: float = 2.0e-4,
    ):
        """
        Initializes a new instance of the LSTMModule class.

        Args:
            embeddings_dim (int): The hidden dimension of the input embeddings
            output_dim (int): The number of classes in the output
            dropout (float): The dropout rate in the MLP (default = 0.25)
            learning_rate (float): The learning rate during training (default = 2.0e-4)

        Methods:
            forward: Forward pass of the model.
        """
        super().__init__()
        self.save_hyperparameters()

        if embeddings_dim < 1:
            raise ValueError(f"Embedding dimension cannot be less than 1 ({embeddings_dim})")
        if dropout <= 0.0 or dropout >= 1.0:
            raise ValueError(f"Dropout must be in the range [0.0,1.0) ({dropout})")

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.output_dim = output_dim

        self.layers = nn.Sequential(
            nn.Linear(96 * 2 * 6, 32),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

        self.linear1 = nn.Linear(embeddings_dim, 64 * 6)
        self.LSTM1 = nn.LSTM(64 * 6, 64, bidirectional=True, batch_first=True)
        self.LSTM2 = nn.LSTM(64 * 2, 96, bidirectional=True, batch_first=True)

        # Configure training
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.loss_fxn = nn.CrossEntropyLoss() if output_dim > 1 else nn.BCELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, sequence_length (must be 6), embeddings_dim] embedding tensor that should be classified

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """

        o = self.linear1(x)
        o = self.LSTM1(o)[0]
        o = self.LSTM2(o)[0]
        o = torch.flatten(o, start_dim=1, end_dim=-1)
        o = self.layers(o)  # [batchsize, output_dim]
        o = self.sigmoid(o) if self.output_dim == 1 else self.softmax(o)  # [batchsize, output_dim]

        return o

    def training_step(self, batch):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fxn(y_pred, y)
        self.log("train.loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fxn(y_pred, y)
        mlflow.log_metric("val.loss", loss)
        self.log("val.loss", loss, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer_type == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"optimizer not defined: {self.optimizer_type}")


def LSTM_custom_collate(batch: list[torch.tensor]):  # noqa: N802
    """
    Takes list of tuples with embeddings and converts them to tensors
    Args:
        batch: list of tuples with embeddings and the corresponding label

    Returns: tuple of tensor of embeddings [batchsize, sequence length, embeddings_dim]
                  and tensor of labels [batchsize, labels_dim] or None in inference mode

    """
    embeddings = torch.stack(
        [item[0] for item in batch], dim=0
    )  # list[tensor] where each tensor is L x H
    labels = (
        torch.tensor([item[1] for item in batch], dtype=torch.float).reshape(-1, 1)
        if len(batch[0]) == 2  # noqa: PLR2004
        else None
    )  # B x 1
    return embeddings, labels
