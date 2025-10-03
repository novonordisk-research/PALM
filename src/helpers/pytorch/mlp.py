import logging

import lightning as L  # noqa: N812
import torch
import torch.nn as nn  # noqa: PLR0402


logger = logging.getLogger(__name__)


class TorchMLPModule(L.LightningModule):
    """Pytorch module adapted from T5EncoderClassificationHead"""

    def __init__(
        self,
        embeddings_dim: int,
        hidden_size: int,
        dropout_rate: float,
        learning_rate: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Configure training
        self.learning_rate = learning_rate
        self.loss_fxn = nn.L1Loss()

        self.dense = nn.Linear(embeddings_dim, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, sequence_length] embedding tensor

        Returns:
            classification: [batch_size] tensor with logits
        """

        hidden_states = self.dropout(x)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)

        return hidden_states.squeeze()

    def training_step(self, batch):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fxn(y_pred, y)
        self.log("train.loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fxn(y_pred, y)
        self.log("val.loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
