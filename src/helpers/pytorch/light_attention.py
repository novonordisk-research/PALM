import logging

import lightning as L
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)


class LightAttentionModule(L.LightningModule):
    """Pytorch module from https://github.com/HannesStark/protein-localization, adapted by EHEC/SBXW"""

    def __init__(
        self,
        embeddings_dim: int,
        output_dim: int,
        kernel_size: int = 6,
        dropout: float = 0.25,
        conv_dropout: float = 0.25,
        optimizer_type: str = "sgd",
        learning_rate: float = 2.0e-4,
        post_attention: str = "mlp",
        conv1d_output_dim: int = -1,
        residue_prediction_mode: bool = False,
        reduction_mode="max",
        penalty_weight: float = 0.0,
    ):
        """
        Initializes a new instance of the LightAttention class.

        Args:
            embeddings_dim (int): The hidden dimension of the input embeddings
            output_dim (int): The number of classes in the output
            kernel_size (int): The size of the filter (should be odd) (default = 6, 9 in original LA)
            dropout (float): The dropout rate in the MLP (default = 0.25)
            conv_dropout (float): The dropout rate in convolutional layer (default = 0.25)
            learning_rate (float): The learning rate during training (default = 2.0e-4)
            post_attention (str): The final layers after attention module: 'mlp', 'mlp_deep', 'linear' (default = 'mlp')
            conv1d_output_dim (int): The hidden dimension after conv1d. If -1, is equal to embeddings_dim (default = -1)
            residue_prediction_mode (bool): If true, predicts the label per-residue instead of per sequence (default = false)

        Methods:
            forward: Forward pass of the model.
        """
        super().__init__()
        self.save_hyperparameters()

        if embeddings_dim < 1:
            raise ValueError(f"Embedding dimension cannot be less than 1 ({embeddings_dim})")
        if output_dim < 1:
            raise ValueError(f"Output dimension cannot be less than 1 ({output_dim})")
        if dropout <= 0.0 or dropout >= 1.0:
            raise ValueError(f"Dropout must be in the range [0.0,1.0) ({dropout})")
        if kernel_size % 2 == 0:
            raise ValueError(
                f"Even-sized convolutional filters are not supported ({(kernel_size)})"
            )
        if conv_dropout <= 0.0 or dropout >= 1.0:
            raise ValueError(f"Dropout must be in the range [0.0,1.0) ({conv_dropout})")
        if post_attention not in ["mlp", "mlp_deep", "linear"]:
            raise ValueError(
                f"Provided option for post-attention {post_attention} is not supported"
            )

        self.output_dim = output_dim
        conv1d_output_dim = conv1d_output_dim if conv1d_output_dim > 0 else embeddings_dim
        self.residue_prediction_mode = residue_prediction_mode
        self.reduction_mode = reduction_mode
        self.penalty_weight = penalty_weight

        self.feature_convolution = nn.Conv1d(
            embeddings_dim, conv1d_output_dim, kernel_size, stride=1, padding=kernel_size // 2
        )
        self.attention_convolution = nn.Conv1d(
            embeddings_dim, conv1d_output_dim, kernel_size, stride=1, padding=kernel_size // 2
        )

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(conv_dropout)
        self.sigmoid = nn.Sigmoid()

        concat_factor = 1  # if self.residue_prediction_mode else 2
        if post_attention == "mlp":
            # from LA paper
            self.mlp = nn.Sequential(
                nn.Linear(concat_factor * conv1d_output_dim, 32),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Linear(32, output_dim),
            )
        elif post_attention == "mlp_deep":
            self.mlp = nn.Sequential(
                nn.Linear(concat_factor * conv1d_output_dim, 256),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Linear(32, output_dim),
            )
        else:
            # linear
            self.mlp = nn.Sequential(
                nn.Linear(concat_factor * conv1d_output_dim, output_dim),
            )

        # Configure training
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.loss_fxn = nn.BCELoss()

        # Initialize empty tensors
        self.o = None
        self.attention = None
        self.o_attention = None
        self.o1 = None
        self.o2 = None
        self.o_cat = None
        self.o = None
        self.pred = None
        self.attention_applied = None  # for residue_prediction_mode
        self.o_unflattened = None
        self.o_unflattened_two = None
        self.o_reduced = None
        self.o_unflattened_weighted = None
        self.penalty = 0.0
        self.plm_embeddings = None

    def forward(self, x: torch.Tensor, mask, training=False) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.

        Returns:
            classification: [batch_size] OR [batch_size, sequence_length] tensor with logits
        """
        self.plm_embeddings = x
        self.o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        self.o = self.dropout(self.o)  # [batch_gsize, embeddings_dim, sequence_length]
        self.attention = self.attention_convolution(
            x
        )  # [batch_size, embeddings_dim, sequence_length]

        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lengths).
        # this padding is added by the dataloader when using the padded_permuted_collate function
        self.attention = self.attention.masked_fill(mask[:, None, :] == False, -1e9)

        if self.residue_prediction_mode:
            raise ValueError("not supported")
        else:
            self.attention_applied = self.o * self.softmax(
                self.attention
            )  # [batchsize, embeddings_dim, sequence_length]

            if (
                self.reduction_mode == "weighted_mean_single_conv"
                or self.reduction_mode == "mean_single_conv"
            ):
                self.attention_applied = self.o

            batchsize, sequence_length = (
                self.attention_applied.shape[0],
                self.attention_applied.shape[2],
            )
            self.attention_applied = torch.permute(
                self.attention_applied, (0, 2, 1)
            )  # [batchsize, sequence_length, embeddings_dim]
            self.o_cat = torch.flatten(
                self.attention_applied, start_dim=0, end_dim=1
            )  # [batchsize * sequence_length, embeddings_dim]

        # if residue_prediction_mode: m = batchsize * sequence_length, else: m = batch_size
        self.o = self.mlp(self.o_cat)  # [m, out_dim]

        if self.output_dim > 1:
            self.o = torch.mean(self.o, dim=-1).unsqueeze(dim=-1)

        self.o = self.o.squeeze()  # [m]

        self.o_unflattened = torch.unflatten(
            self.o, dim=0, sizes=(batchsize, sequence_length)
        )  # [batchsize, sequence_length]
        if self.reduction_mode == "max":
            self.o_unflattened = self.sigmoid(self.o_unflattened)
            self.o_unflattened = self.o_unflattened.masked_fill(mask == False, 0.0)
            self.o_reduced, _ = torch.max(self.o_unflattened, dim=-1, keepdim=False)
        elif self.reduction_mode == "mean" or self.reduction_mode == "mean_single_conv":
            self.o_unflattened = self.sigmoid(self.o_unflattened)
            self.o_unflattened = self.o_unflattened.masked_fill(mask == False, float("nan"))
            self.o_reduced = torch.nanmean(self.o_unflattened, dim=-1, keepdim=False)
        elif self.reduction_mode == "sum":
            self.o_unflattened = self.o_unflattened.masked_fill(mask == False, 0.0)
            self.o_reduced = torch.sum(self.o_unflattened, dim=-1, keepdim=False)
            self.o_reduced = self.sigmoid(self.o_reduced)  # sigmoid after to ensure bound to [0,1]
        elif (
            self.reduction_mode == "weighted_mean"
            or self.reduction_mode == "weighted_mean_single_conv"
        ):
            self.o_unflattened = self.sigmoid(self.o_unflattened)
            self.o_unflattened_weighted = self.o_unflattened * self.softmax(
                self.o_unflattened.masked_fill(mask == False, -1e9)
            )
            self.penalty = torch.nanmean(
                self.o_unflattened_weighted.masked_fill(mask == False, float("nan"))
            )
            self.o_reduced = torch.sum(self.o_unflattened_weighted, dim=-1, keepdim=False)
        elif self.reduction_mode == "weighted_mean_two":
            self.o_unflattened_two = self.sigmoid(self.o_unflattened)
            self.o_unflattened_weighted = self.o_unflattened_two * self.softmax(
                self.o_unflattened.masked_fill(mask == False, -1e9)
            )
            self.penalty = torch.nanmean(
                self.o_unflattened_weighted.masked_fill(mask == False, float("nan"))
            )
            self.o_reduced = torch.sum(self.o_unflattened_weighted, dim=-1, keepdim=False)
            self.o_reduced = torch.clamp(self.o_reduced, max=1)
            logger.info(f"TENSOR MAXIMUM {torch.max(self.o_reduced)}")

        self.o_reduced = self.o_reduced  # [m, 1]

        return self.o_reduced  # [batchsize]

    def convert_to_numpy(self, output: torch.tensor, mask: torch.tensor) -> np.ma.masked_array:
        """
        Move predictions to cpu and convert to numpy masked array

        Args:
            output: [batch_size] OR [batch_size, sequence_length] predictions from the model
            mask: [batch_size, sequence_length] residue mask

        Returns:
            output: [batch_size] OR [batch_size, sequence_length] array with logits
        """
        output_numpy = output.cpu().numpy()
        mask_numpy = ~mask.cpu().numpy().astype(bool)
        # mask is false by default, otherwise it denotes padding residues
        output_numpy = (
            np.ma.masked_array(output_numpy, mask_numpy)
            if self.residue_prediction_mode
            else np.ma.masked_array(output_numpy)
        )
        return output_numpy

    def _flatten_output(self, output, mask):
        # flatten and select the residues for which labels are defined
        flattened_mask = torch.flatten(mask).to(dtype=torch.bool)  # [batchsize * sequence_length]
        return torch.masked_select(output, flattened_mask)  # [batchsize * sequence_length]

    def training_step(self, batch):
        x, mask, y = batch
        y_pred = self(x, mask, training=True)
        loss = self.loss_fxn(y_pred, y)
        penalty = self.penalty_weight * self.penalty
        loss += penalty
        self.log(
            "train.penalty", penalty, on_step=False, on_epoch=True, batch_size=batch[0].shape[0]
        )
        self.log("train.loss", loss, on_step=False, on_epoch=True, batch_size=batch[0].shape[0])
        return loss

    def validation_step(self, batch):
        x, mask, y = batch
        y_pred = self(x, mask, training=True)
        loss = self.loss_fxn(y_pred, y)
        penalty = self.penalty_weight * self.penalty
        loss += penalty
        self.log("val.penalty", penalty, on_step=False, on_epoch=True, batch_size=batch[0].shape[0])
        self.log("val.loss", loss, on_step=False, on_epoch=True, batch_size=batch[0].shape[0])
        return loss

    def configure_optimizers(self):
        if self.optimizer_type == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"optimizer not defined: {self.optimizer_type}")


def LA_custom_collate(batch: list[torch.tensor]):
    """
    Takes list of tuples with embeddings of variable sizes and pads them with zeros
    Args:
        batch: list of tuples with embeddings and the corresponding label

    Returns: tuple of tensor of embeddings [batchsize, embeddings_dim, length_of_longest_sequence]
                      tensor of mask [batchsize, length_of_longest_sequence]
                  and tensor of labels [batchsize, labels_dim] or None in inference mode

    """
    # Adapted from https://github.com/HannesStark/protein-localization/blob/7b0be1e64a91db8ad1a8feae994a4d09aa9d7b1b/utils/general.py#L225
    embeddings = [item[0] for item in batch]  # list[tensor] where each tensor is L x H
    padded_embeddings = pad_sequence(embeddings, batch_first=True).permute(
        0, 2, 1
    )  # B x H x T (where T is max(L))
    mask = pad_sequence([torch.ones(item[0].shape[0]) for item in batch], batch_first=True)  # B x T
    labels = torch.tensor([item[1] for item in batch]) if len(batch[0]) == 2 else None  # B x 1
    return padded_embeddings, mask, labels


def LA_custom_collate_residue_level(batch: list[torch.tensor]):
    """
    Takes list of tuples with embeddings of variable sizes and pads them with zeros
    Args:
        batch: list of tuples with embeddings and the corresponding label

    Returns: tuple of tensor of embeddings [batchsize, embeddings_dim, length_of_longest_sequence]
                      tensor of mask [batchsize, length_of_longest_sequence]
                  and tensor of labels [batchsize, labels_dim] or None in inference mode

    """
    # Adapted from https://github.com/HannesStark/protein-localization/blob/7b0be1e64a91db8ad1a8feae994a4d09aa9d7b1b/utils/general.py#L225
    embeddings = [item[0] for item in batch]  # list[tensor] where each tensor is L x H
    padded_embeddings = pad_sequence(embeddings, batch_first=True).permute(
        0, 2, 1
    )  # B x H x T (where T is max(L))
    mask = pad_sequence([torch.ones(item[0].shape[0]) for item in batch], batch_first=True)  # B x T
    labels = torch.cat([item[1] for item in batch]) if len(batch[0]) == 2 else None  # B x 1
    return padded_embeddings, mask, labels
