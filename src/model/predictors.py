import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, open_dict
from sklearn.linear_model import (
    LogisticRegression as SKLogisticRegression,
)
from sklearn.metrics import matthews_corrcoef
from skops.io import dump as sk_dump
from skops.io import load as sk_load


from src.helpers.pytorch.light_attention import (
    LA_custom_collate,
    LA_custom_collate_residue_level,
    LightAttentionModule,
)
from src.helpers.pytorch.lstm import (
    LSTM_custom_collate,
    LSTMModule,
)
from src.helpers.pytorch.utilities import (
    EmbeddingsDataset,
    EmbeddingsDatasetResidueLevel,
)
from src.helpers.utilities import find_optimal_cutoff
from src.model.abstract_components import (
    SKLPredictorModel,
    TorchPredictorModel,
    run_mode,
)


logger = logging.getLogger(__name__)

###                                         ###
## Begin "predictor" component model section ##
###                                         ###


def validate_predictor(cfg):
    if cfg.predictor.model_type == "classifier":
        if cfg.dataset.data_type == "binary":
            pass
        elif cfg.dataset.data_type == "real-valued" and cfg.dataset.cutoff_value is None:
            raise ValueError(
                "cutoff_value must be defined if real-valued data is provided when training a classifier"
            )
    if cfg.predictor.model_type == "regression":
        pass


class LogisticRegression(SKLPredictorModel):
    """
    A logistic regression predictor model.

    Args:
        cfg (DictConfig): The configuration for the model.

    Attributes:
        model: The logistic regression model.
        trained (bool): Indicates if the model has been trained.

    Methods:
        forward(embeddings:np.array) -> np.array: Forward pass of the logistic regression model.
        save_model(dir_path:Path) -> None: Save the logistic regression model.
        update_predictor_name() -> None: Update the predictor name based on the class name and hyperparameters.
        get_hparams_string() -> str: Get a string representation of the hyperparameters.
        get_param_grid() -> dict: Get the parameter grid for hyperparameter tuning.
        update_config_hparams() -> None: Update the configuration hyperparameters based on the model's current hyperparameters.

    Raises:
        ValueError: If the run mode is not defined.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        validate_predictor(cfg)

        if self.cfg.general.run_mode == run_mode.train:
            if self.cfg.predictor.model_name is None:
                self.model = SKLogisticRegression(random_state=self.cfg.general.random_state)
            else:
                raise ValueError(
                    f"Model name must be null (not {self.cfg.predictor.model_name}) while in training mode"
                )
        elif self.cfg.general.run_mode == run_mode.test:
            if self.cfg.general.composite_model_path:
                model_path = Path(
                    self.cfg.general.composite_model_path,
                    self.cfg.predictor.model_name + ".skops",
                )
            else:
                model_path = Path(self.cfg.predictor.model_name + ".skops")
            self.model = sk_load(model_path, trusted=True)
            self.trained = True
        else:
            raise ValueError(f"run_mode: {self.cfg.general.run_mode} not defined")

    def forward(self, embeddings: np.array) -> np.array:
        """
        Forward pass of the logistic regression model.

        Args:
            embeddings (np.array): The input embeddings.

        Returns:
            np.array: The predicted labels.

        """
        return (
            self.model.predict(embeddings),
            self.model.predict_proba(embeddings)[:, -1],
        )

    def save_model(
        self,
        dir_path: Path,
    ) -> str:
        """
        Save the logistic regression model.

        Args:
            dir_path (Path): The directory path to save the model.

        Returns:
            model_path (str): The path to where the model was saved
        """
        model_path = Path(dir_path, self.cfg.predictor.model_name + ".skops")
        sk_dump(self.model, model_path)
        return model_path

    def update_predictor_name(self) -> None:
        """Update the predictor name based on the class name and hyperparameters."""
        self.cfg.predictor.model_name = f"{self.cfg.dataset.data_name}{self.cfg.embedder.model_name}{self.cfg.predictor.class_name}{self.get_hparams_string()}"

    def get_hparams_string(self) -> str:
        """
        Get a string representation of the hyperparameters.

        Returns:
            str: The hyperparameters string.

        """
        c_val = self.cfg.predictor.hparams.C if self.cfg.predictor.hparams.C is not None else np.nan
        penalty = (
            self.cfg.predictor.hparams.penalty
            if self.cfg.predictor.hparams.penalty is not None
            else np.nan
        )
        solver = (
            self.cfg.predictor.hparams.solver
            if self.cfg.predictor.hparams.solver is not None
            else np.nan
        )
        class_weight = (
            self.cfg.predictor.hparams.class_weight
            if self.cfg.predictor.hparams.class_weight is not None
            else ""
        )
        hparams_string = f"-c:{c_val}-penalty:{penalty}-solver:{solver}-classweight:{class_weight}"
        return hparams_string

    def get_param_grid(self) -> dict:
        """
        Get the parameter grid for hyperparameter tuning.

        Returns:
            dict: The parameter grid.

        """
        param_grid = {
            "C": list(self.cfg.predictor.hparam_tuning.C_values),
            "penalty": list(self.cfg.predictor.hparam_tuning.penalties),
            "solver": [self.cfg.predictor.hparam_tuning.solver],
            "random_state": [self.cfg.general.random_state],
            "max_iter": [self.cfg.predictor.hparam_tuning.max_iter],
            "class_weight": [self.cfg.predictor.hparam_tuning.class_weight],
        }
        return param_grid

    def update_config_hparams(self) -> None:
        """Update the configuration hyperparameters based on the model's current hyperparameters."""
        updated_params = self.model.get_params()
        for param_name in self.cfg.predictor.hparams:
            try:
                self.cfg.predictor.hparams[param_name] = updated_params[param_name]
            except KeyError:
                logger.info(
                    {
                        f"Invalid param_name {param_name} not present in the current list of hyperparameters"
                    }
                )
                
class TorchMLP(TorchPredictorModel):
    def __init__(self, cfg: DictConfig):
        logger.info(f"Load class (TorchMLP): {self.__class__.__name__}")
        super().__init__(cfg)

        self.dataset = EmbeddingsDataset
        self.collate_fn = None  # use torch.utils.data.DataLoader default

        validate_predictor(cfg)
        # print(self.__dict__.keys())
        if self.cfg.general.run_mode == run_mode.train:
            if self.cfg.predictor.model_name is None:
                if cfg.predictor.model_type == "regression":
                    # Lazy load (until we know the embedding dimension)
                    self.model = None
                elif cfg.predictor.model_type == "classification_binary":
                    raise NotImplementedError("Classification not implemented for TorchMLP")
                else:
                    raise ValueError(
                        f'Only model_type "regression" is supported,'  # noqa: F541
                        'not "{self.cfg.predictor.model_type}"'
                    )
            else:
                raise ValueError(
                    f"Model name must be null (not {self.cfg.predictor.model_name}) while in training mode"
                )
        elif self.cfg.general.run_mode == run_mode.test:
            if self.cfg.general.composite_model_path:
                model_path = Path(
                    self.cfg.general.composite_model_path,
                    self.cfg.predictor.model_name + ".pt",
                )
            else:
                model_path = Path(self.cfg.predictor.model_name + ".pt")
            logger.info(f"Loading model from {model_path}")
            self.model = torch.load(model_path)
            self.trained = True
            self.model.eval()
        else:
            raise ValueError(f"run_mode: {self.cfg.general.run_mode} not defined")

    def init_torch_module(self, embedding_dim: int):
        self.model = TorchMLPModule(
            embedding_dim,
            self.cfg.predictor.hparams.hidden_size,
            self.cfg.predictor.hparams.dropout_rate,
            self.cfg.predictor.hparams.learning_rate,
        )

    def forward(self, embeddings):
        """
        Performs forward pass on the TorchMLP model.

        Args:
            embeddings (list[torch.tensor | np.array]): The input embeddings.

        Returns:
            np.array: The predicted values.
            np.array: Predicted probabilites (None for regression)
        """
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).to(torch.float)

        logger.info(f"Moving embeddings to device: {device}")
        embeddings = embeddings.to(device)
        with torch.inference_mode():
            predictions = self.model.forward(embeddings).cpu().numpy()
        return predictions, None

    def save_model(self, dir_path):
        """
        Saves the model.

        Args:
            dir_path (Path): The directory path to save the model.

        Returns:
            model_path (str): The path to where the model was saved
        """
        model_path = Path(dir_path, self.cfg.predictor.model_name + ".pt")
        torch.save(self.model, model_path)
        return model_path

    def update_predictor_name(self) -> None:
        """Updates the predictor model name according to the hparams selected during training"""
        self.cfg.predictor.model_name = "torchmlp_" + self.get_hparams_string()

    def get_hparams_string(self) -> str:
        """
        Get a string representation of the hyperparameters.

        Returns:
            str: The string representation of the hyperparameters.

        """
        return "_".join([f"{k}_{v}" for k, v in dict(self.cfg.predictor.hparams).items()])

    def get_param_grid(self) -> dict:
        """
        Get the parameter grid for hyperparameter tuning.

        Returns:
            dict: The parameter grid.

        """
        return dict()

    def update_config_hparams(self) -> None:
        """
        Updates the configuration hyperparameters based on the model's current parameters.
        """
        pass

class LightAttention(TorchPredictorModel):
    def __init__(self, cfg: DictConfig):
        """
        Initialize the LightAttention predictor.

        Args:
            cfg (DictConfig): The configuration for the model.

        Attributes:
            model: The KNearestNeighbors model.
            trained (bool): Indicates if the model has been trained.

        Methods:
            forward(embeddings:np.array) -> np.array: Forward pass of the LightAttention model.
            save_model(dir_path:Path) -> None: Save the LightAttention model.
            update_predictor_name() -> None: Update the LightAttention name based on the class name and hyperparameters.
            get_hparams_string() -> str: Get a string representation of the hyperparameters.
            get_param_grid() -> dict: Get the parameter grid for hyperparameter tuning.
            update_config_hparams() -> None: Update the configuration hyperparameters based on the model's current hyperparameters.
        """
        logger.info(f"Load class (LightAttention): {self.__class__.__name__}")
        super().__init__(cfg)

        self.residue_prediction_mode = cfg.predictor.residue_prediction_mode
        self.dataset = (
            EmbeddingsDataset
            if not self.cfg.predictor.residue_prediction_mode
            else EmbeddingsDatasetResidueLevel
        )
        self.collate_fn = (
            LA_custom_collate
            if not self.cfg.predictor.residue_prediction_mode
            else LA_custom_collate_residue_level
        )

        validate_predictor(cfg)
        # print(self.__dict__.keys())
        if self.cfg.general.run_mode == run_mode.train:
            self.optimal_cutoff = None
            if self.cfg.predictor.model_name is None:
                if cfg.predictor.model_type == "regression":
                    raise NotImplementedError("Regression not implemented for LightAttention")
                elif cfg.predictor.model_type == "classification_binary":
                    # Lazy load (once we know the embedding dimension)
                    self.model = None
                else:
                    raise ValueError(
                        'Only model_type "classification_binary" is supported,'
                        'not "{self.cfg.predictor.model_type}"'
                    )
            else:
                raise ValueError(
                    f"Model name must be null (not {self.cfg.predictor.model_name}) while in training mode"
                )
        elif self.cfg.general.run_mode == run_mode.test:
            if not self.cfg.predictor.hparams.optimal_cutoff:
                raise ValueError("Optimal cutoff must be defined in test mode")
            self.optimal_cutoff = self.cfg.predictor.hparams.optimal_cutoff
            if self.cfg.general.composite_model_path:
                model_path = Path(
                    self.cfg.general.composite_model_path,
                    "model_state_dict.pt",
                )
            else:
                model_path = Path("model_state_dict.pt")
            logger.info(f"Loading model from {model_path}")
            emb_dims = {
                "esm2_t6_8M_UR50D":320,
                "onehot":20
            }
            self.init_torch_module(emb_dims[self.cfg.embedder.model_name])
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.model.load_state_dict(torch.load(model_path,map_location=device))
            self.trained = True
            self.model.eval()
        else:
            raise ValueError(f"run_mode: {self.cfg.general.run_mode} not defined")

    def init_torch_module(self, embedding_dim: int):
        self.model = LightAttentionModule(
            embedding_dim,
            self.cfg.predictor.hparams.output_dim,
            self.cfg.predictor.hparams.kernel_size,
            self.cfg.predictor.hparams.dropout,
            self.cfg.predictor.hparams.conv_dropout,
            self.cfg.predictor.hparams.optimizer_type,
            self.cfg.predictor.hparams.learning_rate,
            self.cfg.predictor.hparams.post_attention,
            self.cfg.predictor.hparams.conv1d_output_dim,
            self.cfg.predictor.residue_prediction_mode,
            self.cfg.predictor.hparams.reduction_mode,
            self.cfg.predictor.hparams.penalty_weight,
        )

    def forward(self, embeddings):
        """
        Performs forward pass on the LightAttention model.

        Args:
            embeddings (list[torch.tensor]): The input embeddings.

        Returns:
            np.array: The predicted values.
            np.array: Predicted probabilites (None for regression)
        """
        padded_embeddings, mask, _ = self.collate_fn([[x] for x in embeddings])
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logger.info(f"Moving embeddings to device: {device}")
        padded_embeddings, mask = padded_embeddings.to(device), mask.to(device)
        if self.model_type == "classification_binary":
            with torch.inference_mode():
                predicted_probabilities = self.model.forward(padded_embeddings, mask)
                predicted_probabilities = self.model.convert_to_numpy(predicted_probabilities, mask)
            predictions = (
                predicted_probabilities >= self.optimal_cutoff if self.optimal_cutoff else None
            )
        else:
            raise NotImplementedError

        return predictions, predicted_probabilities

    def post_train_model(self, x_val, y_val):
        """
        After training, find the optimal cutoff for classification.

        """
        _, predicted_probabilities = self.forward(x_val)
        if self.residue_prediction_mode:
            # flatten the array and select the non-padding positions
            predicted_probabilities = predicted_probabilities.compressed()
        self.optimal_cutoff, mcc_val = find_optimal_cutoff(y_val, predicted_probabilities)
        with open_dict(self.cfg):
            self.cfg.predictor.hparams.optimal_cutoff = self.optimal_cutoff
        logger.info(f"Optimal cutoff is: {self.optimal_cutoff} with MCC: {mcc_val}")

    def save_model(self, dir_path):
        """
        Saves the model.

        Args:
            dir_path (Path): The directory path to save the model.

        Returns:
            model_path (str): The path to where the model was saved
        """
        model_path = Path(dir_path, "model.pt")
        torch.save(self.model, model_path)
        return model_path

    def update_predictor_name(self) -> None:
        """Updates the predictor model name according to the hparams selected during training"""
        self.cfg.predictor.model_name = "lightattention_" + self.get_hparams_string()

    def get_hparams_string(self) -> str:
        """
        Get a string representation of the hyperparameters.

        Returns:
            str: The string representation of the hyperparameters.

        """
        return "_".join([f"{k}_{v}" for k, v in dict(self.cfg.predictor.hparams).items()])

    def get_param_grid(self) -> dict:
        """
        Get the parameter grid for hyperparameter tuning.

        Returns:
            dict: The parameter grid.

        """
        return dict()

    def update_config_hparams(self) -> None:
        """
        Updates the configuration hyperparameters based on the model's current parameters.
        """
        pass


class LSTM(TorchPredictorModel):
    def __init__(self, cfg: DictConfig):
        """
        Initialize the LSTM predictor.

        Args:
            cfg (DictConfig): The configuration for the model.

        Attributes:
            model: The LSTM model.
            trained (bool): Indicates if the model has been trained.

        Methods:
            forward(embeddings:np.array) -> np.array: Forward pass of the LSTM model.
            save_model(dir_path:Path) -> None: Save the LSTM model.
            update_predictor_name() -> None: Update the LSTM name based on the class name and hyperparameters.
            get_hparams_string() -> str: Get a string representation of the hyperparameters.
            get_param_grid() -> dict: Get the parameter grid for hyperparameter tuning.
            update_config_hparams() -> None: Update the configuration hyperparameters based on the model's current hyperparameters.
        """
        logger.info(f"Load class (LSTM): {self.__class__.__name__}")
        super().__init__(cfg)

        self.dataset = EmbeddingsDataset
        self.collate_fn = LSTM_custom_collate

        validate_predictor(cfg)
        # print(self.__dict__.keys())
        if self.cfg.general.run_mode == run_mode.train:
            self.optimal_cutoff = None
            if self.cfg.predictor.model_name is None:
                if cfg.predictor.model_type == "regression":
                    raise NotImplementedError("Regression not implemented for LightAttention")
                elif cfg.predictor.model_type == "classification_binary":
                    # Lazy load (once we know the embedding dimension)
                    self.model = None
                else:
                    raise ValueError(
                        'Only model_type "classification_binary" is supported,'
                        'not "{self.cfg.predictor.model_type}"'
                    )
            else:
                raise ValueError(
                    f"Model name must be null (not {self.cfg.predictor.model_name}) while in training mode"
                )
        elif self.cfg.general.run_mode == run_mode.test:
            if not self.cfg.predictor.hparams.optimal_cutoff:
                raise ValueError("Optimal cutoff must be defined in test mode")
            self.optimal_cutoff = self.cfg.predictor.hparams.optimal_cutoff
            if self.cfg.general.composite_model_path:
                model_path = Path(
                    self.cfg.general.composite_model_path,
                    self.cfg.predictor.model_name + ".pt",
                )
            else:
                model_path = Path(self.cfg.predictor.model_name + ".pt")
            logger.info(f"Loading model from {model_path}")
            self.model = torch.load(model_path)
            self.trained = True
            self.model.eval()
        else:
            raise ValueError(f"run_mode: {self.cfg.general.run_mode} not defined")

    def init_torch_module(self, embedding_dim: int):
        self.model = LSTMModule(
            embedding_dim,
            1,
            self.cfg.predictor.hparams.dropout,
            self.cfg.predictor.hparams.optimizer_type,
            self.cfg.predictor.hparams.learning_rate,
        )

    def forward(self, embeddings):
        """
        Performs forward pass on the LSTM model.

        Args:
            embeddings (list[torch.tensor]): The input embeddings.

        Returns:
            np.array: The predicted values.
            np.array: Predicted probabilites (None for regression)
        """
        embeddings, _ = self.collate_fn([[x] for x in embeddings])
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logger.info(f"Moving embeddings to device: {device}")
        embeddings = embeddings.to(device)
        if self.model_type == "classification_binary":
            with torch.inference_mode():
                predicted_probabilities = self.model.forward(embeddings).cpu().numpy().squeeze()
            if self.optimal_cutoff:
                predictions = predicted_probabilities >= self.optimal_cutoff
            else:
                predictions = None
        else:
            raise NotImplementedError

        return predictions, predicted_probabilities

    def post_train_model(self, x_val, y_val):
        """
        After training, find the optimal cutoff for classification.

        """
        _, predicted_probabilities = self.forward(x_val)

        # scan over cutoffs with a step size of 0.01
        cutoffs = np.linspace(0, 1, 101)
        thresholded_probabilities = predicted_probabilities > np.expand_dims(cutoffs, axis=-1)

        mcc_df = pd.DataFrame(
            {"cutoffs": cutoffs, "predictions": [x for x in thresholded_probabilities]}
        )
        mcc_df["mcc"] = mcc_df.apply(lambda x: matthews_corrcoef(y_val, x.predictions), axis=1)
        self.optimal_cutoff = float(mcc_df.iloc[mcc_df.mcc.idxmax()].cutoffs)

        with open_dict(self.cfg):
            self.cfg.predictor.hparams.optimal_cutoff = self.optimal_cutoff

        logger.info(f"Optimal cutoff is: {self.optimal_cutoff} with MCC: {mcc_df.mcc.max()}")

    def save_model(self, dir_path):
        """
        Saves the model.

        Args:
            dir_path (Path): The directory path to save the model.

        Returns:
            model_path (str): The path to where the model was saved
        """
        model_path = Path(dir_path, self.cfg.predictor.model_name + ".pt")
        torch.save(self.model, model_path)
        return model_path

    def update_predictor_name(self) -> None:
        """Updates the predictor model name according to the hparams selected during training"""
        self.cfg.predictor.model_name = "lstm_" + self.get_hparams_string()

    def get_hparams_string(self) -> str:
        """
        Get a string representation of the hyperparameters.

        Returns:
            str: The string representation of the hyperparameters.

        """
        return "_".join([f"{k}_{v}" for k, v in dict(self.cfg.predictor.hparams).items()])

    def get_param_grid(self) -> dict:
        """
        Get the parameter grid for hyperparameter tuning.

        Returns:
            dict: The parameter grid.

        """
        return dict()

    def update_config_hparams(self) -> None:
        """
        Updates the configuration hyperparameters based on the model's current parameters.
        """
        pass
