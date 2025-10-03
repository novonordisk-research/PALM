import logging
import os
from abc import ABCMeta, abstractmethod
from datetime import timedelta

import lightning as L  # noqa: N812
import mlflow
import numpy as np
import pandas as pd
import sklearn
import torch
from joblib import parallel_backend
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig
from sklearn.metrics import make_scorer, matthews_corrcoef, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from skops.io import dump as sk_dump
from skops.io import load as sk_load
from strenum import StrEnum
from torch.utils.data import DataLoader

from src.helpers.utilities import CudaOutOfMemoryError, nJobs


sklearn.set_config(enable_metadata_routing=True)

"""
All composite models consist of embedder, dimensionality reducer, and predictor modules chained together. 
The methods of these modules are templated by the abstract base classes defined here, while specific implementations 
defined in distinct source files. From a practical perspective, this helps ensure that new modules implement
the required methods for use within composite models.
"""

component_type = StrEnum("component_type", ["embedder", "dimred", "predictor"])
run_mode = StrEnum("run_mode", ["train", "test", "embed"])

logger = logging.getLogger(__name__)


class EmbedderModel(metaclass=ABCMeta):
    """Abstract class that templates all embedder models"""

    # https://peps.python.org/pep-0487/#subclass-registration
    subclass_registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclass_registry[cls.__name__] = cls

    def __init__(self, cfg: DictConfig):
        """
        Initializes a new instance of the EmbedderModel class.

        Args:
            cfg (DictConfig): The configuration for the embedder model.

        Attributes:
            ctype (component_type): The type of component (embedder).
            cfg (DictConfig): The configuration for the model.

        Methods:
            forward: Forward pass of the model.
        """
        logger.info(f"Load class (Embedder Model): {self.__class__.__name__}")
        super().__init__()
        self.ctype = component_type.embedder  # for convenient access
        self.cfg = cfg
        self.scaler = None  # Initialize scaler as None

    def standardize_embeddings(self, embeddings, fit=False):
        """
        Standardize embeddings using sklearn's MinMaxScaler.

        Args:
            embeddings: Can be either:
                - numpy array of shape (n_sequences, n_features) for mean-pooled embeddings or embeddings from biophys,amortport,etc
                - list of torch tensors of shape (seq_len, n_features) for per-residue embeddings
            fit (bool): Whether to fit the scaler on this data. Should be True for training data.

        Returns:
            Standardized embeddings in the same format as input
        """
        if self.scaler is None and fit:
            if self.cfg.embedder.scalar_type == "MinMaxScaler":
                self.scaler = MinMaxScaler()
            elif self.cfg.embedder.scalar_type == "StandardScaler":
                self.scaler = StandardScaler()
            elif self.cfg.embedder.scalar_type == "RobustScaler":
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaler type: {self.cfg.embedder.scaler_type}")

        if isinstance(embeddings, np.ndarray):
            # logger.info(f"Before standardization - min: {embeddings.min()}, max: {embeddings.max()}")
            if fit:
                result = self.scaler.fit_transform(embeddings)
            else:
                result = self.scaler.transform(embeddings)
            # logger.info(f"After standardization - min: {result.min()}, max: {result.max()}")
            return result

        # In future add logic for embeddings along residue when mean_pool is set to false
        else:
            raise ValueError("Embeddings must be either a numpy array or a list of torch tensors")

    def save_scaler(self, path):
        """
        Save the fitted scaler to disk.

        Args:
            path (str or Path): Path to save the scaler
        """
        if self.scaler is not None:
            sk_dump(self.scaler, path)
            logger.info(f"Saved scaler to {path}")

    def load_scaler(self, path):
        """
        Load a previously fitted scaler from disk.

        Args:
            path (str or Path): Path to the saved scaler
        """
        self.scaler = sk_load(path)
        logger.info(f"Loaded scaler from {path}")

    @property
    @abstractmethod
    def forward(self):
        """
        Abstract method for the forward pass of the embedder model.
        """
        pass


class LLMEmbedderModel(EmbedderModel):
    """Abstract class that templates all LLM embedder models"""

    def __init__(self, cfg: DictConfig):
        """
        Initializes a new instance of the LLMEmbedderModel class.

        Args:
            cfg (DictConfig): The configuration for the embedder model.

        Attributes:
            ctype (component_type): The type of component (embedder).
            cfg (DictConfig): The configuration for the model.

        Methods:
            forward: Forward pass of the model.
        """
        logger.info(f"Load class (LLMEmbedderModel): {self.__class__.__name__}")
        super().__init__(cfg)

        self.batch_scale_factors = {
            "esm2_t6_8M_UR50D": (164.216311, 6796.137809, 1.258964e05),
            "esm2_t12_35M_UR50D": (165.052996, 8850.407079, 5.208933e05),
            "esm2_t30_150M_UR50D": (165.341865, 11189.307840, 1.045312e06),
            "esm2_t33_650M_UR50D": (142.660817, 36701.930699, 3.602179e06),
            "esm2_t36_3B_UR50D": (
                142.660817,
                66701.930699,
                6.602179e06,
            ),  # these are estimates
            "esm2_t48_15B_UR50D": (
                142.660817,
                66701.930699,
                6.602179e06,
            )
        }
        self.buffer_scale_factor = self.cfg.embedder.buffer_scale_factor  # Extra memory left unused
        self.max_batch_size = (
            self.cfg.embedder.max_batch_size
        )  # The maximum number of sequences allowed in a batch
        self.strict = (
            self.cfg.embedder.strict
        )  # If True, will throw an error if there is insufficient memory to embed a sequence
        self.verbose = self.cfg.embedder.verbose  # If True, will print info about the batches
        self.simple_batching = (
            self.cfg.embedder.simple_batching
        )  # If True, will batch each sequence separately (very slow)
        self.mean_pool = (
            self.cfg.embedder.mean_pool
        )  # If True, will mean pool the embeddings over the sequence length dimension

    @staticmethod
    def quadratic_mem_util(X: tuple[float], a: float, b: float, c: float):  # noqa: N803 argument name X should be lower case.
        """
        Returns the memory required to embed a batch in bytes

        Args:
            X (tuple[float]): The sequence length and batch size
            a (float): The first embedder-specific constant
            b (float): The second embedder-specific constant
            c (float): The third embedder-specific constant
        """
        x1, x2 = X
        return a * (x1**2) * x2 + b * x1 * x2 + c * x2

    def mem_req_for_batch(self, batch: list[str]):
        """
        Predict the amount of additional memory (in bytes) that will be required to embed the batch

        Args:
            batch (list[str]): List of sequences


        Returns:
            mem_req (float): The bytes required to embed the batch
        """
        batch_size = float(len(batch))
        max_len = float(np.array([len(x) for x in batch]).max())
        return (
            self.quadratic_mem_util(
                (max_len, batch_size), self.factor_1, self.factor_2, self.factor_3
            )
            * self.buffer_scale_factor
        )

    def get_batches(self, sequences: list[str]):
        """
        Generator that batches sequences

        Args:
            sequences (list[str]): List of sequences

        Returns:
            batch (list[str]): A valid batch
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu" and not self.simple_batching:
            logger.info("No GPU found, switching to simple batching.")

        if self.simple_batching:
            for seq in sequences:
                yield [seq]
        else:
            total_mem_avail = torch.cuda.get_device_properties(0).total_memory
            base_mem_allocated = torch.cuda.memory_allocated(0)
            logger.info(
                f"Total memory: {total_mem_avail / 1e9}GB\nMemory already allocated: {base_mem_allocated / 1e9}GB"
            )

            batch = []
            for seq in sequences:
                seq_mem_req = self.mem_req_for_batch([seq])
                if base_mem_allocated + seq_mem_req > total_mem_avail:
                    logger.info(
                        f"Cannot embed sequence of length: {len(seq)}, estimated memory requirement: {seq_mem_req / 1e9:0.2f}GB"
                    )
                    if self.strict:
                        raise CudaOutOfMemoryError("Unable to embed sequence", [len(seq)])

                if len(batch) >= self.max_batch_size or (
                    base_mem_allocated + self.mem_req_for_batch(batch + [seq]) > total_mem_avail
                ):
                    # Sequence doesn't fit in current batch, process current batch, and then try again with an empty batch
                    if self.verbose:
                        self.summarize_batch(batch)
                    yield batch
                    batch = []

                # Add sequence to batch and iterate
                batch.append(seq)

            if self.verbose:
                self.summarize_batch(batch)
            yield batch

    @staticmethod
    def summarize_batch(sequences: list[str]):
        """
        Print the length of each sequence in the batch

        Args:
            sequences (list[str]): List of sequences.

        Returns:
            None
        """
        logger.info(
            f"Batch with {len(sequences)} sequences of length: {[len(x) for x in sequences]}"
        )

    @staticmethod
    def mean_pool_embeddings(residue_embeddings: list[torch.tensor], to_numpy=True):
        """
        Take the mean over the sequence length dimension

        Args:
            residue_embeddings (list[torch.tensor]): List of residue embeddings. b x l x h

        Returns:
            sequence_embeddings (list[torch.tensor]): List of sequence embeddings b x h
        """
        sequence_embeddings = torch.stack([x.mean(dim=0) for x in residue_embeddings])
        return sequence_embeddings.numpy() if to_numpy else sequence_embeddings

    def validate_layer_idx(self) -> None:
        """Validate the layer index against model configuration.

        This method should be called before processing any sequences to ensure
        the specified layer index is valid for the model.

        Raises:
            ValueError: If layer_idx is specified but greater than the model's number of layers
            NotImplementedError: If called from base class (must be implemented by child class)
        """
        if not hasattr(self, "model"):
            raise NotImplementedError(
                "_validate_layer_idx called on base class. Must be implemented by child class."
            )

        if hasattr(self.cfg.embedder, "layer_idx"):
            layer_idx = self.cfg.embedder.layer_idx
            if layer_idx is not None:  # Check if layer_idx is specified
                try:
                    num_layers = self.model.config.num_hidden_layers
                    if layer_idx > num_layers:
                        raise ValueError(
                            f"Layer index {layer_idx} is greater than the number "
                            f"of hidden layers in the model ({num_layers}). "
                            f"Please specify a layer index between 1 and {num_layers}."
                        )
                except AttributeError as e:
                    logger.error(f"Model configuration appears invalid: {e}")
                    raise

    def should_use_specific_layer(self) -> bool:
        """Check if embeddings should be extracted from a specific layer.

        Returns:
            bool: True if a specific layer should be used, False otherwise
        """
        return (
            hasattr(self.cfg.embedder, "output_hidden_states")
            and hasattr(self.cfg.embedder, "layer_idx")
            and self.cfg.embedder.output_hidden_states
            and self.cfg.embedder.layer_idx
            and self.cfg.embedder.layer_idx <= self.model.config.num_hidden_layers
        )


class AAFeaturizerModel(EmbedderModel):
    """Abstract class that templates all amino-acid level featurizers"""

    def __init__(self, cfg: DictConfig):
        """
        Initializes a new instance of the AAFeaturizer class.

        Args:
            cfg (DictConfig): The configuration for the featurizer model.

        Attributes:
            ctype (component_type): The type of component (embedder).
            cfg (DictConfig): The configuration for the model.

        Methods:
            forward: Forward pass of the model.
        """
        logger.info(f"Load class (AAFeaturizer): {self.__class__.__name__}")
        super().__init__(cfg)

        self.mean_pool = (
            self.cfg.embedder.mean_pool
        )  # If True, will mean pool the embeddings over the sequence length dimension

    @property
    @abstractmethod
    def aa_feature_mapping(self):
        """A residue-level mapping to feature vectors"""
        pass

    @abstractmethod
    def validate_sequences(self):
        """To ensure that the sequences do not contain AA outside of the alphabet"""
        pass

    def forward(self, sequences: list) -> np.array:
        self.validate_sequences(sequences)

        sequence_aa_features = [
            torch.tensor([self.aa_feature_mapping[char] for char in seq], dtype=torch.float)
            for seq in sequences
        ]
        if self.mean_pool:
            sequence_features = self.mean_pool_embeddings(sequence_aa_features)
            logger.info(f"Final mean-pooled shape: {sequence_features.shape}")
            return sequence_features
        else:
            logger.info(f"Final results len: {len(sequence_aa_features)}")
            return sequence_aa_features

    @staticmethod
    def mean_pool_embeddings(residue_embeddings: list[torch.tensor], to_numpy=True):
        """
        Take the mean over the sequence length dimension

        Args:
            residue_embeddings (list[torch.tensor]): List of residue embeddings. b x l x h

        Returns:
            sequence_embeddings (list[torch.tensor]): List of sequence embeddings b x h
        """
        sequence_embeddings = torch.stack([x.mean(dim=0) for x in residue_embeddings])
        return sequence_embeddings.numpy() if to_numpy else sequence_embeddings


class DimRedModel(metaclass=ABCMeta):
    """Abstract class that templates all dimensionality reduction classes"""

    # https://peps.python.org/pep-0487/#subclass-registration
    subclass_registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclass_registry[cls.__name__] = cls

    def __init__(self, cfg: DictConfig):
        """
        Initializes a new instance of the DimRedModel class.

        Args:
            cfg (DictConfig): The configuration for the dimensionality reduction model.

        Attributes:
            ctype (component_type): The type of component (dimensionality reduction).
            cfg (DictConfig): The configuration for the model.
            fit (bool): Whether the model is fit.

        Methods:
            forward: Forward pass of the model.
            fit_data: Fit the data to the model.
            save_model: Save the trained model.
        """
        logger.info(f"Load class (DimRed Model): {self.__class__.__name__}")
        super().__init__()
        self.ctype = component_type.dimred  # for convenient access
        self.cfg = cfg
        self.fit = False

    @property
    @abstractmethod
    def forward(self):
        """
        Abstract property representing the forward pass of the dimensionality reduction model.
        """
        pass

    @property
    @abstractmethod
    def fit_data(self):
        """
        Abstract property representing the fitting of data to the dimensionality reduction model.
        """
        pass

    @property
    @abstractmethod
    def save_model(self):
        """
        Abstract property representing the saving of the dimensionality reduction model.
        """
        pass


class PredictorModel(metaclass=ABCMeta):
    """Abstract class that templates all predictor models"""

    # https://peps.python.org/pep-0487/#subclass-registration
    subclass_registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclass_registry[cls.__name__] = cls

    def __init__(self, cfg: DictConfig):
        """
        Initialize the PredictorModel.

        Args:
            cfg (DictConfig): The configuration for the model.

        Attributes:
            ctype (component_type): The type of component (predictor).
            cfg (DictConfig): The configuration for the model.
            trained (bool): Whether the model is trained.

        Methods:
            forward: Forward pass of the model.
            save_model: Save the trained model.
            get_hparams_string: Get a string representation of the hyperparameters.
            get_param_grid: Get the parameter grid for hyperparameter tuning.
        """
        logger.info(f"Load class (Predictor Model): {self.__class__.__name__}")
        super().__init__()
        self.ctype = component_type.predictor  # for convenient access
        self.cfg = cfg
        self.trained = False
        self.hparamsresults_df = None

    @property
    @abstractmethod
    def forward(self):
        """
        Abstract property to get the forward pass of the model.

        """
        pass

    @property
    @abstractmethod
    def train_model(self):
        """
        Abstract property to train the model.

        """
        pass

    @property
    @abstractmethod
    def save_model(self):
        """
        Abstract property to save the trained model.

        """
        pass

    @property
    @abstractmethod
    def get_hparams_string(self) -> str:
        """
        Abstract property to get a string representation of the hyperparameters.

        Returns:
            str: The string representation of the hyperparameters.

        """
        pass

    @property
    @abstractmethod
    def get_param_grid(self) -> dict:
        """
        Abstract property to get the parameter grid for hyperparameter tuning.

        Returns:
            dict: The parameter grid.

        """
        pass


class SKLPredictorModel(PredictorModel):
    """Abstract class that templates all Scikit-learn predictor models"""

    def __init__(self, cfg: DictConfig):
        """
        Initializes a new instance of the SKLPredictorModel class.

        Args:
            cfg (DictConfig): The configuration for the predictor model.

        Attributes:
            cfg (DictConfig): The configuration for the model.

        Methods:
            forward: Forward pass of the model.
        """
        logger.info(f"Load class (SKLPredictorModel): {self.__class__.__name__}")
        super().__init__(cfg)

        self.model_type = cfg.predictor.model_type
        self.class_name = cfg.predictor.class_name
        self.mem_per_job = cfg.predictor.mem_per_job
        self.n_splits = cfg.predictor.hparam_tuning.n_splits

    def train_model(self, x_train, y_train, x_val, y_val, x_sample_weights=None):
        """
        Train a sklearn predictor model
        Args:
            x_train (np.array): The training data (n x h)
            y_train (np.array): The training data labels (n x 1)
            x_val (np.array): The validation data, not used here (n x h)
            y_val (np.array): The validation data labels, not used here (n x 1)
        """

        # Get the model selector
        param_grid = self.get_param_grid()
        logger.info(f"param_grid: {param_grid}")
        logger.info(f"Model type: {self.model_type}")
        if self.model_type == "regression":
            logger.info("Model type = regression, scorer = mean_squared_error")
            scorer = make_scorer(mean_squared_error)
        elif self.model_type == "classification_binary":
            logger.info("Model type = classifier, scorer = matthews_corrcoef")
            scorer = make_scorer(matthews_corrcoef)
        else:
            raise NotImplementedError

        if self.cfg.dataset.use_sample_weights is not None:
            scorer.set_score_request(sample_weight=True)
            self.model.set_fit_request(sample_weight=True)

        # Calculate the max number of jobs that we can run without an OOM error
        logger.info(f"The predictor is {self}")
        if self.model_type == "classifier" and self.class_name != "MLP":
            mem_per_job = (
                # If we don't know how many GB are required, then we assume a large value
                self.mem_per_job if self.mem_per_job else 16.0
            )
            n_jobs = nJobs(mem_per_job)
            logger.info(f"Training model with {n_jobs} workers in parallel")
            param_grid["n_jobs"] = [n_jobs]
            model_selector = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=self.n_splits,
                scoring=scorer,
                error_score="raise",
                verbose=3,
            )
        elif self.model_type == "classifier" and self.class_name == "MLP":
            # use joblib parallel backend for sklearn mlp as it can't take number of jobs as a parameter
            with parallel_backend("multiprocessing", n_jobs=-1):
                model_selector = GridSearchCV(
                    estimator=self.model,
                    param_grid=param_grid,
                    cv=self.n_splits,
                    scoring=scorer,
                    error_score="raise",
                    verbose=3,
                )
        else:
            model_selector = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=self.n_splits,
                scoring=scorer,
                error_score="raise",
                verbose=3,
            )

        # Search the hyperparameter space for the best model w.r.t. the scoring function
        logger.info(f"X_train shape: {x_train.shape}")
        logger.info(f"y_train shape: {y_train.shape}")
        logger.info(f"Parameter grid of different hyperparamenters: {self.get_param_grid()}")
        if x_sample_weights is not None:
            logger.info("Fitting the model with sample weights")

        model_selector.fit(x_train, y_train, sample_weight=x_sample_weights)
        self.model = model_selector.best_estimator_
        self.hparamsresults_df = pd.DataFrame(model_selector.cv_results_)


class TorchPredictorModel(PredictorModel):
    """Abstract class that templates all Pytorch Lightning predictor models"""

    def __init__(self, cfg: DictConfig):
        """
        Initializes a new instance of the TorchPredictorModel class.

        Args:
            cfg (DictConfig): The configuration for the predictor model.

        Attributes:
            cfg (DictConfig): The configuration for the model.

        Methods:
            forward: Forward pass of the model.
        """
        logger.info(f"Load class (TorchPredictorModel): {self.__class__.__name__}")
        super().__init__(cfg)
        self.model_type = cfg.predictor.model_type
        self.max_time = timedelta(days=0, hours=6)
        self.batch_size = cfg.predictor.hparams.batch_size
        self.max_epochs = cfg.predictor.hparams.max_epochs
        self.patience = cfg.predictor.hparams.patience

    def train_model(self, x_train, y_train, x_val, y_val, sample_weights):
        assert (
            x_train is not None and y_train is not None and x_val is not None and y_val is not None
        ), "Missing train/test data"
        train_set = self.dataset(x_train, y_train)
        val_set = self.dataset(x_val, y_val)

        if self.model is None:
            self.init_torch_module(train_set.embedding_dim())

        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        logger.info(f"This is the train laoder {train_loader}")
        logger.info(f"This is the val loader {val_loader}")

        early_stopping = EarlyStopping(monitor="val.loss", patience=self.patience)
        checkpointer = ModelCheckpoint(save_top_k=1, mode="min", monitor="val.loss")
        mlflow_logger = MLFlowLogger(
            experiment_name=mlflow.get_experiment(mlflow.active_run().info.experiment_id).name,
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
            run_id=mlflow.active_run().info.run_id,
            log_model=True,
        )  # prefix argument automatically uses "-" to join, otherwise I would use it

        trainer = L.Trainer(
            callbacks=[early_stopping, checkpointer],
            max_time=self.max_time,
            deterministic=False,
            max_epochs=self.max_epochs,
            enable_progress_bar=False,
            log_every_n_steps=1,
            logger=mlflow_logger,
        )

        mlflow.pytorch.autolog(checkpoint_monitor="val.loss")
        trainer.fit(model=self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        stopped_epoch = early_stopping.stopped_epoch
        best_dir = checkpointer.best_model_path
        best_score = checkpointer.best_model_score
        logger.info(f"MODEL STOPPED BY EARLY STOPPER AT EPOCH {stopped_epoch}")
        logger.info(f"BEST MODEL VALIDATION LOSS IS {best_score}")
        logger.info(f"BEST MODEL SAVED AT {best_dir}")
        logger.info(f"FINAL MODEL SAVED AT {trainer.log_dir}")
        final_dir = "/".join(best_dir.split("/")[:-1]) + "/final.ckpt"
        trainer.save_checkpoint(filepath=final_dir)

        pytorch_model_type = type(self.model)  # Get the torch lightning module class
        self.model = pytorch_model_type.load_from_checkpoint(best_dir)
        self.model.eval()

        # In case the derived class must do things post training
        post_train_model = getattr(self, "post_train_model", None)
        if callable(post_train_model):
            y_val = val_loader.dataset.get_labels()
            post_train_model(x_val, y_val)
