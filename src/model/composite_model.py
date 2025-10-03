import logging
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf, open_dict

from src.helpers.dataset import CSVDataLoader
from src.helpers.io import create_folder_if_not_exists
from src.model.abstract_components import (
    DimRedModel,
    EmbedderModel,
    PredictorModel,
    component_type,
)
from src.model.scalers import ScalerWrap



logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Dataclass for organizing the output and intermediate values from the model"""

    sequences: np.array
    sequence_embeddings: np.array
    additional_features_untransformed: pd.DataFrame
    features: np.array
    features_concat_dimred_concat: np.array
    predictions: np.ma.masked_array
    predictions_scaled: np.array  # TODO: convert to masked array?
    data_split: str
    labels: np.array = None  # Labels only available during training
    group_names: list[str] = None
    predictions_probability: np.ma.masked_array = None  # Not all methods provide a probability
    residue_level_prediction: bool = False

    def get_dataframe(self):
        if self.sequences.shape[0] == 0:
            return pd.DataFrame()
        # create the sequence-level table
        df = pd.DataFrame(
            {
                "sequence": self.sequences,
                "data_split": self.data_split,
                "group": self.group_names if self.group_names is not None else "",
            }
        )
        df = (
            pd.concat(
                [self.additional_features_untransformed.reset_index(drop=True), df],
                axis=1,
            )
            if self.additional_features_untransformed is not None
            else df
        )

        # expand to residue-level table, if applicable
        if self.residue_level_prediction:
            df["sequence_length_aa"] = df["sequence"].apply(len)
            explode_length = df["sequence_length_aa"].sum()
            df["res_aa"] = df["sequence"].apply(list)
            df["res_aa_idx"] = df["sequence"].apply(lambda x: range(len(x)))
            df["res_value_bool"] = (
                [[int(y) for y in list(x)] for x in self.labels]
                if self.labels is not None
                else None
            )
            df = df.explode(["res_aa", "res_aa_idx", "res_value_bool"]).reset_index(
                names="sequence_idx"
            )
            if len(df) != explode_length:
                raise ValueError(
                    f"Mismatched number of rows in residue table. Expected {explode_length} and found {len(df)}"
                )

            # add the labels and predictions
            df["y"] = np.concatenate(
                [np.array([float(y) for y in list(x)], dtype=float) for x in self.labels]
            )
            df["y_pred"] = self.predictions.compressed()
            df["y_pred_prob"] = self.predictions_probability.compressed()

        else:
            # all the labels and predictions are already flattened
            df["y"] = self.labels
            df["y_pred"] = self.predictions
            df["y_pred_prob"] = (
                self.predictions_probability if self.predictions_probability is not None else np.nan
            )

            if self.additional_features_untransformed is not None:
                df = pd.concat(
                    [self.additional_features_untransformed.reset_index(drop=True), df],
                    axis=1,
                )

        return df

    def describe_data(self):
        for name, val in vars(self).items():
            if not name.startswith("__") and val is not None:
                if val is not None and type(val) not in [str, bool]:
                    shape = getattr(val, "shape", None)
                    shape = shape if shape else len(val)
                    print(f"{name}: {shape}")
                else:
                    print(f"{name}: {val}")


class CompositeModel:
    """Modular class for generating embedding features with large self-supervised models and making predictions with small supervised models trained on small datasets"""

    def __init__(self, cfg: DictConfig, inference_only: bool = False) -> None:
        """
        Initializes a CompositeModel object.

        Args:
            cfg (DictConfig): The configuration for the model.

        Attributes:
            cfg (DictConfig): The configuration for the model.
            embed_hdf5 (h5py.File): The HDF5 file for the embeddings.
            emb_class (type): The class for the embedders.
            dataloader (CSVDataloader): The object that handles the data.

        Methods:
            loadEmbedder() -> None: Method for loading the embedder class.
            embedSequences(sequences: list) -> np.array: Method for converting sequences into embeddings.
            reduceDimensionality(embeddings: np.array) -> np.array: Method for reducing dimensionality of vector embeddings.
            predictProperties(embeddings: np.array) -> np.array: Method for converting vector embeddings into property predictions.
            forward(x: pd.DataFrame) -> np.array: All of the modules: 1) embedding, 2) dimensionality reduction, and 3) prediction chained together.
            trainPredictorModel(X_train=None, X_embed_train=None, y_train=None) -> None: Trains the predictor model.
            getModelConfig() -> DictConfig: Returns the configuration of the model.
            getModelName() -> str: Creates a name for the model that captures all of the defining aspects.
            trainingComplete() -> bool: Checks if the training is complete.
            storeModel() -> None: Stores the model.
        """
        self.cfg = cfg
        self.hparamsresults_df = None

        self.embed_hdf5 = None

        # determine class type according to the hydra config and call constructors
        self.emb_class = EmbedderModel.subclass_registry[cfg[component_type.embedder].class_name]
        self.embedder = None  # only load the embedder class if necessary
        dimred_class = DimRedModel.subclass_registry[cfg[component_type.dimred].class_name]
        self.dimensionreducer = dimred_class(self.cfg)
        pred_class = PredictorModel.subclass_registry[cfg[component_type.predictor].class_name]
        self.predictor = pred_class(self.cfg)

        # ensure that the configs are properly initialized
        self.cfg.general.composite_model_name = self.get_model_name()
        logger.info(
            f"Initialized model: {self.cfg.general.composite_model_name} in run_mode = {self.cfg.general.run_mode}"
        )
        if not inference_only:
            self.dataloader = CSVDataLoader(cfg)
        else:
            self.dataloader = None
            logger.info("Running in inference-only mode - skipping dataset loading")
        self.target_scaler = ScalerWrap(self.cfg, name="targetscaler")

    def load_embedder(self):
        """
        Method for loading the embedder class.
        """
        if self.embedder is None:
            self.embedder = self.emb_class(self.cfg)
        else:
            raise ValueError("The embedder is already loaded")

    def embed_sequences(self, sequences: list) -> np.array:
        """
        Method for converting sequences into embeddings.

        Args:
            sequences (list): The list of sequences to be embedded.

        Returns:
            np.array: The embeddings of the sequences.
        """
        if self.embedder is None:
            self.load_embedder()
        embeddings = self.embedder.forward(sequences)
        # logger.info(f"These are the embeddings before standardization{embeddings}")
        if self.cfg.embedder.standardize:
            # Determine if we should fit the scaler (only during training)
            fit = self.cfg.general.run_mode == "train"
            # logger.info(f"This is the embeddings before standardization {embeddings}") #sanity check
            embeddings = self.embedder.standardize_embeddings(embeddings, fit=fit)
            # logger.info(f"This is the embeddings after standardization {embeddings}") #sanity check
            logger.info("Standardizing the embeddings")

        return embeddings

    def reduce_dimensionality(
        self,
        embeddings: np.array,
    ) -> np.array:
        """
        Method for reducing dimensionality of vector embeddings.

        Args:
            embeddings (np.array): The vector embeddings to be dimensionality reduced.

        Returns:
            np.array: The dimensionality reduced vector embeddings.
        """
        embeddings_compressed = self.dimensionreducer.forward(embeddings.copy())
        return embeddings_compressed

    def predict_properties(self, embeddings: np.array) -> np.array:
        """
        Method for converting dimensionality reduced vector embeddings into property predictions.

        Args:
            embeddings (np.array): The dimensionality reduced vector embeddings.

            np.array: The property predictions.
        """
        predictions, predictions_probability = self.predictor.forward(embeddings)
        return predictions, predictions_probability

    def forward(self, dataloader: CSVDataLoader = None, split: str = "test") -> np.array:
        """
        All of the processing methods: 1) embedding, 2) dimensionality reduction, and 3) prediction chained together.

        Args:
            dataloader (CSVDataLoader): The object storing the data
            split (str): The name of the selected split

        Returns:
            Prediction: Dataclass containing output and intermediate values from the model.
        """
        dataloader = self.dataloader if dataloader is None else dataloader

        # Get the data
        sequences = dataloader.get_sequence_data()[split]
        features = dataloader.get_additional_data(apply_dimred=True)[split]  # n x f
        features_nodimred = dataloader.get_additional_data(apply_dimred=False)[split]  # n x g
        labels = dataloader.get_training_labels()[split]

        group_names = dataloader.get_group_names()
        group_names = group_names[split] if group_names is not None else None

        # Embed the sequences
        sequences_embedded = self.embed_sequences(sequences)  # n x h

        predictions_scaled = None
        predictions_probability = None
        if isinstance(sequences_embedded, np.ndarray):
            logger.info("N x H embeddings (mean-pooled or length-invariant featurizer)")

            # Get the additional features to be included in dimensionality reduction and concat
            features_concat = (
                np.concatenate([sequences_embedded, features], axis=-1)
                if features is not None
                else sequences_embedded
            )  # n x (h + f)

            # Perform PCA with train data and reduce feature dimension
            # The new hidden dimension size is less than or equal to the previous one: h_1 <= (h + f)
            features_concat_dimred = self.reduce_dimensionality(features_concat)  # n x h_1

            # Get the additional data values were not included in dimensionality reduction and concat
            features_concat_dimred_concat = (
                np.concatenate([features_concat_dimred, features_nodimred], axis=-1)
                if features_nodimred is not None
                else features_concat_dimred
            )  # n x (h_1 + g)

        # else:
        logger.info("Residue embeddings")
        # If the sequence features/embeddings have not been reduced by mean-pooling, pass directly to the downstream model (no PCA)
        # TODO: decide how to handle additional data columns when training models with residue embeddings

        features_concat, features_concat_dimred, features_concat_dimred_concat = (
            None,
            None,
            None,
        )
        predictions_scaled, predictions_probability = self.predictor.forward(sequences_embedded)

        # rescaled predictions:
        predictions = self.target_scaler.restore_values(predictions_scaled)

        prediction = Prediction(
            sequences,
            sequences_embedded,
            dataloader.get_additional_data_untransformed()[split],
            features,
            features_concat_dimred_concat,
            predictions,
            predictions_scaled,
            split,
            labels,
            group_names,
            predictions_probability,  
            residue_level_prediction=self.cfg.predictor.residue_prediction_mode,
        )
        return prediction

    def train_predictor_model(self) -> None:  # noqa: PLR0912, PLR0915
        """
        Trains the predictor model.

        Returns:
            tuple[Prediction]: Dataclasses containing output and intermediate values from the model train/validation splits.
        """

        # Get the data
        sequences = self.dataloader.get_sequence_data()
        features = self.dataloader.get_additional_data(apply_dimred=True)  # n x f
        features_raw = self.dataloader.get_additional_data_untransformed()
        features_nodimred = self.dataloader.get_additional_data(apply_dimred=False)  # n x g
        labels = self.dataloader.get_training_labels()
        group_names = self.dataloader.get_group_names()
        sample_weights = self.dataloader.get_sample_weights()

        # Embed the sequences
        sequences_embedded_train = self.embed_sequences(sequences["train"])  # n x h

        if isinstance(sequences_embedded_train, np.ndarray):
            logger.info("N x H embeddings (mean-pooled or length-invariant featurizer)")

            # Get the additional features to be included in dimensionality reduction and concat
            features_concat_train = (
                np.concatenate([sequences_embedded_train, features["train"]], axis=-1)
                if features["train"] is not None
                else sequences_embedded_train
            )  # n x (h + f)

            # Perform PCA with train data and reduce feature dimension
            # The new hidden dimension size is less than or equal to the previous one: h_1 <= (h + f)
            self.dimensionreducer.fit_data(features_concat_train)
            features_concat_dimred_train = self.reduce_dimensionality(
                features_concat_train
            )  # n x h_1

            # Get the additional data values were not included in dimensionality reduction and concat
            features_concat_dimred_concat_train = (
                np.concatenate([features_concat_dimred_train, features_nodimred["train"]], axis=-1)
                if features_nodimred["train"] is not None
                else features_concat_dimred_train
            )  # n x (h_1 + g)

            # If there is a validation split, process that too
            if len(sequences["val"]) > 0:
                sequences_embedded_val = self.embed_sequences(sequences["val"])  # n x h
                features_concat_val = (
                    np.concatenate([sequences_embedded_val, features["val"]], axis=-1)
                    if features["val"] is not None
                    else sequences_embedded_val
                )  # n x (h + f)
                features_concat_dimred_val = self.reduce_dimensionality(
                    features_concat_val
                )  # n x h_1
                features_concat_dimred_concat_val = (
                    np.concatenate([features_concat_dimred_val, features_nodimred["val"]], axis=-1)
                    if features_nodimred["val"] is not None
                    else features_concat_dimred_val
                )  # n x (h_1 + g)
            else:
                (
                    sequences_embedded_val,
                    features_concat_val,
                    features_concat_dimred_val,
                    features_concat_dimred_concat_val,
                ) = None, None, None, None

            # Process labels/targets.
            self.target_scaler.fit_scaler(labels["train"])
            labels_scaled = {
                split: self.target_scaler.transform_values(labels[split]) for split in labels.keys()
            }

            # Fit the predictor model
            self.predictor.train_model(
                features_concat_dimred_concat_train,
                labels_scaled["train"],
                features_concat_dimred_concat_val,
                labels_scaled["val"],
                sample_weights,
            )
            predictions_train_scaled, predictions_probability_train = self.predictor.forward(
                features_concat_dimred_concat_train
            )
            if features_concat_dimred_concat_val is not None:
                predictions_val_scaled, predictions_probability_val = self.predictor.forward(
                    features_concat_dimred_concat_val
                )
            else:
                predictions_val_scaled, predictions_probability_val = None, None
            train_intervals = None
            val_intervals = None

            # Predict output with trained model
            predictions_train_scaled, predictions_probability_train = self.predictor.forward(
                features_concat_dimred_concat_train
            )
            predictions_val_scaled, predictions_probability_val = (
                self.predictor.forward(features_concat_dimred_concat_val)
                if features_concat_dimred_concat_val is not None
                else (None, None)
            )

        else:
            logger.info("Residue-level embeddings")
            # If the sequence features/embeddings have not been reduced by mean-pooling, pass directly to the downstream model (no PCA)
            # TODO: decide how to handle additional data columns when training models with residue embeddings

            (
                features_concat_train,
                features_concat_dimred_train,
                features_concat_dimred_concat_train,
            ) = None, None, None

            # If there is a validation split, process that too
            sequences_embedded_val = (
                self.embed_sequences(sequences["val"]) if len(sequences["val"]) > 0 else None
            )  # n x h
            (
                features_concat_val,
                features_concat_dimred_val,
                features_concat_dimred_concat_val,
            ) = (
                None,
                None,
                None,
            )

            # Process labels/targets.
            self.target_scaler.fit_scaler(labels["train"])
            labels_scaled = {
                split: self.target_scaler.transform_values(labels[split]) for split in labels.keys()
            }

            # Fit the predictor model
            self.predictor.train_model(
                sequences_embedded_train,
                labels_scaled["train"],
                sequences_embedded_val,
                labels_scaled["val"],
                sample_weights,
            )

            # Predict output with trained model
            predictions_train_scaled, predictions_probability_train = self.predictor.forward(
                sequences_embedded_train
            )
            predictions_val_scaled, predictions_probability_val = (
                self.predictor.forward(sequences_embedded_val)
                if sequences_embedded_val is not None
                else (None, None)
            )
            train_intervals = None
            val_intervals = None

        # rescaled predictions:
        predictions_train = self.target_scaler.restore_values(predictions_train_scaled)
        predictions_val = self.target_scaler.restore_values(predictions_val_scaled)

        # Update configs after fitting
        self.predictor.update_config_hparams()
        self.predictor.update_predictor_name()
        logger.info(f"This is the updated predictor name: {self.cfg.predictor.model_name}")
        self.cfg.general.composite_model_name = self.get_model_name()
        logger.info(f"Best model is: {self.predictor.get_hparams_string()}")
        # self.hparamsresults_df = pd.DataFrame(model_selector.cv_results_)
        self.predictor.trained = True
        logger.info("Moving to test mode")
        self.cfg.general.run_mode = "test"

        prediction_train = Prediction(
            sequences["train"],
            sequences_embedded_train,
            features_raw["train"],
            features["train"],
            features_concat_dimred_concat_train,
            predictions_train,
            predictions_train_scaled,
            "train",
            labels["train"],
            group_names["train"] if group_names is not None else None,
            predictions_probability_train,
            residue_level_prediction=self.cfg.predictor.residue_prediction_mode,
        )

        prediction_val = Prediction(
            sequences["val"],
            sequences_embedded_val,
            features_raw["val"],
            features["val"],
            features_concat_dimred_concat_val,
            predictions_val,
            predictions_val_scaled,
            "val",
            labels["val"],
            group_names["val"] if group_names is not None else None,
            predictions_probability_val,
            residue_level_prediction=self.cfg.predictor.residue_prediction_mode,
        )

        return prediction_train, prediction_val

    def get_model_config(self) -> DictConfig:
        """
        Returns the configuration of the model.

        Returns:
            DictConfig: The configuration of the model.
        """
        return self.cfg

    def get_model_name(self) -> str:
        """
        Creates a name for the model that captures all of the defining aspects.

        Returns:
            str: The name of the model.
        """
        composite_model_list = [
            self.cfg[component_type.embedder].model_name.replace("_", "")
            if self.cfg[component_type.embedder].model_name is not None
            else "",
            self.cfg[component_type.predictor].class_name.replace("_", "")
            if self.cfg[component_type.predictor].class_name is not None
            else "",
            self.cfg.dataset.data_name.replace("_", "")
            if self.cfg.dataset.data_name is not None
            else "",
        ]
        composite_model_name = "_".join(composite_model_list)
        logger.info(f"Composite model name: {composite_model_name}")
        return composite_model_name

    def training_complete(self) -> bool:
        """
        Checks if the training is complete.

        Returns:
            bool: True if training is complete, False otherwise.
        """
        dir_path = Path(self.cfg.general.composite_model_name)
        self.cfg.general.composite_model_name = self.get_model_name()
        hparam_results_csv_path = Path(dir_path, self.cfg.general.composite_model_name + ".csv")
        if hparam_results_csv_path.is_file():
            return True
        else:
            return False

    def store_model(
        self,
        run_name: str,
    ) -> None:
        """
        Stores the model.
        """

        self.cfg.general.composite_model_name = self.get_model_name()
        logger.info(f"Saving model: {self.cfg.general.composite_model_name}")

        # Create new directories to store the trained model
        dir_path = Path(self.cfg.persistence.artifacts_root_folder, run_name)

        if dir_path.exists():
            raise ValueError(
                f"This path already exists: {dir_path}, exiting before overwriting files"
            )

        create_folder_if_not_exists(dir_path)

        # Store the data encoders/params
        self.dataloader.save(dir_path)

        # Store the embedder, if applicable
        # (for now, ignore, since we won't store any of these)
        # Store the embedder and scaler if applicable
        if self.embedder is not None:
            # Save embedder-specific components (like kmer model)
            if self.cfg.embedder.model_name == "kmer":
                path = self.embedder.save_model(dir_path)
                mlflow.log_artifact(path)

            # Save scaler if standardization is enabled
            if self.cfg.embedder.standardize:
                scaler_path = dir_path / f"{self.cfg.embedder.model_name}_scaler.pkl"
                self.embedder.save_scaler(scaler_path)
                mlflow.log_artifact(str(scaler_path))
                # logger.info(f"Saved embedder scaler to {scaler_path}")

        # Store the dimensionality reducer
        if self.cfg.dimred.transform_name:
            path = self.dimensionreducer.save_model(dir_path)
            mlflow.log_artifact(path)

        # store target scaler.
        if target_scaler_path := self.target_scaler.save(dirpath=dir_path):
            mlflow.log_artifact(target_scaler_path)

        # Store the predictor
        assert self.cfg.predictor.model_name is not None
        # assert not OmegaConf.is_missing(self.cfg.predictor,"model_name")
        path = self.predictor.save_model(dir_path)
        mlflow.log_artifact(path)

        with open_dict(self.cfg):
            # downstream applications will be testing, not training
            self.cfg.general.run_mode = "test"

        # Save a copy of the hydra configs
        config_path = Path(dir_path, self.cfg.general.composite_model_name + ".yaml")
        with open(config_path, "w") as file:
            OmegaConf.save(self.cfg, file)
        print(mlflow.active_run().info)
        mlflow.log_artifact(str(config_path))

        # Save the results of the hyperparameter scan
        hparam_results_csv_path = Path(dir_path, self.cfg.general.composite_model_name + ".csv")
        if self.predictor.hparamsresults_df is not None:
            self.predictor.hparamsresults_df.to_csv(hparam_results_csv_path)
            mlflow.log_artifact(str(hparam_results_csv_path))
