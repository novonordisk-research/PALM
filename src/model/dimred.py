import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.decomposition import PCA

from src.model.abstract_components import (
    DimRedModel,
    run_mode,
)


logger = logging.getLogger(__name__)


class NoReduction(DimRedModel):
    """
    Class for performing no dimensionality reduction.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initializes the NoReduction class.

        Args:
            cfg (DictConfig): Configuration dictionary.
        Attributes:
            fit (bool): Whether the model is fit.
        Methods:
            forward(embeddings: np.array) -> np.array: Perform forward pass of the dimensionality reduction.
            fitData(embeddings_train: np.array) -> np.array: Fit the training data to find the transform and number of PCs required.
            saveModel(dir_path: Path) -> None: Save the model.
        """
        logger.info(f"Load class (NoReduction): {self.__class__.__name__}")
        super().__init__(cfg)
        self.fit = True  # No transformation

    def forward(
        self,
        embeddings,
    ) -> np.array:
        return embeddings

    def fit_data(self, embeddings_train):
        return

    def save_model(self):
        # No transformation
        return


class PCADimReduction(DimRedModel):
    """Class for performing PCA dimensionality reduction."""

    def __init__(self, cfg: DictConfig):
        """
        Initialize the PCADimReduction class.

        Args:
            cfg (DictConfig): Configuration dictionary.
        Attributes:
            transform (np.array): Transformation matrix.
            n_components (int): Number of components.
            trained (bool): Whether the model is trained.
            fit (bool): Whether the model is fit.
        Methods:
            applyTransform(embeddings: np.array) -> np.array: Apply the transformation to the embeddings.
            forward(embeddings: np.array) -> np.array: Perform forward pass of the dimensionality reduction.
            fitData(embeddings_train: np.array) -> np.array: Fit the training data to find the transform and number of PCs required.
            saveModel(dir_path: Path) -> None: Save the model.

        """
        logger.info(f"Load class (PCADimReduction): {self.__class__.__name__}")
        super().__init__(cfg)

        self.transform = None
        self.n_components = None
        self.trained = False

        if self.cfg.general.run_mode == run_mode.train:
            if self.cfg.dimred.transform_name is None:
                logger.info("No transformation available, must fit the training data")
            else:
                raise ValueError(
                    f"Transformation name must be null (not {self.cfg.dimred.transform_name}) while in training mode"
                )
        elif self.cfg.general.run_mode in (run_mode.test, run_mode.embed):
            if self.cfg.general.composite_model_path:
                path = Path(
                    self.cfg.general.composite_model_path,
                    self.cfg.dimred.transform_name + ".npz",
                )
            else:
                path = Path(self.cfg.dimred.transform_name + ".npz")
            logger.info(f"Loading transformation from {path}")
            data = np.load(path, allow_pickle=False)
            self.transform = data["transform"]
            self.n_components = self.transform.shape[1]
            self.trained = True
        else:
            raise ValueError(f"run_mode: {self.cfg.general.run_mode} not defined")

        self.fit = False

    def apply_transform(self, embeddings: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the transformation to the embeddings.

        Args:
            embeddings (np.array): Input embeddings.

        Returns:
            np.array: Transformed embeddings.

        """
        """Apply transform with truncation"""
        logger.info(f"Before transform, data has dimensions: {embeddings.shape}")
        embeddings_transformed = embeddings.dot(self.transform)
        logger.info(f"After transform, data has dimensions: {embeddings_transformed.shape}")
        return embeddings_transformed

    def forward(
        self,
        embeddings: np.array,
    ) -> np.array:
        """
        Perform forward pass of the dimensionality reduction.

        Args:
            embeddings (np.array): Input embeddings.

        Returns:
            np.array: Transformed embeddings.

        """
        embeddings_transformed = self.apply_transform(embeddings)
        return embeddings_transformed

    def fit_data(self, embeddings_train: np.array) -> np.array:
        """
        Fit the training data to find the transform and number of PCs required.

        Args:
            embeddings_train (np.array): Training embeddings.

        Returns:
            np.array: Transformed embeddings.

        """
        """Find transform and number of PCs required to explain data"""
        # Note: this should only be performed with embeddings from the training set (to avoid any overfitting)
        embeddings_train = embeddings_train.copy()
        max_n_components = min(embeddings_train.shape[0], embeddings_train.shape[1])
        pca = PCA(n_components=max_n_components, random_state=self.cfg.general.random_state)
        pca.fit(embeddings_train)

        # determine the number of PCs necessary to explain the data
        varexp_cumsum = np.cumsum(pca.explained_variance_ratio_)
        # logger.info(varexp_cumsum)
        # logger.info(varexp_cumsum[varexp_cumsum < self.cfg.dimred.fraction_variance_explained])
        # logger.info(varexp_cumsum[varexp_cumsum < self.cfg.dimred.fraction_variance_explained].shape)
        self.n_components = min(
            embeddings_train.shape[1],
            (
                varexp_cumsum[varexp_cumsum < self.cfg.dimred.fraction_variance_explained].shape[0]
                + 1
            ),
        )
        logger.info(
            f"{self.n_components} dimensions describe {100 * varexp_cumsum[self.n_components - 1]:.2f}% (>= {self.cfg.dimred.fraction_variance_explained}) of the variance"
        )

        # set values
        self.transform = pca.components_[: self.n_components].T
        self.cfg.dimred.transform_name = f"PC_{self.n_components}dims"

    def save_model(self, dir_path: Path) -> None:
        """
        Save the model.

        Args:
            dir_path (Path): Directory path to save the model.

        """
        filepath = Path(dir_path, self.cfg.dimred.transform_name + ".npz")
        np.savez(
            filepath,
            transform=self.transform,
        )
        return filepath
