import logging
import os
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from skops.io import dump as sk_dump
from skops.io import load as sk_load

from src.model.abstract_components import run_mode


class PassThroughScaler:
    """
    Pass through scaler honouring the interface of sklearn.preprocessing scalers.
    """

    def fit(self, values):
        pass

    def fit_transform(self, values):
        return values

    def transform(self, values):
        return values

    def inverse_transform(self, values):
        return values


class ScalerWrap:
    """
    Top level implementation of scaler functionality.

    Wraps the low level functionality of Sklearn.preprocessing scalers.
    Includes a pass-through scaler.

    Initialize:
    - cfg: DictConfig: The configuration dictionary.
    - name: str: naming of the scaler (default: 'targetscaler')
    Sideeffect: if in runmode test, the scaler is loaded from disk.

    main methods:
    - fit_scaler: fit the scaler to the data
    - transform_values: transform the values to the scale of the fitted scaler
    - restore_values: restore the scaled values to their original scale
    - save: save the fitted scaler to disk
    - load: load a saved scaler from disk

    """

    scaler_map = {
        "PassThroughScaler": PassThroughScaler,
        None: PassThroughScaler,
        "MinMaxScaler": MinMaxScaler,
        "StandardScaler": StandardScaler,
        "RobustScaler": RobustScaler,
    }

    def __init__(self, cfg: DictConfig, name="targetscaler") -> None:
        self.cfg = cfg
        self.fitted = False
        self.logger = logging.getLogger(__name__)
        self.name = name

        # If initialized during test:
        if self.cfg.general.run_mode in (run_mode.test, run_mode.test.name):
            self.load(self.path_default)

    @property
    def config_scaler_name(self) -> str:
        if self.cfg.dataset.target_scaling is None:
            return "PassThroughScaler"
        return self.cfg.dataset.target_scaling

    @property
    def scaler(
        self,
    ) -> PassThroughScaler | MinMaxScaler | StandardScaler | RobustScaler:
        if not hasattr(self, "_scaler") or self._scaler is None:
            raise AttributeError(
                "Scaler not yet initialized, either call fit_scaler with data or load a saved scaler."
            )
        return self._scaler

    @property
    def path_filename(self) -> Path:
        # return f"_{self.config_scaler_name}.skops"
        if pred_model_name := self.cfg.predictor.model_name:
            return Path(f"{pred_model_name}_{self.name}.skops")

        return Path(f"{self.name}.skops")

    @property
    def path_default(self) -> Path:
        if comp_path := self.cfg.general.composite_model_path:
            path = Path(comp_path, self.path_filename)
        else:
            path = self.path_filename.absolute()
            self.logger.warning("general.composite_model_path not set, path defaults to workdir.")
        return path

    def fit_scaler(self, values: np.ndarray) -> None:
        if values is None or (len(values) == 0):
            raise ValueError("Values cannot be None or empty when fitting the scaler.")

        if hasattr(self, "_scaler") and self._scaler is not None:
            self.logger.warning("Scaler already fitted, overwriting - unexpected behavior.")

        _scaler = self.scaler_map[self.config_scaler_name]()

        if values.ndim == 1:
            _scaler.fit(values.reshape(-1, 1))
        else:
            _scaler.fit(values)

        self._scaler = _scaler
        self.logger.info(f"Fitted scaler: {self.config_scaler_name}")

    def transform_values(self, values: np.ndarray) -> np.ndarray:
        """
        Transform the values to the scale of the fitted scaler.
        """
        if values is None or (len(values) == 0):
            self.logger.debug("skipping transform - empty values")
            return values

        self.logger.info(f"Before standardization - min: {values.min()}, max: {values.max()}")
        if values.ndim == 1:
            values_transformed = self.scaler.transform(values.reshape(-1, 1)).flatten()
        else:
            values_transformed = self.scaler.transform(values)

        self.logger.info(
            f"After standardization - min: {values_transformed.min()}, max: {values_transformed.max()}"
        )
        return values_transformed

    def restore_values(self, values: np.ndarray) -> np.ndarray:
        """
        Restore the scaled values to their original scale.
        """
        if values is None or (len(values) == 0):
            self.logger.debug("skipping restore - empty values")
            return values

        self.logger.info(f"Before standardization - min: {values.min()}, max: {values.max()}")
        if values.ndim == 1:
            values_restored = self.scaler.inverse_transform(values.reshape(-1, 1)).flatten()
        else:
            values_restored = self.scaler.inverse_transform(values)

        self.logger.info(
            f"After standardization - min: {values_restored.min()}, max: {values_restored.max()}"
        )
        return values_restored

    def save(self, filepath: os.PathLike = None, dirpath: os.PathLike = None) -> Path | None:
        """
        Save the fitted scaler to disk.

        Args:
            path (str or Path): Path to save the scaler
        """
        if isinstance(self.scaler, PassThroughScaler):
            self.logger.debug("Skipping save - PassThroughScaler set.")
            return None

        if filepath:
            savepath = filepath
        elif dirpath:
            savepath = Path(dirpath, self.path_filename)
        else:
            savepath = self.path_default
            self.logger.warning("No path provided, using default path.")

        self.logger.info(f"Saving scaler to: {savepath}")
        sk_dump(self.scaler, savepath)
        return savepath

    def load(self, filepath: os.PathLike = None, dirpath: os.PathLike = None) -> None:
        """
        Loads a saved scaler from disk.

        side effect: self.scaler is set.

        Args:
            path (str or Path): Path to dumped scaler.
        returns:
            None
        """

        if (
            self.cfg.dataset.target_scaling is None
            or self.cfg.dataset.target_scaling == "PassThroughScaler"
        ):
            self.logger.debug("skipping load - using PassThroughScaler")
            self._scaler = PassThroughScaler()
            return

        if filepath:
            loadpath = filepath
        elif dirpath:
            loadpath = Path(dirpath, self.path_filename)
        else:
            loadpath = self.path_default
            self.logger.warning("No path provided, using default path.")

        self.logger.info(f"Loading scaler from {loadpath}")
        self._scaler = sk_load(loadpath)
