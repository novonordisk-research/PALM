import logging
import sys
from pathlib import Path

import mlflow
import numpy as np
import omegaconf
import pandas as pd
from omegaconf import DictConfig, open_dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler
from skops.io import dump as sk_dump
from skops.io import load as sk_load

from src.model.abstract_components import run_mode


logger = logging.getLogger(__name__)

"""
Each of the data classes can be initialized in three ways:
- No arguments. Infer parameters from data. Example: training data during training mode
- Arguments. Parameters are passed explicitly. Example: validation/test data during training mode
- From file. Parameters are read from file. Example: inference mode

All classes have a get_data method to facilitate usage
"""


class RadialBasisFunctionGaussian:
    """Class for applying a gaussian radial basis function to real-valued data"""

    def __init__(
        self,
        cfg: DictConfig,
        name: str,
        min_value: float = None,
        max_value: float = None,
    ):
        """
        Initializes a new instance of the RadialBasisFunctionGaussian class.
        Args:
            cfg (DictConfig): The configuration for the model.
            name (str): The name of the radial basis function encoder.
            min_value (float): The minimum value for the given data.
            max_value (float): The maximum value for the given data.
        Attributes:
            cfg (DictConfig): The configuration for the model.
            name (str): The name of the radial basis function encoder.
            n_kernels (int): The number of kernels
            min_value (float): The minimum value for the given data.
            max_value (float): The maximum value for the given data.
            kernel_centers (np.array): The center positions of the kernels
            kernel_width (np.array): The standard deviation of the kernels
            fit (bool): Whether the model is fit.
        Methods:
            _set_rbf_params: Private method for setting the parameters of the kernel.
            forward: Apply radial basis function to data
            save: Stores the model
        """

        self.cfg = cfg
        self.name = name
        self.n_kernels = None
        self.min_value = None
        self.max_value = None
        self.kernel_centers = None
        self.kernel_width = None

        if cfg.general.run_mode == run_mode.train:
            # Initialize the rbf kernel
            self.n_kernels = cfg.dataset.rbf_n_kernels
            if min_value is None:
                raise ValueError("min_value cannot be 'None' when in train mode")
            if max_value is None:
                raise ValueError("max_value cannot be 'None' when in train mode")
            self.min_value = min_value
            self.max_value = max_value

        elif cfg.general.run_mode == run_mode.test:
            self.n_kernels = cfg.dataset.real[self.name].rbf.n_kernels
            self.min_value = cfg.dataset.real[self.name].rbf.min_value
            self.max_value = cfg.dataset.real[self.name].rbf.max_value
        else:
            raise ValueError(f"run_mode: {cfg.general.run_mode} not defined")

        self._set_rbf_params()

    def _set_rbf_params(self) -> None:
        """
        Private method to set the parameters of the kernel
        """
        # code adapted from protein_mpnn_utils.py
        # define the kernel centers and width
        self.kernel_centers = np.linspace(self.min_value, self.max_value, self.n_kernels)
        self.kernel_width = (self.max_value - self.min_value) / self.n_kernels

    def forward(self, data: np.array) -> np.array:
        """
        Apply radial basis function to data
        Args:
            data (np.array): Data with shape n x 1
        Returns
            np.array: Transformed data with shape n x n_kernels
        """

        # apply the rbf to the data (n x 1)
        transformed_data = np.exp(
            -(((data - self.kernel_centers) / self.kernel_width) ** 2)
        )  # n x n_kernels
        return transformed_data

    def save(self, dir_path: Path = None) -> None:
        """
        Save the parameters

        Args:
            dir_path (Path): The path to the directory where the output will be stored


        Note: dir_path is ignored in this method. It's left as an argument to preserve
        interoperability between classes.
        """
        # Update the DictConfig
        with open_dict(self.cfg):
            self.cfg.dataset.real[self.name].rbf = {
                "n_kernels": self.n_kernels,
                "min_value": self.min_value,
                "max_value": self.max_value,
            }


class SequenceData:
    """Class for storing sequence data"""

    def __init__(self, sequences: list[str], name: str):
        """
        Initializes a new instance of the class.
        Args:
            sequences (list[str]): The sequences in the data.
            name (str): The name of the sequences.
        Attributes:
            sequences (list[str]): The sequences in the data.
            name (str): The name of the sequences.
            save: For compatibility
        Methods:
            get_data: Return the sequences
        """

        self.sequences = np.array(sequences)
        self.name = name

    def get_data(self, data_idx: list[int] = None):
        """
        Return the sequences in the data split

        Args:
            data_idx (list[int]): the indices of the sequences in the desired split
        """
        sequences = self.sequences
        if data_idx:
            sequences = sequences[data_idx]  # filter to only the data from the desired split
        return sequences

    def save(self, dir_path: Path = None):
        """Empty method for compatibility

        Args:
            dir_path (Path): The path to the directory where the output will be stored

        Note: dir_path is ignored in this method. It's left as an argument to preserve
        interoperability between classes.
        """
        pass


class CategoricalData:
    """Class for storing categorical data by one-hot encoding"""

    def __init__(
        self,
        cfg: DictConfig,
        data: list[str],
        name: str,
        apply_dimred: bool = False,
        data_train_idx: list[int] = None,
    ):
        """
        Initializes a new instance of the CategoricalData class.
        Args:
            cfg (DictConfig): The configuration for the model.
            data (list[str]): The categorical data.
            name (str): The name of the data
            apply_dimred (bool): Controls whether dimensionality reduction is applied to the data
            data_train_idx (list[int]): The indices of the training data
        Attributes:
            cfg (DictConfig): The configuration for the model.
            data (np.array): The categorical data (n x 1)
            name (str): The name of the data
            apply_dimred (bool): Controls whether dimensionality reduction is applied to the data
            encoder (OneHotEncoder): The sklearn one hot encoder module
        Methods:
            _fit: Fit the one-hot encoder on the training data
            get_data: Get the encoded data
            save: Save the parameters of the module
        """

        self.cfg = cfg
        self.data = np.array(data).reshape(-1, 1)
        self.name = name
        self.apply_dimred = False

        self.encoder = None

        if cfg.general.run_mode == run_mode.train:
            # Initialize a new encoder
            self.encoder = OneHotEncoder(sparse_output=False)
            self.apply_dimred = apply_dimred

            # Fit the encoder
            self._fit(data_train_idx)

        elif cfg.general.run_mode == run_mode.test:
            self.apply_dimred = cfg.dataset.categorical[self.name].apply_dimred
            # Load the encoder from a file
            if cfg.general.composite_model_path:
                model_path = Path(
                    cfg.general.composite_model_path, f"categorical_{self.name}.skops"
                )
            else:
                model_path = Path(f"categorical_{self.name}.skops")
            self.encoder = sk_load(model_path)
        else:
            raise ValueError(f"run_mode: {cfg.general.run_mode} not defined")

    def _fit(self, data_train_idx: list[int]):
        """
        Fit the encoder on the training data

        Args:
            data_train_idx (list[int]): The indices of the training examples in the dataset
        """
        train_data = self.data[data_train_idx]
        self.encoder.fit(train_data)

    def get_data(self, data_idx: list[int] = None):
        """
        Return the data from selected split

        Args:
            data_idx (list[int]): The indices of the data in the selected split

        Returns:
            np.array: The selected one hot encoded data
        """
        data = self.encoder.transform(self.data)  # n x n_categories
        if data_idx:
            data = data[data_idx]  # filter to only the data from the desired split
        return data

    def save(self, dir_path: Path):
        """
        Store the parameters and the fitted encoder

        Args:
            dir_path (str): The path where the files should be stored
        """
        # Update the DictConfig
        with open_dict(self.cfg):
            if "categorical" not in self.cfg.dataset:
                self.cfg.dataset.categorical = dict()
            self.cfg.dataset.categorical[self.name] = {"apply_dimred": self.apply_dimred}

        # Save the file
        model_path = Path(dir_path, f"categorical_{self.name}.skops")
        sk_dump(self.encoder, model_path)
        mlflow.log_artifact(model_path)


class RealData:
    """Class for storing real-valued data"""

    def __init__(  # noqa: C901, PLR0912, PLR0913 too complex, too many arguments, too many branches (ifelses)
        self,
        cfg: DictConfig,
        data: np.array,
        name: str,
        standardize: str = None,
        apply_dimred: bool = None,
        data_train_idx: list[int] = None,
    ):
        """
        Initializes a new instance of the class.
        Args:
            cfg (DictConfig): The configuration for the model.
            data (np.array): The real-valued data, shape: (n,).
            name (str): The name of the module.
            standardize (bool): Controls whether the data is standardized.
            apply_dimred (bool): Controls whether dimensionality reduction is applied to the data
            data_train_idx (list[int]): The indices of the training data
        Attributes:
            cfg (DictConfig): The configuration for the model.
            data (np.array): The real-valued data, shape: (n,).
            name (str): The name of the module.
            standardize (bool): Controls whether the data is standardized.
            apply_dimred (bool): Controls whether dimensionality reduction is applied to the data
            scaler (Class): The sklearn scaler applied to the data
            rbf_encoder (Class): The radial basis function encoder applied to the data
        Methods:
            _fit: Fit the scaler on the training data
            get_data: Get the data from the selected split
            save: Save the module
        """

        self.cfg = cfg
        self.data = data.reshape(-1, 1)  # n
        self.name = name
        self.standardize = None
        self.apply_dimred = None

        self.scaler = None
        self.rbf_encoder = None

        # Determine the class type
        data_scaler_class_name = cfg.dataset.data_scaler if "data_scaler" in cfg.dataset else None
        rbf_encoder_class_name = cfg.dataset.rbf_encoder if "rbf_encoder" in cfg.dataset else None
        rbf_encoder_class = (
            getattr(sys.modules[__name__], rbf_encoder_class_name)
            if rbf_encoder_class_name
            else None
        )

        if cfg.general.run_mode == run_mode.train:
            if standardize is None:
                raise ValueError("standardize cannot be 'None' when in train mode")
            self.standardize = standardize
            if apply_dimred is None:
                raise ValueError("apply_dimred cannot be 'None' when in train mode")
            self.apply_dimred = apply_dimred

            # Set up the scaler using the train data
            if data_scaler_class_name == "RobustScaler":
                self.scaler = RobustScaler()
            elif data_scaler_class_name == "MinMaxScaler":
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
            else:
                raise ValueError(f"data_scaler: {data_scaler_class_name} not defined")
            self._fit(data_train_idx)

            # Set up the RBF encoder using the train data
            if rbf_encoder_class:
                self.rbf_encoder = rbf_encoder_class(cfg, self.name, min_value=-1, max_value=1)

        elif cfg.general.run_mode == run_mode.test:
            self.standardize = cfg.dataset.real[self.name].standardize
            self.apply_dimred = cfg.dataset.real[self.name].apply_dimred

            # Load the scaler from a file
            if self.standardize:
                if cfg.general.composite_model_path:
                    model_path = Path(
                        cfg.general.composite_model_path,
                        f"real_{self.name}_scaler.skops",
                    )
                else:
                    model_path = Path(f"real_{self.name}_scaler.skops")
                self.scaler = sk_load(model_path)

            # Load the RBF encoder
            if rbf_encoder_class:
                self.rbf_encoder = rbf_encoder_class(cfg, self.name, min_value=-1, max_value=1)

        else:
            raise ValueError(f"run_mode: {cfg.general.run_mode} not defined")

    def _fit(self, data_train_idx: list[int]):
        """
        Fit the scaler on the training data

        Args:
            data_train_idx (list[int]): The indices of the training examples in the dataset
        """
        train_data = self.data[data_train_idx]
        self.scaler.fit(train_data)

    def get_data(self, data_idx: list[int] = None):
        """
        Return the data from the selected split

        Args:
            data_idx (list[int]): The indices of the data in the selected split

        Returns:
            np.array: The selected data (optionally rescaled and/or encoded)
        """
        data = self.data
        if self.scaler:
            data = self.scaler.transform(self.data)  # n x 1
        if self.rbf_encoder:
            data = self.rbf_encoder.forward(data)  # n x n_kernels
        if data_idx:
            data = data[data_idx]  # filter to only the data from the desired split
        return data

    def save(self, dir_path: Path):
        """
        Store the parameters and the fitted scaler

        Args:
            dir_path (str): The path where the files should be stored

        """

        # Update the DictConfig
        with open_dict(self.cfg):
            if "real" not in self.cfg.dataset:
                self.cfg.dataset.real = dict()
            self.cfg.dataset.real[self.name] = {
                "standardize": self.standardize,
                "apply_dimred": self.apply_dimred,
            }
            self.cfg.dataset.real[self.name].apply_dimred = self.apply_dimred

        # Save the files
        if self.scaler:
            model_path = Path(dir_path, f"real_{self.name}_scaler.skops")
            sk_dump(self.scaler, model_path)
            mlflow.log_artifact(model_path)

        if self.rbf_encoder:
            self.rbf_encoder.save()


class CSVDataLoader:
    """A custom dataloader for working with csv files"""

    def __init__(self, cfg: DictConfig, df: pd.DataFrame = None):  # noqa: C901, PLR0912
        """
        Initialize the CSVDataLoader.

        Args:
            cfg (DictConfig): The configuration dictionary.

        Attributes:
            cfg (DictConfig): The configuration dictionary.
            df (pd.DataFrame): The dataframe containing the data.

        Methods:
            getData(data_type: str = None) -> (list, np.array): Get the data from the CSV file.
        """
        self.cfg = cfg

        if df is None:
            csv_path = Path(
                cfg.persistence.data_root_folder,
                cfg.persistence.training_data,
                cfg.dataset.data_name + ".csv",
            )
            parquet_path = Path(
                cfg.persistence.data_root_folder,
                cfg.persistence.training_data,
                cfg.dataset.data_name + ".parquet.gzip",
            )

            if csv_path.exists():
                self.df = pd.read_csv(csv_path, index_col=0)
                logger.info(f"Loading file: {csv_path}")

            elif parquet_path.exists():
                logger.info("No csv found, moving to parquet")
                self.df = pd.read_parquet(parquet_path, index_col=0)
                logger.info(f"Loading file: {parquet_path}")

            else:
                raise ValueError(
                    f"Could not find a .csv or .parquet file with the name {cfg.dataset.data_name}"
                )
        else:
            self.df = df

        # determine the kind of data
        assert cfg.dataset.task in ["classification_binary", "regression"]
        if isinstance(cfg.dataset.residue_prediction_labels, str):
            # overrides other options
            self.target_value_type = self.cfg.dataset.residue_prediction_labels
        elif cfg.dataset.task == "classification_binary":
            self.target_value_type = "value_bool"
        elif cfg.dataset.task == "regression":
            self.target_value_type = "value_real"
        else:
            raise ValueError

        # validate the split fractions
        if self.cfg.dataset.use_predefined_split:
            if self.cfg.dataset.data_split_column not in self.df.columns:
                raise ValueError(
                    f"Pre-defined split is requested, but there is no '{self.cfg.dataset.data_split_column}' column in the dataframe"
                )
            for split_name in ["train", "val", "test"]:
                if split_name not in self.cfg.dataset.data_split:
                    raise ValueError(
                        f"Data split fractions (train, val, test) must be provided in the configuration. Missing: {split_name}"
                    )
        self.train_frac = cfg.dataset.data_split.train
        self.val_frac = cfg.dataset.data_split.val
        self.test_frac = cfg.dataset.data_split.test
        if (self.train_frac + self.val_frac + self.test_frac) != 1.0:
            raise ValueError(
                f"Data splits must sum to 1: \
                            train: {self.train_frac}, \
                            val: {self.val_frac}, \
                            test: {self.test_frac}"
            )

        # get the sample weight column name, if they are defined
        self.sample_weight_col_name = (
            "sample_weight" if self.cfg.dataset.use_sample_weights else None
        )

        if self.cfg.dataset.group_column is not None:
            assert self.cfg.dataset.group_column in self.df.columns, (
                f'Chosen group_column "{self.cfg.dataset.group_column}" not found!'
            )
        logger.info(f'Using group_column "{self.cfg.dataset.group_column}".')

        if self.cfg.general.run_mode == run_mode.train:
            # determine which columns from the table will be used as input to the model
            self._define_additional_data_columns(
                cfg.dataset.add_data_columns,
                cfg.dataset.data_columns_standard,
                cfg.dataset.data_columns_dimred,
            )

        elif self.cfg.general.run_mode == run_mode.test:
            # use the same columns that were used when training the model
            self.add_data_columns = self.cfg.dataset.add_data_columns
            self.data_columns_standard = self.cfg.dataset.data_columns_standard
            self.data_columns_dimred = self.cfg.dataset.data_columns_dimred
        else:
            raise ValueError(f"run_mode: {cfg.general.run_mode} not defined")

        # load the data from the table
        self._load_data()

        # convert to data types
        self._convert_df_to_data()

        # get the sample weights from the table
        self._load_sample_weights()

    def _define_additional_data_columns(  # noqa: C901, PLR0912 too complex, too many branches
        self, add_data_columns, data_columns_standard, data_columns_dimred
    ):
        """
        Define additional data columns in the dataframe

        Args:
            add_data_columns (list[str] or True or null): The additional data columns that will be added.
            data_columns_standard (list[str] or True or null): The additional data columns that will be standardized.
            data_columns_dimred (list[str] or True or null): The additional data columns that will be dimensionality reduced.

        """

        # checking for additional data columns starting with 'data_' (ignoring data_split_column)
        additional_data_columns = sorted(
            [
                col
                for col in self.df.columns
                if col.startswith("data_") and col != self.cfg.dataset.data_split_column
            ]
        )
        if len(additional_data_columns) > 0:
            # check add_data_columns option - true all are added, false nothing is added, if list, listed columns are added
            logger.info(
                "%i additional data columns available: %s"
                % (len(additional_data_columns), ", ".join(additional_data_columns))
            )
            if isinstance(add_data_columns, bool) and add_data_columns:
                logger.info("Adding all additional data columns to embedding.")
                add_data_columns = additional_data_columns
            elif isinstance(add_data_columns, bool) and not add_data_columns:
                logger.info("Additional data columns NOT added to embedding.")
                add_data_columns = []
            elif isinstance(add_data_columns, omegaconf.listconfig.ListConfig):
                logger.info(
                    "Listed %i data columns added to embedding: %s"
                    % (len(add_data_columns), ", ".join(add_data_columns))
                )
                assert all([x in self.df.columns for x in add_data_columns])
            else:
                raise ValueError(
                    'dataset.add_data_columns either needs to be a boolean or a list of columns (omegaconf.listconfig.ListConfig). Type was "%s" '
                    % type(add_data_columns)
                )
        else:
            logger.info("No additional data columns found while parsing.")
            add_data_columns = []

        if len(add_data_columns) > 0:
            # assess standardization options of add_data_columns
            if isinstance(data_columns_standard, bool) and data_columns_standard:
                data_columns_standard = add_data_columns
            elif isinstance(data_columns_standard, bool) and not data_columns_standard:
                data_columns_standard = []
            elif isinstance(data_columns_standard, omegaconf.listconfig.ListConfig):
                assert all([x in self.df.columns for x in data_columns_standard])

            else:
                raise ValueError(
                    'data_columns_standard either needs to be a boolean or a list of columns (omegaconf.listconfig.ListConfig). Type was "%s" '
                    % type(data_columns_standard)
                )

            if len(data_columns_standard) > 0:
                logger.info(
                    "%i data columns will be standardized: %s"
                    % (len(data_columns_standard), ", ".join(data_columns_standard))
                )

            # assess dimred options of add_data_columns
            if isinstance(data_columns_dimred, bool) and data_columns_dimred:
                data_columns_dimred = add_data_columns
            elif isinstance(data_columns_dimred, bool) and not data_columns_dimred:
                data_columns_dimred = []
            elif isinstance(data_columns_dimred, omegaconf.listconfig.ListConfig):
                assert all([x in self.df.columns for x in data_columns_dimred])
            else:
                raise ValueError(
                    'dataset.data_columns_dimred either needs to be a boolean or a list of columns (omegaconf.listconfig.ListConfig). Type was "%s" '
                    % type(data_columns_dimred)
                )

            if len(data_columns_dimred) > 0:
                logger.info(
                    "%i data columns will be transformed: %s"
                    % (len(data_columns_dimred), ", ".join(data_columns_dimred))
                )
        else:
            data_columns_standard = []
            data_columns_dimred = []

        self.add_data_columns = add_data_columns
        self.data_columns_standard = data_columns_standard
        self.data_columns_dimred = data_columns_dimred

    def _load_data(self) -> None:
        """
        Load the training/val/test data from the CSV file.

        Returns:
            tuple: A tuple containing the data as X (list) and y (numpy array).
        """

        if (
            self.cfg.dataset.use_predefined_split
            and self.cfg.dataset.data_split_column not in self.df.columns
        ):
            raise ValueError(
                f"Pre-defined split is requested, but there is no '{self.cfg.dataset.data_split_column}' column in the dataframe"
            )
        predefined_split = self.cfg.dataset.use_predefined_split

        # convert int columns to float to make mlflow happy
        m = self.df.select_dtypes("int64")
        self.df[m.columns] = m.astype("float64")

        if predefined_split:
            self.data_idx = {
                "train": list(self.df[self.cfg.dataset.data_split_column] == "train"),
                "val": list(self.df[self.cfg.dataset.data_split_column] == "val"),
                "test": list(self.df[self.cfg.dataset.data_split_column] == "test"),
            }
            n_train, n_val, n_test = (
                np.array(self.data_idx["train"]).sum(),
                np.array(self.data_idx["val"]).sum(),
                np.array(self.data_idx["test"]).sum(),
            )

            if self.cfg.general.run_mode == run_mode.train:
                assert n_train > 0 and n_test > 0, (
                    f"Train (N = {n_train}), validation (N = {n_val}), test (N = {n_test})"
                )
                logger.info(
                    f"Select data according to predetermined splits in the dataframe, "
                    f"train (N = {n_train}), validation (N = {n_val}), test (N = {n_test})"
                )
            else:
                assert n_test > 0, (
                    f"Train (N = {n_train}), validation (N = {n_val}), test (N = {n_test})"
                )
                logger.info(
                    f"Select data according to predetermined splits in the dataframe, "
                    f"train (N = {n_train}), validation (N = {n_val}), test (N = {n_test})"
                )

        else:
            # Create indices for the full dataset
            indices = np.arange(len(self.df))
            total_samples = len(indices)

            if self.val_frac == 0.0:
                # Simple train/test split when no validation set is needed
                train_indices, test_indices = train_test_split(
                    indices,
                    train_size=self.train_frac,
                    random_state=self.cfg.general.random_state,
                )
                val_indices = []  # Empty validation set
            else:
                # Regular three-way split
                train_indices, valtest_indices = train_test_split(
                    indices,
                    train_size=self.train_frac,
                    random_state=self.cfg.general.random_state,
                )

                val_indices, test_indices = train_test_split(
                    valtest_indices,
                    train_size=self.val_frac / (self.val_frac + self.test_frac),
                    random_state=self.cfg.general.random_state,
                )

            # Create boolean masks for each split
            self.data_idx = {
                "train": [i in train_indices for i in range(total_samples)],
                "val": [i in val_indices for i in range(total_samples)],
                "test": [i in test_indices for i in range(total_samples)],
            }

            n_train = len(train_indices)
            n_val = len(val_indices)
            n_test = len(test_indices)

            logger.info(
                f"Select train:val:test data according to random split \
                    {int(100 * self.train_frac)}: \
                    {int(100 * self.val_frac)}: \
                    {int(100 * self.test_frac)} \
                    (total size N = {total_samples})"
            )
            logger.info(f"Split sizes - Train: {n_train}, Validation: {n_val}, Test: {n_test}")
            # raise NotImplementedError
            # logger.info(
            #     f"Select train:val:test data according to random split \
            #         {int(100*self.train_frac)}: \
            #         {int(100*self.val_frac)}: \
            #         {int(100*self.test_frac)} \
            #         (total size N = {len(self.df)})"

    def _convert_df_to_data(self):
        """
        Create data objects from the additional columns
        """

        # Get sequence data
        self.data = [SequenceData(sequences=self.df["sequence"], name="sequence")]

        # Get additional columns
        col2dtype = {}
        for col in self.add_data_columns:
            # infer datatypes for now but one day perhaps they should be defined?
            if type(self.df[col].iloc[0]) is str:
                col2dtype[col] = str
                self.data.append(
                    CategoricalData(
                        self.cfg,
                        data=self.df[col],
                        name=col,
                        apply_dimred=col in self.data_columns_dimred,
                        data_train_idx=self.data_idx["train"],
                    )
                )
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                col2dtype[col] = self.df[col].dtype
                self.data.append(
                    RealData(
                        self.cfg,
                        data=np.array(self.df[col]),
                        name=col,
                        standardize=col in self.data_columns_standard,
                        apply_dimred=col in self.data_columns_dimred,
                        data_train_idx=self.data_idx["train"],
                    )
                )
            else:
                raise ValueError(f"dtype for col: {col} unknown")

        logger.info(f"Loaded columns: {col2dtype}")

        # Get training labels
        if self.target_value_type not in self.df.columns:
            # Inference mode, no labels available
            self.labels = {
                "train": None,
                "val": None,
                "test": None,
            }
        else:
            self.labels = {
                "train": np.array(self.df[self.target_value_type].iloc[self.data_idx["train"]]),
                "val": np.array(self.df[self.target_value_type].iloc[self.data_idx["val"]]),
                "test": np.array(self.df[self.target_value_type].iloc[self.data_idx["test"]]),
            }

        # Get group labels, if available
        if self.cfg.dataset.group_column is None:
            self.group_names = None
        else:
            self.group_names = {
                "train": list(self.df[self.cfg.dataset.group_column].iloc[self.data_idx["train"]]),
                "val": list(self.df[self.cfg.dataset.group_column].iloc[self.data_idx["val"]]),
                "test": list(self.df[self.cfg.dataset.group_column].iloc[self.data_idx["test"]]),
            }

    def _load_sample_weights(self):
        if self.sample_weight_col_name:
            self.sample_weights = {
                "train": np.array(
                    self.df[self.sample_weight_col_name].iloc[self.data_idx["train"]]
                ),
                "val": np.array(self.df[self.sample_weight_col_name].iloc[self.data_idx["val"]]),
                "test": np.array(self.df[self.sample_weight_col_name].iloc[self.data_idx["test"]]),
            }
        else:
            self.sample_weights = {"train": None, "val": None, "test": None}

    # def get_sequences(self):
    #     sequences = {}
    #     for split in ['train','val','test']:
    #         sequences[split] = self.df['sequence'].iloc[self.data_idx[split]]
    #     assert len(sequences) == 3
    #     return sequences

    def get_sequence_data(self):
        """
        Returns the sequence data

        Returns:
            dict[np.array]: The sequence data for all splits
        """
        sequence_data = {}
        splits = ["train", "val", "test"]
        for split in splits:
            for data_channel in self.data:
                if data_channel.name == "sequence":
                    sequence_data[split] = data_channel.get_data(self.data_idx[split])
        assert len(sequence_data) == len(splits)
        return sequence_data

    def get_additional_data_untransformed(self):
        """
        Returns the additional data columns concatenated together without transformations

        Returns:
            dict[np.array]: The additional data, untransformed, for all splits
        """
        additional_data = {}
        splits = ["train", "val", "test"]
        for split in splits:
            additional_data[split] = (
                self.df[self.add_data_columns].iloc[self.data_idx[split]]
                if self.add_data_columns != []
                else None
            )
        assert len(additional_data) == len(splits)
        return additional_data

    def get_additional_data(self, apply_dimred: bool):
        """
        Returns the additional data columns concatenated together

        Returns:
            dict[np.array]: The additional data for all splits
        """
        additional_data = {}
        splits = ["train", "val", "test"]
        for split in splits:
            split_data = []
            for data_channel in self.data:
                if data_channel.name != "sequence" and data_channel.apply_dimred == apply_dimred:
                    col_additional_data = data_channel.get_data(self.data_idx[split])
                    logger.info(f"{data_channel.name} : {col_additional_data.shape}")
                    split_data.append(col_additional_data)
            additional_data[split] = (
                np.concatenate(split_data, axis=-1) if len(split_data) > 0 else None
            )
        assert len(additional_data) == len(splits)
        return additional_data

    def get_training_labels(self):
        """
        Returns the labels

        Returns:
            dict[np.array]: The labels for all the splits
        """
        return self.labels

    def get_group_names(self):
        """
        Returns the group labels

        Returns:
            dict[list[str]]: The group names for all the splits
        """
        return self.group_names

    def get_sample_weights(self):
        """
        Returns the sample weights

        Returns:
            dict[np.array]: The sample weights
        """
        return self.sample_weights["train"]

    def save(self, dir_path: str):
        """Save the parameters and fitted encoders

        Args:
            dir_path (str): Path to the directory where the output will be stored
        """
        # Update the DictConfig
        with open_dict(self.cfg):
            self.cfg.dataset.add_data_columns = self.add_data_columns
            self.cfg.dataset.data_columns_dimred = self.data_columns_dimred
            self.cfg.dataset.data_columns_standard = self.data_columns_standard

        for data_channel in self.data:
            data_channel.save(dir_path)

    # Need to recreate this
    # def summarize_loaded_data(self):
    #     """Summarize the training and test data

    #     Returns:
    #         None
    #     """
    #     if self.x_train is None:
    #         self.get_data()

    #     logger.info("Loaded the following data:")
    #     logger.info(f"  x_train: {self.x_train.shape}")
    #     logger.info(f"  x_val: {self.x_val.shape}")
    #     logger.info(f"  x_test: {self.x_test.shape}")
    #     logger.info(f"  y_train: {self.y_train.shape}")
    #     logger.info(f"  y_val: {self.y_val.shape}")
    #     logger.info(f"  y_test: {self.y_test.shape}")
    #     if not self.cfg.dataset.add_data_columns is None:
    #         logger.info(f"  Additional data_columns: {self.cfg.dataset.add_data_columns}")
    #         logger.info(f"  Additional data_columns standardize: {self.cfg.dataset.data_columns_standard}")
    #         logger.info(f"  Additional data_columns dimred: {self.cfg.dataset.data_columns_dimred}")
    #     else:
    #         logger.info(f"  No additional data_columns loaded")
