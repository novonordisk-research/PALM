import logging

import mlflow
from omegaconf import DictConfig

from src.helpers.git_helpers import get_repo_name
from src.model.abstract_components import SKLPredictorModel
from src.model.composite_model import CompositeModel


logger = logging.getLogger(__name__)


def setup_mlflow(cfg: DictConfig, model: CompositeModel):
    logger.info("Setting up MLflow...")

    if cfg.mlflow.experiment_name:
        experiment_name = cfg.mlflow.experiment_name
    else:
        logger.info("Defaulting to repo name for experiment...")
        experiment_name = get_repo_name()
    logger.info(f"Experiment name: {experiment_name}")

    if cfg.mlflow.tracking_uri:
        if cfg.mlflow.tracking_uri != "DOMINO_TRACKING_URI":
            logger.info(f"Setting MLflow tracking URI to {cfg.mlflow.tracking_uri}")
            mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Autlog sets up training normalized confusion metric
    # mlflow.sklearn.autolog(log_post_training_metrics=False)
    if issubclass(type(model.predictor), SKLPredictorModel):
        mlflow.autolog()
    logger.info(f"Setting MLflow experiment to {experiment_name}")
