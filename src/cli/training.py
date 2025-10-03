import logging
from datetime import datetime, timezone
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from omegaconf import DictConfig, OmegaConf

from src.helpers.figure_helpers import (
    plot_confusion_matrix,
    plot_cosine_similarity,
    plot_pca_components_with_hue,
    plot_precision_recall,
    plot_regression_scatter,
    plot_roc_curve,
    plot_seqid_distribution,
    plot_train_test_distribution,
)
from src.helpers.mlflow_helpers import setup_mlflow
from src.helpers.stats_utilities import calculate_statistics_from_df
from src.model.composite_model import CompositeModel


# This is the main script for training models
logger = logging.getLogger(__name__)
TIMESTAMP_FORMAT = "%Y-%m-%d-%H%M"


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def my_app(cfg: DictConfig):  # noqa: PLR0915 C901 Too long and too complex a function - consider breaking it down.
    """
    This function is the entry point of the application.

    Args:
        cfg (DictConfig): The configuration object containing the application settings.
    """
    timestamp = datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)
    logger.info(f"Workflow configuration:\n\n{OmegaConf.to_yaml(cfg)}")

    if cfg.predictor.model_type is None:  # if none, both classification and regression is allowed
        cfg.predictor.model_type = cfg.dataset.task
    assert cfg.dataset.task == cfg.predictor.model_type, (
        f"The data type {cfg.dataset.task} doesn't match the predictor model type {cfg.predictor.model_type}"
    )

    composite_model = CompositeModel(cfg)

    setup_mlflow(cfg, composite_model)

    with mlflow.start_run() as run:
        logger.info(run.info)
        dir_name = (
            f"{timestamp}_{run.info.run_id}"  # run_id is necessary when using datalab mlflow server
        )
        mlflow.log_params(
            pd.json_normalize(OmegaConf.to_container(cfg, resolve=True)).iloc[0].to_dict()
        )

        # Train the model
        logger.info("Compute embeddings on-the-fly and train model")
        model_output_train, model_output_val = composite_model.train_predictor_model()
        composite_model.store_model(dir_name)

        mlflow.log_param("model_type", cfg.predictor.model_type)

        model_output_test = composite_model.forward()

        model_output_train.describe_data()
        model_output_val.describe_data()
        model_output_test.describe_data()

        # Create dataframe storing the predictions
        pred_df = pd.concat(
            [
                model_output_train.get_dataframe(),
                model_output_val.get_dataframe(),
                model_output_test.get_dataframe(),
            ],
            ignore_index=True,
        )
        pred_df.to_csv("pred_df.csv")
        stats_df = calculate_statistics_from_df(
            pred_df,
            "y",
            "y_pred",
            "y_pred_prob",
            cfg.predictor.model_type,
            "data_split",
            "",
        )
        if cfg.dataset.group_column is not None:
            stats_group_df = calculate_statistics_from_df(
                pred_df,
                "y",
                "y_pred",
                "y_pred_prob",
                cfg.predictor.model_type,
                "data_split",
                "group",
            )

        if cfg.plots.generate_plots is True:
            figures_df_path_base = Path(
                cfg.persistence.artifacts_root_folder,
                dir_name,
            )
            logger.info("Starting to set up plots")

            logger.info("Setting up % of sequence identity plots")
            plot_seqid_distribution(
                model_output_train.sequences,
                model_output_test.sequences,
                figure_path=figures_df_path_base.joinpath("sequence_identity_distribution.png"),
            )
            logger.info("Plotting the cosine similiarity histograms for embeddings")
            plot_cosine_similarity(
                model_output_train.sequence_embeddings,
                model_output_test.sequence_embeddings,
                figure_path=figures_df_path_base.joinpath(
                    "cosine_similiarity_training_vs_test_distribution.png"
                ),
            )

            if cfg.predictor.model_type == "regression":
                logger.info("Setting up training vs test distribution of target values")

                plot_train_test_distribution(
                    model_output_train.labels,
                    model_output_test.labels,
                    figure_path=figures_df_path_base.joinpath("training_vs_test_distribution.png"),
                )

                logger.info("Logging regression plots for training")
                plot_regression_scatter(
                    pred_df[pred_df["data_split"] == "train"],
                    hue="group" if cfg.dataset.group_column is not None else None,
                    figure_path=figures_df_path_base.joinpath("regression_scatter_plot_train.png"),
                    label="Train",
                )

                logger.info("Setting up regression plots for testing")

                plot_regression_scatter(
                    pred_df[pred_df["data_split"] == "test"],
                    hue="group" if cfg.dataset.group_column is not None else None,
                    figure_path=figures_df_path_base.joinpath("regression_scatter_plot_test.png"),
                )

            if cfg.predictor.model_type == "classification_binary":
                if "tp" in stats_df.columns:
                    logger.info("Setting up confusion_matrix_plot")

                    plot_confusion_matrix(
                        np.array(model_output_test.labels),
                        np.array(model_output_test.predictions),
                        figure_path=figures_df_path_base.joinpath(
                            "confusion_matrix_plot_testset.png"
                        ),
                    )
                    logger.info("Setting up roc curve plot")
                    plot_roc_curve(
                        model_output_test.labels,
                        model_output_test.predictions_probability,
                        figure_path=figures_df_path_base.joinpath("roc_curve_testset.png"),
                        model_name=cfg.predictor.class_name,
                    )
                    logger.info("Setting up precision_recall_plot")
                    plot_precision_recall(
                        model_output_test.labels,
                        model_output_test.predictions_probability,
                        figure_path=figures_df_path_base.joinpath(
                            "precision_recall_curve_testset.png"
                        ),
                        model_name=cfg.predictor.class_name,
                    )
                    if cfg.dimred.class_name == "PCADimReduction":
                        logger.info("Setting up plot with hue ")

                        plot_pca_components_with_hue(
                            model_output_train.features_concat_dimred_concat,
                            model_output_test.features_concat_dimred_concat,
                            model_output_train.labels,
                            model_output_test.labels,
                            figure_path=figures_df_path_base.joinpath("pca_analysis_with_hue.png"),
                        )
                else:
                    logger.info("no confusion matrix for these plot")

        # Setting up the base path
        base_path = Path(cfg.persistence.artifacts_root_folder, dir_name)
        model_name = composite_model.get_model_name()

        predictions_df_path = base_path / f"{model_name}_predictions.csv"
        pred_df.to_csv(predictions_df_path)
        mlflow.log_artifact(predictions_df_path)

        stats_df_path = base_path / f"{model_name}_stats.csv"
        stats_df.to_csv(stats_df_path)
        mlflow.log_artifact(stats_df_path)

        # EHEC (do we need this still?)
        data = stats_df.to_dict(orient="index")[0]
        data = {
            "run_name": run.info.run_name,
            "run_id": run.info.run_id,
            "data_name": cfg.dataset.data_name,
        }
        for name, split_df in stats_df.groupby("data_split"):
            data[name] = dict(split_df.iloc[0])

        yaml_path = base_path / f"{model_name}_stats_metrics.yaml"
        with open(yaml_path, "w") as file:
            yaml.dump(data, file, default_flow_style=False)

        # Then save the stats group if condition is not None
        if cfg.dataset.group_column is not None:
            stats_group_df_path = base_path / f"{model_name}_group_stats.csv"
            stats_group_df.to_csv(stats_group_df_path)
            mlflow.log_artifact(stats_group_df_path)

        logger.info("Done!")

        def print_auto_logged_info(r):
            tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
            artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
            print(f"run_id: {r.info.run_id}")
            print(f"artifacts: {artifacts}")
            print(f"params: {r.data.params}")
            print(f"metrics: {r.data.metrics}")
            print(f"tags: {tags}")
            print(f"results stored in {stats_df_path}")

        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
        print(mlflow.get_artifact_uri())
        print(mlflow.doctor())


if __name__ == "__main__":
    my_app()
