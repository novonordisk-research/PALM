import logging

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from src.helpers.utilities import find_optimal_cutoff


logger = logging.getLogger(__name__)


def get_mlflow_metric_name(split_name: str, group_name: str, metric_name: str) -> str:
    """Define a '.'-separated name given the split, group, and metric"""
    items = [x for x in [split_name, group_name, metric_name] if x != ""]
    return ".".join(items)


def calculate_classification_statistics(  # noqa: C901, PLR0913 Too complex, too many arguments
    y: np.array,
    y_pred: np.array,
    y_pred_prob: np.array,
    split_name: str,
    group_name: str,
    verbose: bool = False,
) -> dict:
    """
    Calculate various classification statistics for a group of data points.

    Parameters:
        y: the labels
        y_pred: the predicted labels
        y_pred_prob: the probability of the labels
        split_name: the split for which the statistics are being computed
        group_name: the group for which the statistics are being computed
        verbose: whether to print additional information. Defaults to False.

    Returns:
        dict: A dictionary containing the calculated statistics.

    """

    def calc_precision(tp, fp):
        """Calculate the precision.

        Precision is the fraction of items that are predicted to be positive that are truly labeled as positive.

        Parameters:
            tp (int): The number of true positives.
            fp (int): The number of false positives.

        Returns:
            float: The precision value.

        """
        denom = tp + fp
        if denom == 0:
            return np.nan
        return tp / denom

    def calc_recall(tp, fn):
        """Calculate the recall.

        Recall is the fraction of positive items that are correctly identified.

        Parameters:
            tp (int): The number of true positives.
            fn (int): The number of false negatives.

        Returns:
            float: The recall value.

        """
        denom = tp + fn
        if denom == 0:
            return np.nan
        return tp / denom

    def calc_true_negative_rate(tn, fp):
        """Calculate the true negative rate.

        True negative rate, also known as specificity, is the fraction of negative items that are correctly predicted.

        Parameters:
            tn (int): The number of true negatives.
            fp (int): The number of false positives.

        Returns:
            float: The true negative rate value.

        """
        denom = tn + fp
        if denom == 0:
            return np.nan
        return tn / denom

    def calc_f_score(tp, fp, fn):
        """Calculate the F-score.

        The F-score is the harmonic mean of precision and recall.

        Parameters:
            tp (int): The number of true positives.
            fp (int): The number of false positives.
            fn (int): The number of false negatives.

        Returns:
            float: The F-score value.

        """

        denom = 2 * tp + fp + fn
        if denom == 0:
            return np.nan
        return 2 * tp / denom

    if not (len(y) == len(y_pred) == len(y_pred_prob)):
        raise ValueError(
            f"Predictions do not have the same shape: {y.shape}, {y_pred.shape}, {y_pred_prob.shape}"
        )
    if np.isnan(y_pred).any():
        raise ValueError(f"y_pred contains NaNs: {y_pred}")
    if np.isnan(y_pred_prob).any():
        raise ValueError(f"y_pred_prob contains NaNs: {y_pred_prob}")

    # Calculate true positives, false positives, and false negatives
    n_total = len(y)
    n_correct = (y == y_pred).sum()
    tp = ((y == 1) & (y_pred == 1)).sum()
    fp = ((y == 0) & (y_pred == 1)).sum()
    tn = ((y == 0) & (y_pred == 0)).sum()
    fn = ((y == 1) & (y_pred == 0)).sum()

    if verbose:
        logger.info(
            f"Total: {n_total}\nCorrect: {n_correct}\nTP: {tp}\nTN: {tn}\nFP: {fp}\nFN: {fn}"
        )

    frac_correct = float(n_correct / n_total)
    balanced_accuracy = float(balanced_accuracy_score(y, y_pred))
    precision = float(calc_precision(tp, fp))
    recall = float(calc_recall(tp, fn))
    true_neg_rate = float(calc_true_negative_rate(tn, fp))
    matt_corr_coef = float(matthews_corrcoef(y, y_pred))
    fscore = float(calc_f_score(tp, fp, fn))

    if np.isnan(y_pred_prob).any():
        rocauc, optimal_cutoff, optimal_mcc = np.nan, np.nan, np.nan
    else:
        rocauc = float(roc_auc_score(y, y_pred_prob))
        optimal_cutoff, optimal_mcc = find_optimal_cutoff(y, y_pred_prob)

    mlflow.log_metrics(
        {
            get_mlflow_metric_name(split_name, group_name, "accuracy"): frac_correct,
            get_mlflow_metric_name(split_name, group_name, "precision"): precision,
            get_mlflow_metric_name(split_name, group_name, "recall"): recall,
            get_mlflow_metric_name(split_name, group_name, "specificity"): true_neg_rate,
            get_mlflow_metric_name(split_name, group_name, "f_score"): fscore,
            get_mlflow_metric_name(split_name, group_name, "matthews_corr_coef"): matt_corr_coef,
            get_mlflow_metric_name(split_name, group_name, "optimal_mcc"): optimal_mcc,
            get_mlflow_metric_name(split_name, group_name, "optimal_mcc_cutoff"): optimal_cutoff,
            get_mlflow_metric_name(split_name, group_name, "balanced_accuracy"): balanced_accuracy,
            get_mlflow_metric_name(split_name, group_name, "roc_auc_score"): rocauc,
        }
    )

    if verbose:
        logger.info(
            f"Fraction correctly predicted: {frac_correct:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nSpecificity: {true_neg_rate:.2f}\nCorr: {matt_corr_coef:.2f}\nBalanced accuracy:{balanced_accuracy:.2f}"
        )

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tp+tn": n_correct,
        "total": n_total,
        "accuracy": frac_correct,
        "precision": precision,
        "recall": recall,
        "specificity": true_neg_rate,
        "f_score": fscore,
        "matthews_corr_coef": matt_corr_coef,
        "balanced_accuracy": balanced_accuracy,
        "roc_auc_score": rocauc,
        "optimal_mcc_cutoff": optimal_cutoff,
        "optimal_mcc": optimal_mcc,
    }


def calculate_regression_statistics(
    y: np.array, y_pred: np.array, split_name, group_name, verbose=False
) -> dict:
    """
    Calculate various regression statistics for a group of data points.

    Parameters:
        y: the labels
        y_pred: the predicted labels
        split_name: the split for which the statistics are being computed
        group_name: the group for which the statistics are being computed
        verbose: whether to print additional information. Defaults to False.

    Returns:
        dict: A dictionary containing the calculated statistics.

    """

    if not (len(y) == len(y_pred)):
        raise ValueError(f"Predictions do not have the same shape: {y.shape}, {y_pred.shape}")

    # Calculate metrics
    if len(y) > 2:  # noqa: PLR2004 Magic values comparison, consider using a constant with a meaningful name
        pearson_r_val = float(pearsonr(y, y_pred)[0])
        spearman_r_val = float(spearmanr(y, y_pred)[0])
        r_squared = float(r2_score(y, y_pred))
    else:
        pearson_r_val = np.nan
        spearman_r_val = np.nan
        r_squared = np.nan
    mse_val = float(mean_squared_error(y, y_pred))
    mae_val = float(mean_absolute_error(y, y_pred))

    if group_name == "":
        # For now, the group-specific metrics are not logged to mflow
        mlflow.log_metrics(
            {
                get_mlflow_metric_name(split_name, group_name, "pearson_r"): pearson_r_val,
                get_mlflow_metric_name(split_name, group_name, "spearman_r"): spearman_r_val,
                get_mlflow_metric_name(split_name, group_name, "mse"): mse_val,
                get_mlflow_metric_name(split_name, group_name, "mae"): mae_val,
                get_mlflow_metric_name(split_name, group_name, "r_squared"): r_squared,
            }
        )

    if verbose:
        logger.info(
            f"Pearson R: {pearson_r_val:.2f}\nSpearman R: {spearman_r_val:.2f}\nMSE: {mse_val:.2f}\nMAE: {mae_val:.2f}\nR_squared:{r_squared:.2f}"
        )

    return {
        "pearson_r": pearson_r_val,
        "spearman_r": spearman_r_val,
        "mse": mse_val,
        "mae": mae_val,
        "r_squared": r_squared,
    }


def calculate_statistics_from_df(  # noqa: PLR0913 too many arguments.
    df, exp_val, pred_val, probabilities, task, split_col, group_col="", verbose=False
) -> pd.DataFrame:
    """
    Calculate statistics from a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        exp_val (str): The column name for the expected values.
        pred_val (str): The column name for the predicted values.
        probabilities (str): The column name for the prediction probabilities.
        task (str): The prediction task ("classification_binary","regression")
        split_col (str): The column name for grouping the DataFrame by splits..
        group_col (str, optional): The column name for grouping the DataFrame. Defaults to ''.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated statistics.
    """

    statistics_list = []

    groups = (
        {group_name: group_df for group_name, group_df in df.groupby(group_col)}
        if group_col != ""
        else {"": df}
    )
    for group_name, group_df in groups.items():
        for data_split, split_group_df in group_df.groupby(split_col):
            if task == "classification_binary":
                stat_dict = calculate_classification_statistics(
                    np.array(split_group_df[exp_val]),
                    np.array(split_group_df[pred_val]),
                    np.array(split_group_df[probabilities]),
                    data_split,
                    group_name,
                    verbose,
                )
            elif task == "regression":
                stat_dict = calculate_regression_statistics(
                    np.array(split_group_df[exp_val]),
                    np.array(split_group_df[pred_val]),
                    data_split,
                    group_name,
                    verbose,
                )
            else:
                raise ValueError(f"Prediction task: {task} not recognized")
            stat_dict["group"] = group_name
            stat_dict["data_split"] = data_split
            statistics_list.append(stat_dict)

    return pd.DataFrame(statistics_list)
