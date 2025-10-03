import logging
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    auc,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_curve,
)
from sklearn.metrics.pairwise import cosine_similarity

from src.helpers.plot_utilities import SequenceSplitter


logger = logging.getLogger(__name__)


def get_novo_colors():
    color_dict = {
        "darkblue": "#001965",
        "yellow": "#F3AB00",
        "red": "#E6553F",
        "green": "#3F9C3",
        "lightblue_strong": "#3B97DE",
        "lightblue_medium": "#B1D5F2",
        "lightblue_weak": "#D8EAF8",
        "mediumblue_strong": "#005AD2",
        "mediumblue_medium": "#99BDED",
        "mediumblue_weak": "#CCDEF6",
        "pink_strong": "#EEA7BF",
        "pink_medium": "#F8DCE5",
        "pink_weak": "#FCEDF2",
        "teal_strong": "#2A918B",
        "teal_medium": "#AAD3D1",
        "teal_weak": "#D4E9E8",
        "gray_strong": "#939AA7",
        "gray_medium": "#D4D7DC",
        "gray_weak": "#E9EBED",
    }
    return color_dict


def get_novo_cmap():
    color_dict = get_novo_colors()
    cmap = [v for k, v in color_dict.items() if not (k.endswith("_medium") or k.endswith("_weak"))]
    cmap = ListedColormap(cmap)
    return cmap


def plot_regression_scatter(df, hue=None, figure_path="regression_scatter_plot.png", label="Test"):
    """
    Generates a scatter plot comparing experimental results with predicted results, including MSE, MAE, and R-squared metrics.

    Args:
        df (DataFrame): DataFrame containing 'y' (true values) and 'y_pred' (predicted values) columns.
        figure_path (str): Path to save the scatter plot image.
        label (str): Label indicating the set being plotted (e.g., 'Test' or 'Train').
    """

    mse = mean_squared_error(df["y"], df["y_pred"])
    mae = mean_absolute_error(df["y"], df["y_pred"])
    pearson_r = pearsonr(df["y"], df["y_pred"])[0]
    spearman_r = spearmanr(df["y"], df["y_pred"])[0]
    r_squared = r2_score(df["y"], df["y_pred"])
    sns.set_theme(style="ticks")  # Set the theme for Seaborn
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="y",
        y="y_pred",
        hue=hue,
        color=get_novo_colors()["darkblue"] if hue is None else None,
        palette="viridis" if hue is not None else None,
        alpha=0.8,
        ax=ax,
    )
    plt.title(f"Experimental vs Predicted Results {label} Set")
    plt.xlabel("Experimental Results")
    plt.ylabel("Predicted Results")
    textstr = f"MSE: {mse:.2f}\nMAE: {mae:.2f}\nPearson_R: {pearson_r:.2f}\nSpearman_R:{spearman_r:.2f}\nR_squared:{r_squared:.2f}"
    props = dict(boxstyle="round", facecolor="white", alpha=0.5)
    plt.text(
        0.95,
        0.05,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
    )
    plt.axis("square")
    ax.axline((1, 1), slope=1, color="k", lw=1, linestyle=":")
    if hue is not None:
        plt.legend()
        h, l = ax.get_legend_handles_labels()  # noqa: E741 Ambigous variable names

        def add_label(label):
            n = len(df[df[hue] == label])
            if n > 2:  # noqa: PLR2004 Magic value 2 - consider to define variable with meaningful name
                spearman_r = spearmanr(df[df[hue] == label]["y"], df[df[hue] == label]["y_pred"])[0]
                spearman_r = "%.2f" % spearman_r
            else:
                spearman_r = "N/A"

            label = label + " (n=%i, sp_r=%s)" % (n, spearman_r)
            return label

        ax.legend(h, [add_label(i) for i in l])
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(figure_path)


def plot_confusion_matrix(y_true, y_pred, figure_path="confusion_matrix_plot.png"):
    """
    Generates a heatmap representing the confusion matrix for classification results.

    Args:
        df (DataFrame): DataFrame containing 'tp', 'fp', 'tn', 'fn' values.
        figure_path (str): Path to save the confusion matrix image.
    """
    # Extracting values from the DataFrame
    conf_matrix = confusion_matrix(y_true, y_pred, normalize="true")
    conf_matrix_unnorm = confusion_matrix(y_true, y_pred)

    logger.info(f"This is how the matrix looks like {conf_matrix} ")
    confusion_df = pd.DataFrame(
        conf_matrix,
        index=["Actual Positive", "Actual Negative"],
        columns=["Predicted Positive", "Predicted Negative"],
    )
    n_rows, n_cols = conf_matrix.shape
    annot_labels = [
        [
            "%s\n%2.1f%%"
            % (format(conf_matrix_unnorm[i, j], ",").replace(",", "'"), conf_matrix[i, j] * 100)
            for j in range(n_cols)
        ]
        for i in range(n_rows)
    ]

    sns.set(font_scale=1.4)  # for label size
    plt.figure(figsize=(8, 6))
    confs_matrix_plot = sns.heatmap(confusion_df, annot=annot_labels, fmt="s", cmap="Blues")
    plt.title("Normalized Confusion Matrix Test Set")
    confs_matrix_plot.figure.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(figure_path)


def plot_roc_curve(y_test, y_pred_prob, figure_path="roc_curve.png", model_name=""):
    """
    Generates an ROC curve plot for classification results.

    Args:
        df (DataFrame): DataFrame containing 'y' (true labels) and 'y_pred' (predicted probabilities) columns.
        figure_path (str): Path to save the ROC curve image.
        model_name (str): Name of the model to label the plot.
    """
    # retrieve just the probabilities of the positive class
    if y_pred_prob is None:
        logger.info(
            "The y probabilistic predictions for this model are set to None roc and auc no plots being plotted for those models"
        )
    else:
        pos_probs = y_pred_prob
        fpr, tpr, _ = roc_curve(y_test, pos_probs)
        roc_auc = auc(fpr, tpr)
        sns.set_theme(style="ticks")  # Set the theme for Seaborn
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC = {roc_auc:.2f})",
            color=get_novo_colors()["darkblue"],
        )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve test set")
        plt.legend(loc="lower right")  # Specify the legend location
        plt.tight_layout()  # Increase the padding around the plot
        plt.savefig(figure_path, dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(figure_path)


def plot_precision_recall(
    y_test, y_pred_prob, figure_path="precision_recall_curve.png", model_name=""
):
    """
    Generates a precision-recall curve plot for classification results.

    Args:
        df (DataFrame): DataFrame containing 'y' (true labels) and 'y_pred' (predicted probabilities) columns.
        figure_path (str): Path to save the precision-recall curve image.
        model_name (str): Name of the model to label the plot.
    """
    if y_pred_prob is None:
        logger.info(
            "The y probabilistic predictions for this model are set to None roc and auc no plots being plotted for those models"
        )
    else:
        pos_probs = y_pred_prob
        # retrieve just the probabilities for the positive class
        # calculate the no skill l
        lr_precision, lr_recall, _ = precision_recall_curve(y_test, pos_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(
            lr_recall,
            lr_precision,
            marker=".",
            label=model_name,
            color=get_novo_colors()["darkblue"],
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision Recall Curve test set")
        plt.legend()
        plt.savefig(figure_path, dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(figure_path)
        # instead of logging an actual local file, one could also hand over the figure object to mlflow
        # mlflow.log_figure(fig, "precision_recall_curve.png")


def plot_train_test_distribution(y_train, y_test, figure_path="training_vs_test_distribution.png"):
    """
    Generates histograms for the distribution of target values in the training and testing sets.

    Args:
        y_train (array-like): Target values for the training set.
        y_test (array-like): Target values for the testing set.
        figure_path (str): Path to save the distribution plot image.
    """
    plt.figure(figsize=(10, 6))

    # helper function to repeatily call plotting
    def hist_plot(y_train, y_test):
        # Plot the distribution of the training set
        sns.histplot(
            y_train,
            color=get_novo_colors()["darkblue"],
            label="Train Set",
            kde=False,
            bins=20,
            stat="density",
            linewidth=0,
        )

        # Plot the distribution of the testing set
        sns.histplot(
            y_test,
            color=get_novo_colors()["teal_strong"],
            label="Test Set",
            kde=False,
            bins=20,
            stat="density",
            linewidth=0,
        )

        plt.xlabel("Target Value")
        plt.ylabel("Density")
        plt.title("Train vs Test Distribution of Target Values")
        plt.legend()
        plt.savefig(figure_path, dpi=300)
        plt.close()
        mlflow.log_artifact(figure_path)

    try:
        # old behaviour that fails for too narrow distributions
        hist_plot(y_train, y_test)
    except ValueError:
        # fall back to SNS default
        hist_plot(y_train, y_test, binwidth=None)


def plot_seqid_distribution(
    train_seqs, test_seqs, figure_path="sequence_identity_distribution.png"
):
    """
    Generates histograms for the distribution of sequence identity within the training set, within the testing set, and between the training and testing sets.

    Args:
        train_seqs (list[str]): List of training sequences.
        test_seqs (list[str]): List of testing sequences.
        figure_path (str): Path to save the sequence identity distribution plot image.
    """
    splitter = SequenceSplitter(list(set(train_seqs)), list(set(test_seqs)))
    # Manually set the train_inds and test_inds based on predefined splits
    intra_train, intra_test, inter_train_test = splitter.plot_dist(similarity=True)
    plt.figure(figsize=(12, 8))
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))  # 1 row, 3 columns
    # Plot Train vs Train histogram
    axes[0].hist(
        intra_train,
        bins=20,
        range=[0, 1],
        alpha=0.7,
        label="Train vs Train",
        color=get_novo_colors()["darkblue"],
    )
    axes[0].set_title("Train vs Train")
    axes[0].set_xlabel("Distance")
    axes[0].set_ylabel("Frequency")
    # axes[0].legend()

    # Plot Test vs Test histogram
    axes[1].hist(
        intra_test,
        bins=20,
        range=[0, 1],
        alpha=0.7,
        label="Test vs Test",
        color=get_novo_colors()["teal_strong"],
    )
    axes[1].set_title("Test vs Test")
    axes[1].set_xlabel("Distance")
    axes[1].set_ylabel("Frequency")
    # axes[1].legend()

    # Plot Train vs Test histogram
    axes[2].hist(
        inter_train_test,
        bins=20,
        range=[0, 1],
        alpha=0.7,
        label="Train vs Test",
        color=get_novo_colors()["yellow"],
    )
    axes[2].set_title("Train vs Test")
    axes[2].set_xlabel("Distance")
    axes[2].set_ylabel("Frequency")
    # axes[2].legend()

    fig.suptitle("Distribution of Sequence identity along each label", fontsize=16)
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close()
    plt.show()
    mlflow.log_artifact(figure_path)


def plot_cosine_similarity(
    x_train, x_test, figure_path="cosine_similiarity_training_vs_test_distribution.png"
):
    """
    Generates histograms for the distribution of cosine similarity within the training set, within the testing set, and between the training and testing sets.

    Args:
        x_train (list[str]): List of embedded training sequences.
        x_test (list[str]): List of embedded testing sequences.
        figure_path (str): Path to save the sequence identity distribution plot image.
    """
    # Calculating cosine similarity between train and train
    similarity_train_train = cosine_similarity(x_train, x_train)

    # Calculating cosine similarity between test and test
    similarity_test_test = cosine_similarity(x_test, x_test)

    # Calculating cosine similarity between train and test
    similarity_train_test = cosine_similarity(x_train, x_test)

    plt.figure(figsize=(12, 8))
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))  # 1 row, 3 columns
    # Plot Train vs Train histogram
    axes[0].hist(
        similarity_train_train.flatten(),
        bins=20,
        range=[0, 1],
        alpha=0.7,
        label="Train vs Train",
        color=get_novo_colors()["darkblue"],
    )
    axes[0].set_title("Train vs Train")
    axes[0].set_xlabel("Cosine_similarity")
    axes[0].set_ylabel("Frequency")
    # axes[0].legend()

    # Plot Test vs Test histogram
    axes[1].hist(
        similarity_test_test.flatten(),
        bins=20,
        range=[0, 1],
        alpha=0.7,
        label="Test vs Test",
        color=get_novo_colors()["teal_strong"],
    )
    axes[1].set_title("Test vs Test")
    axes[1].set_xlabel("Cosine_similarity")
    axes[1].set_ylabel("Frequency")
    # axes[1].legend()

    # Plot Train vs Test histogram
    axes[2].hist(
        similarity_train_test.flatten(),
        bins=20,
        range=[0, 1],
        alpha=0.7,
        label="Train vs Test",
        color=get_novo_colors()["yellow"],
    )
    axes[2].set_title("Train vs Test")
    axes[2].set_xlabel("Cosine_similarity")
    axes[2].set_ylabel("Frequency")
    # axes[2].legend()

    fig.suptitle("Distribution of Cosine Similarity of Embeddings", fontsize=16)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close()
    plt.show()

    mlflow.log_artifact(figure_path)


def plot_pca_components_with_hue(
    x_embed_compressed_train,
    x_embed_compressed_test,
    y_train,
    y_test,
    figure_path="pca_analysis_with_hue.png",
):
    # Plotting the first two components of the train data with hue based on y_train
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    sns.set_theme(style="ticks")  # Set the theme for Seaborn
    sns.scatterplot(
        x=x_embed_compressed_train[:, 0],
        y=x_embed_compressed_train[:, 1],
        hue=y_train,
        palette="viridis",
        legend="full",
        ax=axes[0],
        color=get_novo_colors()["darkblue"],
        alpha=0.8,
    )
    axes[0].set_title("Train Data - First Two Components")
    axes[0].set_xlabel("Principal Component 1")
    axes[0].set_ylabel("Principal Component 2")
    axes[0].legend(title="Classes", loc="upper right")
    # axes[0].set_xlim([-5, 10.0])
    # axes[0].set_ylim([-5, 10.0])

    # Plotting the first two components of the test data with hue based on y_test
    sns.scatterplot(
        x=x_embed_compressed_test[:, 0],
        y=x_embed_compressed_test[:, 1],
        hue=y_test,
        palette="viridis",
        legend="full",
        ax=axes[1],
        color=get_novo_colors()["teal_strong"],
        alpha=0.8,
    )
    axes[1].set_title("Test Data - First Two Components")
    axes[1].set_xlabel("Principal Component 1")
    axes[1].set_ylabel("Principal Component 2")
    axes[1].legend(title="Classes", loc="upper right")
    # axes[1].set_xlim([-5, 10.0])
    # axes[1].set_ylim([-5, 10.0])
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(figure_path)


##test the different plot
