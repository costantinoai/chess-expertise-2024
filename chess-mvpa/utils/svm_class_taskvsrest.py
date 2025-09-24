#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eye-tracking Data Analysis for Chess Expertise Classification

This script performs SVM decoding analysis on eye-tracking data to task and rest trials.
It uses stratified k-fold cross-validation with balanced training sets and implements various statistical tests and visualizations to evaluate the model's performance.

Created on Tue Aug 13 17:51:34 2024
@author: costantino_ai
"""

# Standard library imports
import glob
import json
import os
import random
import shutil
import inspect
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime
import logging

# Third-party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import binomtest, ttest_1samp
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

def setup_logging(outdir: str) -> logging.Logger:
    """
    Set up logging configuration for the script.

    Args:
        outdir (str): The output directory where the log file will be stored.

    Returns:
        logging.Logger: Configured logger object.
    """
    log_file = os.path.join(outdir, "analysis_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def load_eye_tracking_data(
    root_dir: str, participants: List[Tuple[str, bool]]
) -> pd.DataFrame:
    """
    Load and preprocess eye tracking data from the specified directory for the given participants.

    Args:
        root_dir (str): Path to the root directory containing subject data.
        participants (List[Tuple[str, bool]]): List of tuples containing subject IDs and expert status.

    Returns:
        pd.DataFrame: Combined and preprocessed eye tracking data for all subjects.
    """
    logger.info(f"Loading eye tracking data from {root_dir}")
    root_path = Path(root_dir)
    subjects = [
        d for d in root_path.iterdir() if d.is_dir() and d.name.startswith("sub-")
    ]

    expert_dict = dict(participants)
    all_data = []

    for subject in subjects:
        logger.debug(f"Processing subject: {subject.name}")
        func_dir = subject / "func"

        for eyetrack_file in func_dir.glob("*_desc-1to6_eyetrack.tsv"):
            # Read the TSV file
            df = pd.read_csv(eyetrack_file, sep="\t")

            # Extract subject ID and run number from the filename
            parts = eyetrack_file.stem.split("_")
            subject_id = parts[0]
            run = next(p for p in parts if p.startswith("run-")).split("-")[1]

            # Add subject, run, and expert status to the dataframe
            df["subject"] = subject_id
            df["run"] = run
            df["expert"] = expert_dict.get(subject_id, np.nan)

            # Load and add metadata from the corresponding JSON file
            json_file = eyetrack_file.with_suffix(".json")
            with json_file.open() as f:
                metadata = json.load(f)

            for key, value in metadata.items():
                if isinstance(value, (str, int, float)):
                    df[key] = value

            all_data.append(df)

    # Combine all data into a single dataframe
    combined_data = pd.concat(all_data, ignore_index=True)
    logger.info(
        f"Loaded eye-tracking data: {len(combined_data)} rows, {combined_data['subject'].nunique()} subjects"
    )

    # Sort combined_data
    combined_data = combined_data.sort_values(
        ["subject", "run", "timestamp"]
    ).reset_index(drop=True)

    return combined_data


# def prepare_features(eye_tracking_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Prepare features for SVM classification from eye-tracking data.

#     Args:
#         eye_tracking_data (pd.DataFrame): The raw eye-tracking data.

#     Returns:
#         Tuple[np.ndarray, np.ndarray, np.ndarray]: Features, labels, and group identifiers.
#     """
#     # Group the data by subject
#     grouped_by_subject = eye_tracking_data.groupby('subject')

#     all_subject_data = []
#     all_labels = []
#     all_groups = []

#     for subject, subject_data in grouped_by_subject:
#         # Group each subject's data by run
#         subject_runs = subject_data.groupby('run')

#         subject_run_data = []
#         for _, run_data in subject_runs:
#             # Extract x and y coordinates, limiting to the minimum number of timepoints
#             X = run_data[['x_coordinate', 'y_coordinate']].values
#             subject_run_data.append(X.flatten())

#         # Add the data for all runs of this subject
#         all_subject_data.extend(subject_run_data)
#         # Add labels (expert status) for each run
#         all_labels.extend([subject_data['expert'].iloc[0]] * len(subject_run_data))
#         # Add group identifiers (subject ID) for each run
#         all_groups.extend([subject] * len(subject_run_data))

#     return np.array(all_subject_data), np.array(all_labels), np.array(all_groups)


def prepare_features_taskvsrest(
    eye_tracking_data: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare features for SVM classification from eye-tracking data.

    Args:
        eye_tracking_data (pd.DataFrame): The raw eye-tracking data.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Features, labels, and group identifiers.
    """

    eye_tracking_data["trial_type_binary"] = eye_tracking_data["trial_type"].apply(
        lambda x: "task" if x != "fixation" else x
    )

    X = eye_tracking_data["displacement"].values
    # X = eye_tracking_data[['x_coordinate', 'y_coordinate']].values
    X = np.nan_to_num(X, copy=True)

    y = np.array([
        1 if trial == "task" else 0
        for trial in eye_tracking_data["trial_type_binary"].values
    ])

    sub = eye_tracking_data["subject"].values

    return X, y, sub


# def run_svm_decoding(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict:
#     """
#     Perform SVM decoding analysis using stratified group k-fold cross-validation.

#     Args:
#         X (np.ndarray): Feature matrix.
#         y (np.ndarray): Labels.
#         groups (np.ndarray): Group identifiers for cross-validation.

#     Returns:
#         Dict: A dictionary containing evaluation metrics and predictions.
#     """
#     # Initialize stratified group k-fold cross-validation
#     skf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)

#     y_true, y_pred, y_pred_proba = [], [], []
#     fold_accuracies = []

#     for fold, (train_index, test_index) in enumerate(skf.split(X, y, groups)):
#         # Split the data into training and test sets
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         groups_test = groups[test_index]

#         logger.info(f"Fold {fold + 1} - Held-out subjects: {np.unique(groups_test)}")

#         # Create and train the SVM model
#         svm = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
#         svm.fit(X_train, y_train)

#         # Make predictions on the test set
#         fold_pred = svm.predict(X_test)
#         fold_pred_proba = svm.predict_proba(X_test)[:, 1]
#         fold_accuracy = accuracy_score(y_test, fold_pred)
#         fold_accuracies.append(fold_accuracy)

#         logger.info(f"Fold {fold + 1} Accuracy: {fold_accuracy:.3f}")
#         for i, (true, pred, prob) in enumerate(zip(y_test, fold_pred, fold_pred_proba)):
#             logger.info(f"Fold {fold + 1}, Sample {i + 1}: Target={true}, Pred={pred}, Prob={prob:.3f}, Subject={groups_test[i]}")

#         # Collect results
#         y_true.extend(y_test)
#         y_pred.extend(fold_pred)
#         y_pred_proba.extend(fold_pred_proba)

#     # Convert lists to numpy arrays
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     y_pred_proba = np.array(y_pred_proba)

#     # Calculate evaluation metrics
#     overall_accuracy = np.mean(fold_accuracies)
#     balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred, average='weighted')

#     # Perform statistical tests
#     t_stat, t_pvalue = ttest_1samp(fold_accuracies, 0.5)
#     n_samples = len(y_true)
#     n_correct = int(overall_accuracy * n_samples)
#     binom_result = binomtest(n_correct, n_samples, p=0.5)

#     # Log results
#     logger.info(f"Overall Accuracy: {overall_accuracy:.3f}")
#     logger.info(f"Balanced Accuracy: {balanced_accuracy:.3f}")
#     logger.info(f"F1 Score: {f1:.3f}")
#     logger.info(f"T-test p-value: {t_pvalue:.4f}")
#     logger.info(f"Binomial test p-value: {binom_result.pvalue:.4f}")

#     # Return results as a dictionary
#     return {
#         'accuracy': overall_accuracy,
#         'balanced_accuracy': balanced_accuracy,
#         'f1_score': f1,
#         'y_true': y_true,
#         'y_pred': y_pred,
#         'y_pred_proba': y_pred_proba,
#         'fold_accuracies': fold_accuracies,
#         't_pvalue': t_pvalue,
#         'binom_test_pvalue': binom_result.pvalue
#     }

def balance_dataset(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance the dataset by undersampling the majority class.

    This function ensures that all classes have the same number of samples
    as the minority class. It uses random undersampling for the majority classes.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Label array of shape (n_samples,).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Balanced feature matrix and label array.

    Logs:
        - Original and balanced class distribution
        - Minority class information
        - Whether the dataset is perfectly balanced after resampling
        - Original and balanced dataset shapes
    """
    classes, counts = np.unique(y, return_counts=True)
    logger.info(f"Original class distribution: {dict(zip(classes, counts))}")

    minority_class = classes[np.argmin(counts)]
    minority_count = np.min(counts)
    logger.info(f"Minority class: {minority_class}, count: {minority_count}")

    balanced_indices = []
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        if len(cls_indices) > minority_count:
            cls_indices = resample(cls_indices, n_samples=minority_count, random_state=42)
        balanced_indices.extend(cls_indices)

    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]

    # Verify balance
    _, balanced_counts = np.unique(y_balanced, return_counts=True)
    is_balanced = len(set(balanced_counts)) == 1
    logger.info(f"Balanced class distribution: {dict(zip(classes, balanced_counts))}")
    logger.info(f"Dataset is balanced: {is_balanced}")

    if not is_balanced:
        logger.warning("The dataset is not perfectly balanced after resampling!")

    logger.info(f"Original dataset shape: {X.shape}")
    logger.info(f"Balanced dataset shape: {X_balanced.shape}")

    return X_balanced, y_balanced

def run_svm_decoding_taskvsrest(X: np.ndarray, y: np.ndarray, sub: np.ndarray) -> Dict:
    """
    Perform SVM decoding analysis at the subject level using stratified k-fold cross-validation.

    This function processes each subject's data separately, performs SVM classification
    with cross-validation, and then aggregates results across all subjects.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Label array of shape (n_samples,).
        sub (np.ndarray): Subject identifier array of shape (n_samples,).

    Returns:
        Dict: A dictionary containing evaluation metrics and predictions at subject,
              fold, and overall levels.

    The returned dictionary structure:
    {
        'subject_id': {
            'accuracy': float,
            'balanced_accuracy': float,
            'f1_score': float,
            'y_true': np.ndarray,
            'y_pred': np.ndarray,
            'y_pred_proba': np.ndarray,
            'fold_accuracies': List[float],
            't_pvalue': float,
            'binom_test_pvalue': float
        },
        ...,
        'overall': {
            'accuracy': float,
            'balanced_accuracy': float,
            'f1_score': float,
            'subject_accuracies': List[float],
            't_pvalue': float,
            'binom_test_pvalue': float
        }
    }
    """
    all_results = {}
    subject_accuracies = []

    # Iterate over each unique subject
    for subject in np.unique(sub):
        logger.info(f"Processing subject: {subject}")

        # Filter data for the current subject
        subject_mask = sub == subject
        X_subject = X[subject_mask]
        y_subject = y[subject_mask]

        # Initialize stratified k-fold cross-validation
        skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
        subject_y_true, subject_y_pred, subject_y_pred_proba = [], [], []
        subject_fold_accuracies = []

        # Perform cross-validation for the current subject
        for fold, (train_index, test_index) in enumerate(skf.split(X_subject, y_subject)):
            # Split the data into training and test sets
            X_train, X_test = X_subject[train_index].reshape(-1, 1), X_subject[test_index].reshape(-1, 1)
            y_train, y_test = y_subject[train_index], y_subject[test_index]

            # Balance the training set
            X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)

            # Create and train the SVM model
            svm = make_pipeline(StandardScaler(), SVC(kernel="linear", probability=True))
            svm.fit(X_train_balanced, y_train_balanced)

            # Make predictions on the test set (not balanced)
            fold_pred = svm.predict(X_test)
            fold_pred_proba = svm.predict_proba(X_test)[:, 1]
            fold_accuracy = accuracy_score(y_test, fold_pred)
            subject_fold_accuracies.append(fold_accuracy)

            logger.info(f"Subject {subject}, Fold {fold + 1} Accuracy: {fold_accuracy:.3f}")

            # Collect results
            subject_y_true.extend(y_test)
            subject_y_pred.extend(fold_pred)
            subject_y_pred_proba.extend(fold_pred_proba)

        # Calculate subject-level metrics
        subject_accuracy = np.mean(subject_fold_accuracies)
        subject_accuracies.append(subject_accuracy)
        subject_y_true = np.array(subject_y_true)
        subject_y_pred = np.array(subject_y_pred)
        subject_balanced_accuracy = balanced_accuracy_score(subject_y_true, subject_y_pred)
        subject_f1 = f1_score(subject_y_true, subject_y_pred, average="weighted")

        # Perform statistical tests for the subject
        subject_t_stat, subject_t_pvalue = ttest_1samp(subject_fold_accuracies, 0.5)
        subject_n_samples = len(subject_y_true)
        subject_n_correct = int(subject_accuracy * subject_n_samples)
        subject_binom_result = binomtest(subject_n_correct, subject_n_samples, p=0.5)

        # Store subject-level results
        all_results[subject] = {
            "accuracy": subject_accuracy,
            "balanced_accuracy": subject_balanced_accuracy,
            "f1_score": subject_f1,
            "y_true": subject_y_true,
            "y_pred": subject_y_pred,
            "y_pred_proba": np.array(subject_y_pred_proba),
            "fold_accuracies": subject_fold_accuracies,
            "t_pvalue": subject_t_pvalue,
            "binom_test_pvalue": subject_binom_result.pvalue,
        }

        logger.info(f"Subject {subject} - Overall Accuracy: {subject_accuracy:.3f}")

    # Calculate overall metrics based on subject averages
    overall_accuracy = np.mean(subject_accuracies)
    overall_t_stat, overall_t_pvalue = ttest_1samp(subject_accuracies, 0.5)

    # Aggregate all predictions and true labels
    all_y_true = np.concatenate([results["y_true"] for results in all_results.values()])
    all_y_pred = np.concatenate([results["y_pred"] for results in all_results.values()])

    overall_balanced_accuracy = balanced_accuracy_score(all_y_true, all_y_pred)
    overall_f1 = f1_score(all_y_true, all_y_pred, average="weighted")

    # Overall binomial test based on subject averages
    n_subjects = len(subject_accuracies)
    n_subjects_above_chance = sum(acc > 0.5 for acc in subject_accuracies)
    overall_binom_result = binomtest(n_subjects_above_chance, n_subjects, p=0.5)

    # Log overall results
    logger.info(f"Overall Accuracy across all subjects: {overall_accuracy:.3f}")
    logger.info(f"Overall Balanced Accuracy: {overall_balanced_accuracy:.3f}")
    logger.info(f"Overall F1 Score: {overall_f1:.3f}")
    logger.info(f"Overall T-test p-value (based on subject averages): {overall_t_pvalue:.4f}")
    logger.info(f"Overall Binomial test p-value (based on subject averages): {overall_binom_result.pvalue:.4f}")

    # Add overall results to the dictionary
    all_results["overall"] = {
        "accuracy": overall_accuracy,
        "balanced_accuracy": overall_balanced_accuracy,
        "f1_score": overall_f1,
        "subject_accuracies": subject_accuracies,
        "t_pvalue": overall_t_pvalue,
        "binom_test_pvalue": overall_binom_result.pvalue,
    }

    return all_results

# def plot_results(results: Dict):
#     """
#     Create and display publication-ready visualizations of the SVM decoding results.

#     Args:
#         results (Dict): Dictionary containing the evaluation metrics and predictions.
#     """
#     # Set a consistent style for all plots
#     colors = sns.color_palette("deep")

#     # Font settings
#     plt.rcParams["font.family"] = "sans-serif"
#     plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"]
#     plt.rcParams["font.size"] = 12
#     plt.rcParams["axes.labelsize"] = 14
#     plt.rcParams["axes.titlesize"] = 16
#     plt.rcParams["xtick.labelsize"] = 12
#     plt.rcParams["ytick.labelsize"] = 12
#     plt.rcParams["legend.fontsize"] = 12
#     plt.rcParams["figure.titlesize"] = 18

#     # Plot 1: Accuracy Distribution
#     plt.figure(figsize=(10, 6))
#     sns.histplot(results["fold_accuracies"], kde=True, color=colors[0])
#     plt.axvline(
#         results["accuracy"],
#         color=colors[1],
#         linestyle="--",
#         label="Mean Accuracy",
#         linewidth=2,
#     )
#     plt.axvline(0.5, color=colors[2], linestyle=":", label="Chance Level", linewidth=2)
#     plt.xlabel("Accuracy")
#     plt.ylabel("Frequency")
#     plt.title("Distribution of Fold Accuracies", fontweight="bold")
#     plt.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper left")
#     plt.tight_layout()
#     plt.savefig(
#         os.path.join(OUT_DIR, "accuracy_distribution.png"), dpi=300, bbox_inches="tight"
#     )
#     plt.show()

#     # Plot 2: ROC Curve
#     fpr, tpr, _ = roc_curve(results["y_true"], results["y_pred_proba"])
#     roc_auc = auc(fpr, tpr)

#     plt.figure(figsize=(10, 6))
#     plt.plot(fpr, tpr, color=colors[0], lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
#     plt.plot(
#         [0, 1], [0, 1], color=colors[3], lw=2, linestyle="--", label="Random Classifier"
#     )
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("Receiver Operating Characteristic (ROC) Curve", fontweight="bold")
#     plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=300, bbox_inches="tight")
#     plt.show()

#     # Plot 3: Average Accuracy with Distribution and Confidence Interval
#     plt.figure(figsize=(8, 10))

#     # Calculate the confidence interval
#     conf_int = stats.t.interval(
#         confidence=0.95,
#         df=len(results["fold_accuracies"]) - 1,
#         loc=np.mean(results["fold_accuracies"]),
#         scale=stats.sem(results["fold_accuracies"]),
#     )

#     # Plot the underlying distribution
#     sns.stripplot(
#         y=results["fold_accuracies"],
#         orient="v",
#         color=colors[0],
#         alpha=0.4,
#         jitter=True,
#         label="Fold Accuracies",
#         size=8,
#     )

#     # Plot the average accuracy
#     plt.axhline(
#         results["accuracy"],
#         color=colors[1],
#         linestyle="-",
#         label="Mean Accuracy",
#         linewidth=2,
#     )

#     # Plot the confidence interval
#     plt.axhspan(conf_int[0], conf_int[1], alpha=0.2, color=colors[1])

#     # Plot chance level
#     plt.axhline(0.5, color=colors[2], linestyle=":", label="Chance Level", linewidth=2)

#     plt.ylabel("Accuracy")
#     plt.title(
#         "Model Performance:\nAccuracy Distribution and Confidence Interval",
#         fontweight="bold",
#     )
#     plt.legend(frameon=True, facecolor="white", edgecolor="none", loc="lower right")

#     # Remove x-axis labels and ticks
#     plt.xlabel("")
#     plt.xticks([])

#     # Adjust y-axis to start from 0 and end at 1
#     plt.ylim(0, 1)

#     # # Add text annotations
#     # plt.text(0.02, 0.02, f"Mean Accuracy: {results['accuracy']:.3f}\n95% CI: [{conf_int[0]:.3f}, {conf_int[1]:.3f}]",
#     #          transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom')

#     plt.tight_layout()
#     plt.savefig(
#         os.path.join(OUT_DIR, "accuracy_distribution_ci.png"),
#         dpi=300,
#         bbox_inches="tight",
#     )
#     plt.show()

#     # Plot 4: Normalized Confusion Matrix
#     cm = confusion_matrix(results["y_true"], results["y_pred"], normalize="true")
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", cbar=True, vmin=0, vmax=1)

#     plt.title("Normalized Confusion Matrix", fontweight="bold")
#     plt.xlabel("Predicted Label")
#     plt.ylabel("True Label")

#     # Set tick labels
#     tick_labels = ["Non-Expert", "Expert"]
#     plt.xticks(ticks=[0.5, 1.5], labels=tick_labels)
#     plt.yticks(ticks=[0.5, 1.5], labels=tick_labels, rotation=0)

#     plt.tight_layout()
#     plt.savefig(
#         os.path.join(OUT_DIR, "normalized_confusion_matrix.png"),
#         dpi=300,
#         bbox_inches="tight",
#     )
#     plt.show()

def plot_results_taskvsrest(results: Dict, participants, OUT_DIR: str):
    """
    Create and display publication-ready visualizations of the SVM decoding results,
    using subject-level accuracies as datapoints.

    Args:
        results (Dict): Dictionary containing the evaluation metrics and predictions.
        OUT_DIR (str): Directory to save the output plots.
    """
    # Set a consistent style for all plots
    colors = sns.color_palette("deep")

    # Font settings
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18
    })

    # Extract subject accuracies
    subject_accuracies = results['overall']['subject_accuracies']

    # Plot 1: Subject Accuracy Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(subject_accuracies, kde=True, color=colors[0])
    plt.axvline(
        results['overall']['accuracy'],
        color=colors[1],
        linestyle="--",
        label="Mean Accuracy",
        linewidth=2,
    )
    plt.axvline(0.5, color=colors[2], linestyle=":", label="Chance Level", linewidth=2)
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.title("Distribution of Subject Accuracies", fontweight="bold")
    plt.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper left")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "subject_accuracy_distribution.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()

    # Plot 2: ROC Curve (using aggregated data)
    all_y_true = np.concatenate([results[subject]['y_true'] for subject in results if subject != 'overall'])
    all_y_pred_proba = np.concatenate([results[subject]['y_pred_proba'] for subject in results if subject != 'overall'])
    fpr, tpr, _ = roc_curve(all_y_true, all_y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color=colors[0], lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot(
        [0, 1], [0, 1], color=colors[3], lw=2, linestyle="--", label="Random Classifier"
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontweight="bold")
    plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # Plot 3: Average Accuracy with Distribution and Confidence Interval
    plt.figure(figsize=(8, 10))

    # Calculate the confidence interval
    subject_accuracies = results['overall']['subject_accuracies']

    # Create a dictionary mapping subject IDs to their expert status
    subject_status = {subject: is_expert for subject, is_expert in participants}

    # Create lists for expert and non-expert accuracies
    expert_accuracies = []
    non_expert_accuracies = []

    for subject, accuracy in zip(results.keys(), subject_accuracies):
        if subject != 'overall':
            if subject_status[subject]:
                expert_accuracies.append(accuracy)
            else:
                non_expert_accuracies.append(accuracy)

    # Plot the underlying distribution
    plt.scatter(
        np.random.normal(0.1, 0.02, len(expert_accuracies)),
        expert_accuracies,
        color='green',
        alpha=0.6,
        label="Expert",
        s=80,
    )
    plt.scatter(
        np.random.normal(-0.1, 0.02, len(non_expert_accuracies)),
        non_expert_accuracies,
        color='red',
        alpha=0.6,
        label="Non-Expert",
        s=80,
    )

    # Plot the average accuracy
    plt.axhline(
        np.mean(expert_accuracies),
        color='green',
        linestyle="-",
        linewidth=2,
    )
    # Plot the confidence interval
    conf_int = stats.t.interval(
        confidence=0.95,
        df=len(expert_accuracies) - 1,
        loc=np.mean(expert_accuracies),
        scale=stats.sem(expert_accuracies),
    )
    plt.axhspan(conf_int[0], conf_int[1], alpha=0.4, color='green')


    # Plot the average accuracy
    plt.axhline(
        np.mean(non_expert_accuracies),
        color='red',
        linestyle="-",
        linewidth=2,
    )
    # Plot the confidence interval
    conf_int = stats.t.interval(
        confidence=0.95,
        df=len(non_expert_accuracies) - 1,
        loc=np.mean(non_expert_accuracies),
        scale=stats.sem(non_expert_accuracies),
    )
    plt.axhspan(conf_int[0], conf_int[1], alpha=0.4, color='red')

    # Plot chance level
    plt.axhline(0.5, color=colors[2], linestyle=":", label="Chance Level", linewidth=2)

    plt.ylabel("Accuracy")
    plt.title(
        "Model Performance:\nSubject Accuracy Distribution and Confidence Interval",
        fontweight="bold",
    )
    plt.legend(frameon=True, facecolor="white", edgecolor="none", loc="lower right")

    # Remove x-axis labels and ticks
    plt.xlabel("")
    plt.xticks([])

    # Adjust y-axis to start from 0 and end at 1
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "subject_accuracy_distribution_ci_colored.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Plot 4: Normalized Confusion Matrix (using aggregated data)
    all_y_true = np.concatenate([results[subject]['y_true'] for subject in results if subject != 'overall'])
    all_y_pred = np.concatenate([results[subject]['y_pred'] for subject in results if subject != 'overall'])
    cm = confusion_matrix(all_y_true, all_y_pred, normalize="true")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", cbar=True, vmin=0, vmax=1)

    plt.title("Normalized Confusion Matrix", fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Set tick labels
    tick_labels = ["Fixation", "Task"]
    plt.xticks(ticks=[0.5, 1.5], labels=tick_labels)
    plt.yticks(ticks=[0.5, 1.5], labels=tick_labels, rotation=0)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "normalized_confusion_matrix.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info("All plots have been saved in the directory: %s", OUT_DIR)

def save_script_to_file(output_directory: str):
    """
    Saves the script file that is calling this function to the specified output directory.

    Args:
        output_directory (str): The directory where the script file will be saved.

    This function automatically detects the script file that is executing this function
    and creates a copy of it in the output directory.
    It logs the process, indicating whether the saving was successful or if any error occurred.
    """
    try:
        # Get the frame of the caller to this function
        caller_frame = inspect.stack()[1]
        # Get the file name of the script that called this function
        script_file = caller_frame.filename

        # Construct the output file path
        script_file_out = os.path.join(output_directory, os.path.basename(script_file))

        # Log the attempt to save the script file
        logging.debug(f"Attempting to save the script file to: {script_file_out}")

        # Copy the script file to the output directory
        shutil.copy(script_file, script_file_out)

        # Log the successful save
        logging.info("Script file saved successfully.")
    except Exception as e:
        # Log any errors encountered during the saving process
        logging.error(
            f"An error occurred while saving the script file: {e}", exc_info=True
        )


def set_random_seeds(seed: int = 42):
    """
    Set the random seed for reproducibility in NumPy and Python's random module.

    Args:
        seed (int): The random seed. Default is 42.
    """
    # Set the seed for generating random numbers
    np.random.seed(seed)
    random.seed(seed)


def create_run_id() -> str:
    """
    Create a unique run ID based on the current timestamp.

    Returns:
        str: A string representing the current date and time in the format "YYYYMMDD-HHMMSS".
    """
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")


# def svm_decoding(eye_tracking_data: pd.DataFrame) -> Dict:
#     """
#     Main function to run the SVM decoding analysis.

#     Args:
#         eye_tracking_data (pd.DataFrame): The raw eye-tracking data.

#     Returns:
#         Dict: A dictionary containing the evaluation metrics and predictions.
#     """
#     logger.info("Starting main analysis function")

#     # Prepare features from the eye-tracking data
#     X, y, groups = prepare_features(eye_tracking_data)

#     # Run SVM decoding
#     results = run_svm_decoding_taskvsrest(X, y, groups)

#     # Plot the results
#     plot_results(results)

#     logger.info("Completed main analysis function")
#     return results

def svm_decoding_taskvsrest(eye_tracking_data: pd.DataFrame, participants, OUT_DIR) -> Dict:
    """
    Main function to run the SVM decoding analysis.

    Args:
        eye_tracking_data (pd.DataFrame): The raw eye-tracking data.

    Returns:
        Dict: A dictionary containing the evaluation metrics and predictions.
    """
    logger.info("Starting main analysis function")

    # Prepare features from the eye-tracking data
    X, y, sub = prepare_features_taskvsrest(eye_tracking_data)

    # Run SVM decoding
    results = run_svm_decoding_taskvsrest(X, y, sub)

    # Plot the results
    plot_results_taskvsrest(results, participants, OUT_DIR)

    logger.info("Completed main analysis function")
    return results


def add_fixation_trials(df):
    """
    Add fixation trials to fill gaps between existing trials in the event dataframe.

    Parameters:
    df (pandas.DataFrame): Input dataframe with columns 'onset', 'duration', 'trial_type', and 'sub'.

    Returns:
    pandas.DataFrame: A new dataframe with fixation trials added.
    """
    # Sort the dataframe by onset time to ensure correct order
    df = df.sort_values("onset")

    # Initialize an empty list to store all trials (including fixations)
    all_trials = []

    # Initialize the end time of the previous trial
    prev_end = 0

    for _, row in df.iterrows():
        # Check if there's a gap between the previous trial and the current one
        if row["onset"] > prev_end:
            # Add a fixation trial
            fixation_trial = {
                "onset": prev_end,
                "duration": row["onset"] - prev_end,
                "trial_type": "fixation",
            }
            all_trials.append(fixation_trial)

        # Add the current trial
        all_trials.append(row.to_dict())

        # Update the end time of the previous trial
        prev_end = row["onset"] + row["duration"]

    # Add last fix block (10 seconds)
    last_fix_trial = {
        "onset": prev_end,
        "duration": 10,
        "trial_type": "fixation",
    }
    all_trials.append(last_fix_trial)

    # Create a new dataframe with all trials
    new_df = pd.DataFrame(all_trials)

    # Sort the new dataframe by onset time
    new_df = new_df.sort_values("onset").reset_index(drop=True)

    return new_df


def load_events_data(bids_dir="/data/projects/chess/data/BIDS"):
    event_paths = glob.glob(os.path.join(bids_dir, "sub-*", "func", "*_events.tsv"))
    all_events_data = []
    for event_path in event_paths:

        fname = os.path.basename(event_path)
        sub, task, run, tag = fname.split("_")

        event_file = pd.read_csv(event_path, sep="\t")

        # Here, we explicitly add the fixation trials:
        # 10 s at the beginning, 10 s at the end, and 0.5 between each trial
        event_file_long = add_fixation_trials(event_file)

        event_file_long["sub"] = sub.split("-")[-1]
        event_file_long["run"] = run.split("-")[-1]

        all_events_data.append(event_file_long)

    all_events_df = pd.concat(all_events_data)

    # Sort all_events_df
    all_events_df = all_events_df.sort_values(["sub", "run", "onset"]).reset_index(
        drop=True
    )

    return all_events_df


def add_trial_type_to_eye_tracking(eye_tracking_df, events_df, mapping_method="half"):
    """
    Add a 'trial_type' column to the eye-tracking dataframe based on the events dataframe,
    processing each subject and run independently.

    Parameters:
    eye_tracking_df (pandas.DataFrame): Eye-tracking data with 'subject', 'run', and 'timestamp' columns.
    events_df (pandas.DataFrame): Events data with 'sub', 'run', 'onset', 'duration', and 'trial_type' columns.
    mapping_method (str): Method to use for mapping timestamps to trial types.
                          'start' uses the stimulus at the start of each TR,
                          'half' (default) uses the midpoint between consecutive TRs.

    Returns:
    pandas.DataFrame: A new dataframe with the 'trial_type' column added to the eye-tracking data.

    Raises:
    ValueError: If an invalid mapping_method is provided.
    """
    if mapping_method not in ["start", "half"]:
        raise ValueError("mapping_method must be either 'start' or 'half'")

    def find_trial_type(timestamp, next_timestamp, events, method):
        if method == "start":
            reference_point = timestamp
        elif method == "half":
            reference_point = (
                timestamp + ((next_timestamp - timestamp) / 2)
                if pd.notna(next_timestamp)
                else timestamp
            )

        matching_events = events[events["onset"] <= reference_point]
        if matching_events.empty:
            return "unknown"
        last_matching_event = matching_events.iloc[-1]
        if reference_point < last_matching_event["end_time"]:
            return last_matching_event["trial_type"]
        else:
            return "unknown"

    result_dfs = []

    # Loop over subjects
    for subject in eye_tracking_df["subject"].unique():
        subject_eye_data = eye_tracking_df[eye_tracking_df["subject"] == subject]
        subject_events = events_df[events_df["sub"] == subject.split("-")[-1]]

        # Loop over runs for each subject
        for run in subject_eye_data["run"].unique():
            run_eye_data = subject_eye_data[subject_eye_data["run"] == run]
            run_events = subject_events[subject_events["run"] == run]

            # Sort the data by timestamp
            run_eye_data = run_eye_data.sort_values("timestamp").reset_index(drop=True)
            run_events = run_events.sort_values("onset")
            run_events["end_time"] = run_events["onset"] + run_events["duration"]

            # Create a series of next timestamps
            next_timestamps = run_eye_data["timestamp"].shift(-1)

            # Apply the function to create the new 'trial_type' column
            run_eye_data["trial_type"] = [
                find_trial_type(timestamp, next_timestamp, run_events, mapping_method)
                for timestamp, next_timestamp in zip(
                    run_eye_data["timestamp"], next_timestamps
                )
            ]

            result_dfs.append(run_eye_data)

    # Combine all processed data
    result_df = pd.concat(result_dfs, ignore_index=True)

    # Sort all_events_df
    result_df = result_df.sort_values(["subject", "run", "timestamp"]).reset_index(
        drop=True
    )

    return result_df


# Set up the data paths
root_dir = "/data/projects/chess/data/BIDS/derivatives/bidsmreye"
OUT_DIR = f"/data/projects/chess/temp_bidsmreye/{create_run_id()}_et-mvpa-taskvsrest"
participants_xlsx_path = "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/Projects/Expertise/chess_files/chess_project_files/participants.xlsx"
bids_dir = "/data/projects/chess/data/BIDS"

# Create output directory if it doesn't exist
os.makedirs(OUT_DIR, exist_ok=True)

# Setup logging
logger = setup_logging(OUT_DIR)

logger.info("Starting script execution")
logger.info(f"Results can be found in {OUT_DIR}")

# Save script to file
save_script_to_file(OUT_DIR)

# Set random seeds for reproducibility, ensuring consistent results across runs
logging.info("Setting random seeds to 42 for reproducibility.")
set_random_seeds(seed=42)

# Load and process participants data
participants_df = pd.read_excel(participants_xlsx_path)
filtered_df = participants_df.dropna(subset=["Expert"])
participants = [
    (f"sub-{sub_id:02d}", bool(expert))
    for sub_id, expert in zip(filtered_df["sub_id"].astype(int), filtered_df["Expert"])
]

# Load eye-tracking data
eye_tracking_data = load_eye_tracking_data(root_dir, participants)

# Here, once we load the et data, we also need to:
# 1. Load the event file for that subj and run
# 2. Map the eye data to Task vs. Rest
# 3. Then, we can perform the analysis below (changes may be needed)

# Load event files
events_data = load_events_data(bids_dir)

# Map eye data to stimulus type (Task vs Fix)
mapped_eye_tracking_df = add_trial_type_to_eye_tracking(
    eye_tracking_data, events_data, mapping_method="half"
)

# TODO: edit this function below to do an SVM between Task vs. Fixation
# Run the main analysis
results = svm_decoding_taskvsrest(mapped_eye_tracking_df, participants, OUT_DIR)

logger.info("SVM Decoding Results:")
logger.info(f"Accuracy: {results['overall']['accuracy']:.3f}")
logger.info(f"Balanced Accuracy: {results['overall']['balanced_accuracy']:.3f}")
logger.info(f"F1 Score: {results['overall']['f1_score']:.3f}")
logger.info(f"T-test p-value: {results['overall']['t_pvalue']:.4f}")
logger.info(f"Binomial test p-value: {results['overall']['binom_test_pvalue']:.4f}")

# # Save results to a JSON file
# with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
#     json.dump(
#         {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in results.items()}, f
#     )

logger.info("Analysis completed. Results saved to JSON file.")
