#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 23:03:32 2024

@author: costantino_ai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eye-tracking Data Analysis for Chess Expertise Classification

This script performs SVM decoding analysis on eye-tracking data to classify chess players as experts or non-experts.
It uses stratified k-fold cross-validation with balanced training sets and implements various statistical tests
and visualizations to evaluate the model's performance.

Created on Tue Aug 13 17:51:34 2024
@author: costantino_ai
"""

# Standard library imports
import json
import os
import random
import shutil
import inspect
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime

# Third-party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import binomtest, ttest_1samp
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Logging setup
import logging
from logging_utils import setup_logging
from common_utils import create_run_id, save_script_to_file, set_rnd_seed

def load_eye_tracking_data(root_dir: str, participants: List[Tuple[str, bool]]) -> pd.DataFrame:
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
    subjects = [d for d in root_path.iterdir() if d.is_dir() and d.name.startswith('sub-')]

    expert_dict = dict(participants)
    all_data = []

    for subject in subjects:
        logger.debug(f"Processing subject: {subject.name}")
        func_dir = subject / 'func'

        for eyetrack_file in func_dir.glob('*_desc-1to6_eyetrack.tsv'):
            # Read the TSV file
            df = pd.read_csv(eyetrack_file, sep='\t')

            # Extract subject ID and run number from the filename
            parts = eyetrack_file.stem.split('_')
            subject_id = parts[0]
            run = next(p for p in parts if p.startswith('run-')).split('-')[1]

            # Add subject, run, and expert status to the dataframe
            df['subject'] = subject_id
            df['run'] = run
            df['expert'] = expert_dict.get(subject_id, np.nan)

            # Load and add metadata from the corresponding JSON file
            json_file = eyetrack_file.with_suffix('.json')
            with json_file.open() as f:
                metadata = json.load(f)

            for key, value in metadata.items():
                if isinstance(value, (str, int, float)):
                    df[key] = value

            all_data.append(df)

    # Combine all data into a single dataframe
    combined_data = pd.concat(all_data, ignore_index=True)
    logger.info(f"Loaded eye-tracking data: {len(combined_data)} rows, {combined_data['subject'].nunique()} subjects")

    return combined_data

def prepare_features(eye_tracking_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare features for SVM classification from eye-tracking data.

    Args:
        eye_tracking_data (pd.DataFrame): The raw eye-tracking data.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Features, labels, and group identifiers.
    """
    # Group the data by subject
    grouped_by_subject = eye_tracking_data.groupby('subject')

    all_subject_data = []
    all_labels = []
    all_groups = []

    for subject, subject_data in grouped_by_subject:
        # Group each subject's data by run
        subject_runs = subject_data.groupby('run')

        subject_run_data = []
        for _, run_data in subject_runs:
            # Extract x and y coordinates, limiting to the minimum number of timepoints
            X = np.nan_to_num(run_data[['displacement']].values, nan=0.0)
            subject_run_data.append(X.flatten())

        # Add the data for all runs of this subject
        all_subject_data.extend(subject_run_data)
        # Add labels (expert status) for each run
        all_labels.extend([subject_data['expert'].iloc[0]] * len(subject_run_data))
        # Add group identifiers (subject ID) for each run
        all_groups.extend([subject] * len(subject_run_data))

    return np.array(all_subject_data), np.array(all_labels), np.array(all_groups)

def run_svm_decoding(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict:
    """
    Perform SVM decoding analysis using stratified group k-fold cross-validation.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        groups (np.ndarray): Group identifiers for cross-validation.

    Returns:
        Dict: A dictionary containing evaluation metrics and predictions.
    """
    # Initialize stratified group k-fold cross-validation
    skf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)

    y_true, y_pred, y_pred_proba = [], [], []
    fold_accuracies = []

    for fold, (train_index, test_index) in enumerate(skf.split(X, y, groups)):
        # Split the data into training and test sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        groups_test = groups[test_index]

        logger.info(f"Fold {fold + 1} - Held-out subjects: {np.unique(groups_test)}")

        # Create and train the SVM model
        svm = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
        svm.fit(X_train, y_train)

        # Make predictions on the test set
        fold_pred = svm.predict(X_test)
        fold_pred_proba = svm.predict_proba(X_test)[:, 1]
        fold_accuracy = accuracy_score(y_test, fold_pred)
        fold_accuracies.append(fold_accuracy)

        logger.info(f"Fold {fold + 1} Accuracy: {fold_accuracy:.3f}")
        for i, (true, pred, prob) in enumerate(zip(y_test, fold_pred, fold_pred_proba)):
            logger.info(f"Fold {fold + 1}, Sample {i + 1}: Target={true}, Pred={pred}, Prob={prob:.3f}, Subject={groups_test[i]}")

        # Collect results
        y_true.extend(y_test)
        y_pred.extend(fold_pred)
        y_pred_proba.extend(fold_pred_proba)

    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)

    # Calculate evaluation metrics
    overall_accuracy = np.mean(fold_accuracies)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Perform statistical tests
    t_stat, t_pvalue = ttest_1samp(fold_accuracies, 0.5)
    n_samples = len(y_true)
    n_correct = int(overall_accuracy * n_samples)
    binom_result = binomtest(n_correct, n_samples, p=0.5)

    # Log results
    logger.info(f"Overall Accuracy: {overall_accuracy:.3f}")
    logger.info(f"Balanced Accuracy: {balanced_accuracy:.3f}")
    logger.info(f"F1 Score: {f1:.3f}")
    logger.info(f"T-test p-value: {t_pvalue:.4f}")
    logger.info(f"Binomial test p-value: {binom_result.pvalue:.4f}")

    # Return results as a dictionary
    return {
        'accuracy': overall_accuracy,
        'balanced_accuracy': balanced_accuracy,
        'f1_score': f1,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'fold_accuracies': fold_accuracies,
        't_pvalue': t_pvalue,
        'binom_test_pvalue': binom_result.pvalue
    }

def plot_results(results: Dict):
    """
    Create and display publication-ready visualizations of the SVM decoding results.

    Args:
        results (Dict): Dictionary containing the evaluation metrics and predictions.
    """
    # Set a consistent style for all plots
    colors = sns.color_palette("deep")

    # Font settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18

    # Plot 1: Accuracy Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(results['fold_accuracies'], kde=True, color=colors[0])
    plt.axvline(results['accuracy'], color=colors[1], linestyle='--', label='Mean Accuracy', linewidth=2)
    plt.axvline(0.5, color=colors[2], linestyle=':', label='Chance Level', linewidth=2)
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.title('Distribution of Fold Accuracies', fontweight='bold')
    plt.legend(frameon=True, facecolor='white', edgecolor='none', loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'accuracy_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 2: ROC Curve
    fpr, tpr, _ = roc_curve(results['y_true'], results['y_pred_proba'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color=colors[0], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color=colors[3], lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold')
    plt.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 3: Average Accuracy with Distribution and Confidence Interval
    plt.figure(figsize=(8, 10))

    # Calculate the confidence interval
    conf_int = stats.t.interval(confidence=0.95, df=len(results['fold_accuracies'])-1,
                                loc=np.mean(results['fold_accuracies']),
                                scale=stats.sem(results['fold_accuracies']))

    # Plot the underlying distribution
    sns.stripplot(y=results['fold_accuracies'], orient='v', color=colors[0], alpha=0.4, jitter=True, label='Fold Accuracies', size=8)

    # Plot the average accuracy
    plt.axhline(results['accuracy'], color=colors[1], linestyle='-', label='Mean Accuracy', linewidth=2)

    # Plot the confidence interval
    plt.axhspan(conf_int[0], conf_int[1], alpha=0.2, color=colors[1])

    # Plot chance level
    plt.axhline(0.5, color=colors[2], linestyle=':', label='Chance Level', linewidth=2)

    plt.ylabel('Accuracy')
    plt.title('Accuracy per Fold', fontweight='bold')
    plt.legend(frameon=True, facecolor='white', edgecolor='none', loc='lower right')

    # Remove x-axis labels and ticks
    plt.xlabel('')
    plt.xticks([])

    # Adjust y-axis to start from 0 and end at 1
    plt.ylim(0, 1)

    # # Add text annotations
    # plt.text(0.02, 0.02, f"Mean Accuracy: {results['accuracy']:.3f}\n95% CI: [{conf_int[0]:.3f}, {conf_int[1]:.3f}]",
    #          transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'accuracy_distribution_ci.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 4: Normalized Confusion Matrix
    cm = confusion_matrix(results['y_true'], results['y_pred'], normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', cbar=True, vmin=0, vmax=1)

    plt.title('Normalized Confusion Matrix', fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Set tick labels
    tick_labels = ['Non-Expert', 'Expert']
    plt.xticks(ticks=[0.5, 1.5], labels=tick_labels)
    plt.yticks(ticks=[0.5, 1.5], labels=tick_labels, rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'normalized_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

 

def main(eye_tracking_data: pd.DataFrame) -> Dict:
    """
    Main function to run the SVM decoding analysis.

    Args:
        eye_tracking_data (pd.DataFrame): The raw eye-tracking data.

    Returns:
        Dict: A dictionary containing the evaluation metrics and predictions.
    """
    logger.info("Starting main analysis function")

    # Prepare features from the eye-tracking data
    X, y, groups = prepare_features(eye_tracking_data)

    # Run SVM decoding
    results = run_svm_decoding(X, y, groups)

    # Plot the results
    plot_results(results)

    logger.info("Completed main analysis function")
    return results

# Set up the data paths
root_dir = '/data/projects/chess/temp_bidsmreye/bidsmreye'
OUT_DIR = f"/data/projects/chess/temp_bidsmreye/{create_run_id()}_et-mvpa-displacement"
participants_xlsx_path = "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/Projects/Expertise/chess_files/chess_project_files/participants.xlsx"

# Create output directory if it doesn't exist
os.makedirs(OUT_DIR, exist_ok=True)

# Setup logging
logger = setup_logging(log_file=os.path.join(OUT_DIR, 'analysis_log.txt'))

logger.info("Starting script execution")
logger.info(f"Results can be found in {OUT_DIR}")

# Save script to file
save_script_to_file(OUT_DIR)

# Set random seeds for reproducibility, ensuring consistent results across runs
logging.info("Setting random seeds to 42 for reproducibility.")
set_rnd_seed(seed=42)

# Load and process participants data
participants_df = pd.read_excel(participants_xlsx_path)
filtered_df = participants_df.dropna(subset=["Expert"])
participants = [
    (f"sub-{sub_id:02d}", bool(expert))
    for sub_id, expert in zip(
        filtered_df["sub_id"].astype(int), filtered_df["Expert"]
    )
]

# Load eye-tracking data
eye_tracking_data = load_eye_tracking_data(root_dir, participants)

# Run the main analysis
results = main(eye_tracking_data)

logger.info("SVM Decoding Results:")
logger.info(f"Accuracy: {results['accuracy']:.3f}")
logger.info(f"Balanced Accuracy: {results['balanced_accuracy']:.3f}")
logger.info(f"F1 Score: {results['f1_score']:.3f}")
logger.info(f"T-test p-value: {results['t_pvalue']:.4f}")
logger.info(f"Binomial test p-value: {results['binom_test_pvalue']:.4f}")

# Save results to a JSON file
with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
    json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in results.items()}, f)

logger.info("Analysis completed. Results saved to JSON file.")
