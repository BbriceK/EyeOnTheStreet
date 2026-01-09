import numpy as np

def binarize_predictions(y_pred, threshold):
    """
    Convert predicted probabilities into binary predictions based on a threshold.

    Args:
        y_pred (np.ndarray): Predicted probabilities (num_samples x num_classes).
        threshold (float): Probability threshold for assigning 1 vs 0.

    Returns:
        np.ndarray: Binary predictions (0 or 1) of the same shape as y_pred.
    """
    # if >= threshold → 1, else → 0
    return (y_pred >= threshold).astype(int)

def multilabel_metrics(y_true, y_pred_binary, category_names):
    """
    Calculate per-class and macro precision, recall, and F1 score for multi-label classification.

    Args:
        y_true (np.ndarray): Ground truth labels (num_samples x num_classes).
        y_pred_binary (np.ndarray): Binary predictions (num_samples x num_classes).
        category_names (list): List of class/category names.

    Returns:
        dict: Dictionary containing:
            - 'per_class': metrics for each category (precision, recall, F1)
            - 'macro': average metrics across all categories
    """
    per_class = {}

    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for i, name in enumerate(category_names):
        tp = int(np.sum((y_true[:, i] == 1) & (y_pred_binary[:, i] == 1)))
        fp = int(np.sum((y_true[:, i] == 0) & (y_pred_binary[:, i] == 1)))
        fn = int(np.sum((y_true[:, i] == 1) & (y_pred_binary[:, i] == 0)))
        tn = int(np.sum((y_true[:, i] == 0) & (y_pred_binary[:, i] == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    n = len(category_names)
    macro = {
        "precision": macro_precision / n,
        "recall": macro_recall / n,
        "f1": macro_f1 / n,
    }

    return {
        "per_class": per_class,
        "macro": macro,
    }
