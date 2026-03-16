"""Compute Equal Error Rate (EER) from score files."""

import argparse
from typing import List, Tuple

from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve


def read_score_file(path: str) -> Tuple[List[float], List[float]]:
    """Read score file with format: <score> <label> <genre> per line."""
    y_pred, y_true = [], []
    with open(path, 'r') as f:
        for line in f:
            score, label, _genre = line.strip().split(' ')
            y_pred.append(float(score))
            y_true.append(float(label))
    return y_pred, y_true


def compute_eer(y_true: List[float], y_pred: List[float]) -> Tuple[float, float]:
    """Compute Equal Error Rate and corresponding threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    threshold = float(interp1d(fpr, thresholds)(eer))
    return eer, threshold


def main():
    parser = argparse.ArgumentParser(description='Compute EER from score file')
    parser.add_argument('--score-filepath', dest='score_filepath',
                        required=True, type=str)
    args = parser.parse_args()

    y_pred, y_true = read_score_file(args.score_filepath)
    eer, threshold = compute_eer(y_true, y_pred)
    print(f"EER: {100 * eer:.3f}%")
    print(f"Threshold: {threshold:.4f}")


if __name__ == '__main__':
    main()
