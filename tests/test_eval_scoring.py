import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

from eval_scoring import compute_detection_scores


class EvalScoringTests(unittest.TestCase):
    def test_trapz_scores_without_requiring_labels_for_training(self) -> None:
        curves = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0, 0.0],
            ]
        )

        scores = compute_detection_scores(curves, [0, 1], mode="trapz", layer_start=1, layer_end=2)

        self.assertEqual(scores, [1.5, 1.5])

    def test_logreg_reports_only_out_of_fold_scores(self) -> None:
        curves = np.arange(24, dtype=float).reshape(6, 4)
        labels = np.array([0, 0, 0, 1, 1, 1])
        fit_rows = []
        scored_rows = []

        class RecordingLogisticRegression:
            def __init__(self, **kwargs):
                pass

            def fit(self, x_train, y_train):
                self.training_rows = {tuple(row) for row in x_train.tolist()}
                fit_rows.append(self.training_rows)
                return self

            def decision_function(self, x_test):
                test_rows = [tuple(row) for row in x_test.tolist()]
                scored_rows.extend(test_rows)
                overlap = self.training_rows.intersection(test_rows)
                if overlap:
                    raise AssertionError(f"scored training rows: {overlap}")
                return np.arange(len(test_rows), dtype=float)

        with patch("eval_scoring.LogisticRegression", RecordingLogisticRegression):
            scores = compute_detection_scores(
                curves,
                labels,
                mode="logreg",
                n_folds=3,
                seed=539,
            )

        self.assertEqual(len(scores), len(labels))
        self.assertEqual({tuple(row) for row in curves.tolist()}, set(scored_rows))
        self.assertTrue(all(len(rows) == 4 for rows in fit_rows))

    def test_logreg_c_can_be_configured(self) -> None:
        curves = np.arange(24, dtype=float).reshape(6, 4)
        labels = np.array([0, 0, 0, 1, 1, 1])
        configured_c_values = []

        class RecordingLogisticRegression:
            def __init__(self, **kwargs):
                configured_c_values.append(kwargs["C"])

            def fit(self, x_train, y_train):
                return self

            def decision_function(self, x_test):
                return np.arange(len(x_test), dtype=float)

        with patch("eval_scoring.LogisticRegression", RecordingLogisticRegression):
            compute_detection_scores(
                curves,
                labels,
                mode="logreg",
                n_folds=3,
                seed=539,
                logreg_c=1.25,
            )

        self.assertEqual(configured_c_values, [1.25, 1.25, 1.25])

    def test_logreg_rejects_non_positive_or_non_finite_c(self) -> None:
        curves = np.arange(16, dtype=float).reshape(4, 4)
        labels = np.array([0, 0, 1, 1])

        for logreg_c in (0, -1, np.nan, np.inf):
            with self.subTest(logreg_c=logreg_c):
                with self.assertRaisesRegex(ValueError, "logreg_c must be a positive finite value"):
                    compute_detection_scores(
                        curves,
                        labels,
                        mode="logreg",
                        n_folds=2,
                        seed=539,
                        logreg_c=logreg_c,
                    )

    def test_supervised_modes_use_all_layers_by_default(self) -> None:
        curves = np.arange(60, dtype=float).reshape(6, 10)
        labels = np.array([0, 0, 0, 1, 1, 1])
        training_widths = []

        class RecordingLogisticRegression:
            def __init__(self, **kwargs):
                pass

            def fit(self, x_train, y_train):
                training_widths.append(x_train.shape[1])
                return self

            def decision_function(self, x_test):
                return np.arange(len(x_test), dtype=float)

        with patch("eval_scoring.LogisticRegression", RecordingLogisticRegression):
            compute_detection_scores(
                curves,
                labels,
                mode="logreg",
                n_folds=3,
                seed=539,
                layer_start=2,
                layer_end=4,
            )

        self.assertEqual(training_widths, [10, 10, 10])

    def test_selected_supervised_layer_scope_restricts_logreg_layers(self) -> None:
        curves = np.arange(60, dtype=float).reshape(6, 10)
        labels = np.array([0, 0, 0, 1, 1, 1])
        training_widths = []
        scored_widths = []

        class RecordingLogisticRegression:
            def __init__(self, **kwargs):
                pass

            def fit(self, x_train, y_train):
                training_widths.append(x_train.shape[1])
                return self

            def decision_function(self, x_test):
                scored_widths.append(x_test.shape[1])
                return np.arange(len(x_test), dtype=float)

        with patch("eval_scoring.LogisticRegression", RecordingLogisticRegression):
            compute_detection_scores(
                curves,
                labels,
                mode="logreg",
                n_folds=3,
                seed=539,
                layer_start=2,
                layer_end=4,
                supervised_layer_scope="selected",
            )

        self.assertEqual(training_widths, [3, 3, 3])
        self.assertEqual(scored_widths, [3, 3, 3])

    def test_selected_supervised_layer_scope_restricts_fisher_layers(self) -> None:
        curves = np.arange(60, dtype=float).reshape(6, 10)
        labels = np.array([0, 0, 0, 1, 1, 1])
        training_widths = []
        scored_widths = []

        def recording_fisher_scores(x_train, y_train, x_test, fisher_epsilon):
            training_widths.append(x_train.shape[1])
            scored_widths.append(x_test.shape[1])
            return np.arange(len(x_test), dtype=float)

        with patch("eval_scoring._fisher_scores", recording_fisher_scores):
            compute_detection_scores(
                curves,
                labels,
                mode="fisher",
                n_folds=3,
                seed=539,
                layer_start=2,
                layer_end=4,
                supervised_layer_scope="selected",
            )

        self.assertEqual(training_widths, [3, 3, 3])
        self.assertEqual(scored_widths, [3, 3, 3])

    def test_fisher_reports_only_out_of_fold_scores(self) -> None:
        curves = np.array(
            [
                [0.0, 0.1, 0.2],
                [0.1, 0.1, 0.3],
                [0.2, 0.2, 0.4],
                [1.0, 1.1, 1.2],
                [1.1, 1.1, 1.3],
                [1.2, 1.2, 1.4],
            ]
        )
        labels = np.array([0, 0, 0, 1, 1, 1])

        scores = compute_detection_scores(curves, labels, mode="fisher", n_folds=3, seed=539)

        self.assertEqual(len(scores), len(labels))
        self.assertTrue(all(isinstance(score, float) for score in scores))

    def test_fisher_epsilon_can_be_configured(self) -> None:
        curves = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        )
        labels = np.array([0, 0, 1, 1])

        default_scores = compute_detection_scores(
            curves,
            labels,
            mode="fisher",
            n_folds=2,
            seed=539,
            fisher_epsilon=1e-8,
        )
        smaller_epsilon_scores = compute_detection_scores(
            curves,
            labels,
            mode="fisher",
            n_folds=2,
            seed=539,
            fisher_epsilon=1e-12,
        )

        self.assertGreater(max(smaller_epsilon_scores), max(default_scores))

    def test_supervised_modes_reject_single_sample_class(self) -> None:
        curves = np.array([[0.0, 0.1], [0.2, 0.3], [1.0, 1.1]])
        labels = np.array([0, 0, 1])

        with self.assertRaisesRegex(ValueError, "at least 2 samples"):
            compute_detection_scores(curves, labels, mode="logreg", n_folds=5, seed=539)


if __name__ == "__main__":
    unittest.main()
