import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


SCORING_MODES = ("trapz", "fisher", "logreg")
DEFAULT_FISHER_EPSILON = 1e-8
DEFAULT_LOGREG_C = 0.5
DEFAULT_LAYER_START = 16
DEFAULT_LAYER_END = 29
SUPERVISED_LAYER_SCOPES = ("all", "selected")


def _as_2d_curves(f_curves) -> np.ndarray:
    curves = np.asarray(f_curves, dtype=float)
    if curves.ndim != 2:
        raise ValueError(f"Expected a 2D array of layer curves, got shape {curves.shape}.")
    if curves.shape[0] == 0:
        raise ValueError("Expected at least one sample.")
    if curves.shape[1] == 0:
        raise ValueError("Expected at least one layer score per sample.")
    return curves


def _as_labels(labels, sample_count: int) -> np.ndarray:
    labels_arr = np.asarray(labels)
    if labels_arr.shape[0] != sample_count:
        raise ValueError(
            f"Expected {sample_count} labels for {sample_count} samples, got {labels_arr.shape[0]}."
        )
    return labels_arr


def _validate_mode(mode: str) -> str:
    if mode not in SCORING_MODES:
        raise ValueError(f"Unknown scoring mode {mode!r}; expected one of {SCORING_MODES}.")
    return mode


def _select_layers(curves: np.ndarray, layer_start: int, layer_end: int) -> np.ndarray:
    if layer_start < 0 or layer_end < layer_start:
        raise ValueError("layer_start and layer_end must define a non-empty inclusive range.")
    selected = curves[:, layer_start : layer_end + 1]
    if selected.shape[1] == 0:
        raise ValueError(
            f"Layer range {layer_start}:{layer_end} is outside available layers 0:{curves.shape[1] - 1}."
        )
    return selected


def _fold_count(labels: np.ndarray, requested_folds: int) -> int:
    if requested_folds < 2:
        raise ValueError("n_folds must be at least 2 for supervised scoring.")

    _, class_counts = np.unique(labels, return_counts=True)
    if class_counts.shape[0] < 2:
        raise ValueError("Supervised scoring requires at least 2 classes.")

    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        raise ValueError("Supervised scoring requires each class to have at least 2 samples.")

    return min(int(requested_folds), min_class_count)


def _validate_fisher_epsilon(fisher_epsilon: float) -> float:
    fisher_epsilon = float(fisher_epsilon)
    if not np.isfinite(fisher_epsilon) or fisher_epsilon <= 0:
        raise ValueError("fisher_epsilon must be a positive finite value.")
    return fisher_epsilon


def _validate_logreg_c(logreg_c: float) -> float:
    logreg_c = float(logreg_c)
    if not np.isfinite(logreg_c) or logreg_c <= 0:
        raise ValueError("logreg_c must be a positive finite value.")
    return logreg_c


def _validate_supervised_layer_scope(supervised_layer_scope: str) -> str:
    if supervised_layer_scope not in SUPERVISED_LAYER_SCOPES:
        raise ValueError(
            "Unknown supervised_layer_scope "
            f"{supervised_layer_scope!r}; expected one of {SUPERVISED_LAYER_SCOPES}."
        )
    return supervised_layer_scope


def _fisher_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    fisher_epsilon: float,
) -> np.ndarray:
    unsafe_curves = x_train[y_train == 1]
    safe_curves = x_train[y_train != 1]
    if unsafe_curves.shape[0] == 0 or safe_curves.shape[0] == 0:
        raise ValueError("Fisher scoring requires training data with labels 1 and non-1.")

    mean_diff = unsafe_curves.mean(axis=0) - safe_curves.mean(axis=0)
    avg_class_var = 0.5 * (unsafe_curves.var(axis=0) + safe_curves.var(axis=0))
    weights = mean_diff / (np.sqrt(avg_class_var) + fisher_epsilon)
    return x_test @ weights


def _supervised_out_of_fold_scores(
    curves: np.ndarray,
    labels: np.ndarray,
    mode: str,
    n_folds: int,
    seed: int,
    fisher_epsilon: float,
    logreg_c: float,
) -> list[float]:
    split_count = _fold_count(labels, n_folds)
    splitter = StratifiedKFold(n_splits=split_count, shuffle=True, random_state=seed)
    scores = np.empty(labels.shape[0], dtype=float)

    for train_idx, test_idx in splitter.split(curves, labels):
        x_train = curves[train_idx]
        y_train = labels[train_idx]
        x_test = curves[test_idx]

        if mode == "logreg":
            clf = LogisticRegression(
                penalty="l2",
                C=logreg_c,
                max_iter=1000,
                class_weight=None,
            ).fit(x_train, y_train)
            fold_scores = clf.decision_function(x_test)
        elif mode == "fisher":
            fold_scores = _fisher_scores(x_train, y_train, x_test, fisher_epsilon)
        else:
            raise ValueError(f"Unsupported supervised scoring mode {mode!r}.")

        scores[test_idx] = np.asarray(fold_scores, dtype=float)

    return scores.tolist()


def compute_detection_scores(
    f_curves,
    labels,
    mode: str,
    n_folds: int = 5,
    seed: int = 539,
    layer_start: int = DEFAULT_LAYER_START,
    layer_end: int = DEFAULT_LAYER_END,
    fisher_epsilon: float = DEFAULT_FISHER_EPSILON,
    logreg_c: float = DEFAULT_LOGREG_C,
    supervised_layer_scope: str = "all",
) -> list[float]:
    curves = _as_2d_curves(f_curves)
    labels_arr = _as_labels(labels, curves.shape[0])
    mode = _validate_mode(mode)
    supervised_layer_scope = _validate_supervised_layer_scope(supervised_layer_scope)

    if mode == "trapz":
        selected_curves = _select_layers(curves, layer_start, layer_end)
        return [float(np.trapz(curve)) for curve in selected_curves]

    if mode == "fisher":
        fisher_epsilon = _validate_fisher_epsilon(fisher_epsilon)
    elif mode == "logreg":
        logreg_c = _validate_logreg_c(logreg_c)

    if supervised_layer_scope == "selected":
        curves = _select_layers(curves, layer_start, layer_end)

    return _supervised_out_of_fold_scores(
        curves,
        labels_arr,
        mode,
        n_folds,
        seed,
        fisher_epsilon,
        logreg_c,
    )
