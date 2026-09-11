from __future__ import annotations

import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .dataset_utils import flatten_for_sklearn, to_sequence_channels_first


SKTIME_INSTALL_HINT = (
    "sktime is required for model_type='minirocket'. "
    "Install with: pip install sktime scikit-learn joblib numpy scipy numba"
)


class OptionalModelUnavailable(RuntimeError):
    pass


def _num_samples(X) -> int:
    return int(np.asarray(X).shape[0])


def _as_feature_array(X) -> np.ndarray:
    if hasattr(X, "to_numpy"):
        return X.to_numpy(dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


def _make_prefit_calibrator(base_estimator, method: str = "sigmoid"):
    try:
        return CalibratedClassifierCV(estimator=base_estimator, method=method, cv="prefit")
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base_estimator, method=method, cv="prefit")


def _can_calibrate(y) -> bool:
    return np.unique(np.asarray(y, dtype=np.int64)).size >= 2


def _proba_from_classes(proba, classes, n_samples: int) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float32)
    classes = np.asarray(classes, dtype=np.int64)

    if proba.ndim != 2:
        raise ValueError(f"expected probability shape (N, K), got {proba.shape}")

    out = np.zeros((n_samples, 2), dtype=np.float32)
    for col, klass in enumerate(classes):
        if klass == 0:
            out[:, 0] = proba[:, col]
        elif klass == 1:
            out[:, 1] = proba[:, col]
    missing = np.sum(out, axis=1) == 0.0
    if np.any(missing):
        out[missing, 0] = 1.0
    row_sum = np.sum(out, axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    return np.clip(out / row_sum, 0.0, 1.0)


def _confidence_from_estimator(estimator, X, n_samples: int) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            proba = estimator.predict_proba(X)
        classes = getattr(estimator, "classes_", np.asarray([0, 1]))
        return _proba_from_classes(proba, classes, n_samples)[:, 1]

    if hasattr(estimator, "decision_function"):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            scores = np.asarray(estimator.decision_function(X), dtype=np.float32)
        if scores.ndim == 2 and scores.shape[1] > 1:
            scores = scores[:, 1]
        scores = np.ravel(scores)
        confidence = 1.0 / (1.0 + np.exp(-scores))
        return np.clip(confidence, 0.0, 1.0)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names.*",
            category=UserWarning,
        )
        pred = np.asarray(estimator.predict(X), dtype=np.float32)
    return np.clip(pred, 0.0, 1.0)


class ConstantProbabilityModel:
    def __init__(self, constant_class: int):
        self.constant_class = int(constant_class)
        if self.constant_class not in (0, 1):
            raise ValueError("constant_class must be 0 or 1")
        self.classes_ = np.asarray([0, 1], dtype=np.int64)

    def fit(self, X, y=None):
        return self

    def predict_confidence(self, X) -> np.ndarray:
        n_samples = _num_samples(X)
        value = 1.0 if self.constant_class == 1 else 0.0
        return np.full(n_samples, value, dtype=np.float32)

    def predict_proba(self, X) -> np.ndarray:
        conf = self.predict_confidence(X)
        return np.column_stack([1.0 - conf, conf]).astype(np.float32)


class FlatSklearnBinaryModel:
    def __init__(
        self,
        model_type: str = "extratrees",
        n_estimators: int = 300,
        random_state: int = 42,
        calibrate: bool = True,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
        l2_regularization: float = 0.0,
        max_bins: int = 64,
    ):
        self.model_type = model_type
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)
        self.calibrate = bool(calibrate)
        self.learning_rate = float(learning_rate)
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.l2_regularization = float(l2_regularization)
        self.max_bins = int(max_bins)
        self.classifier = None
        self.calibrator = None
        self.constant_model = None

    def _make_classifier(self):
        if self.model_type == "extratrees":
            return ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                class_weight="balanced",
                n_jobs=-1,
            )
        if self.model_type == "randomforest":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                class_weight="balanced",
                n_jobs=-1,
            )
        if self.model_type == "histgb":
            return HistGradientBoostingClassifier(
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_leaf_nodes=self.max_leaf_nodes,
                l2_regularization=self.l2_regularization,
                max_bins=self.max_bins,
                early_stopping=False,
                random_state=self.random_state,
            )
        if self.model_type == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
            except ImportError as exc:
                raise OptionalModelUnavailable(
                    "lightgbm model is optional and not installed. Install with: pip install lightgbm"
                ) from exc
            return LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.max_leaf_nodes,
                objective="binary",
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1,
            )
        raise ValueError(f"unsupported flat model_type: {self.model_type}")

    def fit(self, X, y, X_val=None, y_val=None):
        y = np.asarray(y, dtype=np.int64)
        if y.size == 0:
            raise ValueError("cannot fit on an empty target array")
        unique = np.unique(y)
        if unique.size < 2:
            self.constant_model = ConstantProbabilityModel(int(unique[0]))
            return self

        X_flat = flatten_for_sklearn(X)
        self.classifier = self._make_classifier()
        if self.model_type == "histgb":
            sample_weight = compute_sample_weight(class_weight="balanced", y=y)
            self.classifier.fit(X_flat, y, sample_weight=sample_weight)
        else:
            self.classifier.fit(X_flat, y)

        if self.calibrate and X_val is not None and y_val is not None and len(y_val) > 0:
            y_val = np.asarray(y_val, dtype=np.int64)
            if _can_calibrate(y_val):
                try:
                    X_val_flat = flatten_for_sklearn(X_val)
                    self.calibrator = _make_prefit_calibrator(self.classifier)
                    self.calibrator.fit(X_val_flat, y_val)
                except Exception as exc:  # calibration compatibility varies by sklearn version
                    warnings.warn(f"calibration failed for {self.model_type}; using raw probabilities: {exc}")
                    self.calibrator = None
            else:
                warnings.warn(f"skipping calibration for {self.model_type}; validation set has one class")
        return self

    def predict_confidence(self, X) -> np.ndarray:
        if self.constant_model is not None:
            return self.constant_model.predict_confidence(X)
        X_flat = flatten_for_sklearn(X)
        estimator = self.calibrator if self.calibrator is not None else self.classifier
        confidence = _confidence_from_estimator(estimator, X_flat, X_flat.shape[0])
        return np.clip(np.asarray(confidence, dtype=np.float32), 0.0, 1.0)

    def predict_proba(self, X) -> np.ndarray:
        conf = self.predict_confidence(X)
        return np.column_stack([1.0 - conf, conf]).astype(np.float32)


class MiniRocketBinaryModel:
    def __init__(
        self,
        num_kernels: int = 10000,
        random_state: int = 42,
        calibrate: bool = True,
        use_scaler: bool = True,
    ):
        self.model_type = "minirocket"
        self.num_kernels = int(num_kernels)
        self.random_state = int(random_state)
        self.calibrate = bool(calibrate)
        self.use_scaler = bool(use_scaler)
        self.transformer = None
        self.scaler = None
        self.classifier = None
        self.calibrator = None
        self.constant_model = None

    def _make_transformer(self):
        try:
            from sktime.transformations.panel.rocket import MiniRocketMultivariate
        except ImportError as exc:
            raise ImportError(SKTIME_INSTALL_HINT) from exc

        try:
            return MiniRocketMultivariate(
                num_kernels=self.num_kernels,
                random_state=self.random_state,
                n_jobs=-1,
            )
        except ModuleNotFoundError as exc:
            raise ImportError(SKTIME_INSTALL_HINT) from exc
        except TypeError:
            try:
                return MiniRocketMultivariate(
                    num_kernels=self.num_kernels,
                    random_state=self.random_state,
                )
            except ModuleNotFoundError as exc:
                raise ImportError(SKTIME_INSTALL_HINT) from exc

    def _transform(self, X) -> np.ndarray:
        X_seq = to_sequence_channels_first(X)
        X_features = self.transformer.transform(X_seq)
        X_features = _as_feature_array(X_features)
        if self.scaler is not None:
            X_features = self.scaler.transform(X_features)
        return X_features

    def fit(self, X, y, X_val=None, y_val=None):
        y = np.asarray(y, dtype=np.int64)
        if y.size == 0:
            raise ValueError("cannot fit on an empty target array")
        unique = np.unique(y)
        if unique.size < 2:
            self.constant_model = ConstantProbabilityModel(int(unique[0]))
            return self

        X_seq = to_sequence_channels_first(X)
        self.transformer = self._make_transformer()
        X_features = _as_feature_array(self.transformer.fit_transform(X_seq))

        if self.use_scaler:
            self.scaler = StandardScaler(with_mean=False)
            X_features = self.scaler.fit_transform(X_features)

        self.classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self.classifier.fit(X_features, y)

        if self.calibrate and X_val is not None and y_val is not None and len(y_val) > 0:
            y_val = np.asarray(y_val, dtype=np.int64)
            if _can_calibrate(y_val):
                try:
                    X_val_features = self._transform(X_val)
                    self.calibrator = _make_prefit_calibrator(self.classifier)
                    self.calibrator.fit(X_val_features, y_val)
                except Exception as exc:
                    warnings.warn(f"calibration failed for minirocket; using logistic probabilities: {exc}")
                    self.calibrator = None
            else:
                warnings.warn("skipping calibration for minirocket; validation set has one class")
        return self

    def predict_confidence(self, X) -> np.ndarray:
        if self.constant_model is not None:
            return self.constant_model.predict_confidence(X)
        X_features = self._transform(X)
        estimator = self.calibrator if self.calibrator is not None else self.classifier
        confidence = _confidence_from_estimator(estimator, X_features, X_features.shape[0])
        return np.clip(np.asarray(confidence, dtype=np.float32), 0.0, 1.0)

    def predict_proba(self, X) -> np.ndarray:
        conf = self.predict_confidence(X)
        return np.column_stack([1.0 - conf, conf]).astype(np.float32)


def create_model(model_type: str = "minirocket", **kwargs):
    model_type = str(model_type).lower()
    if model_type in ("gru", "lstm", "rnn", "tcn", "cnn1d"):
        from .deep_models import TorchSequenceBinaryModel, require_torch

        require_torch()
        return TorchSequenceBinaryModel(
            model_type=model_type,
            hidden_size=kwargs.get("hidden_size", 128),
            num_layers=kwargs.get("num_layers", 2),
            kernel_size=kwargs.get("kernel_size", 3),
            dropout=kwargs.get("dropout", 0.2),
            bidirectional=kwargs.get("bidirectional", False),
            epochs=kwargs.get("epochs", 30),
            batch_size=kwargs.get("batch_size", 256),
            learning_rate=kwargs.get("deep_learning_rate", 1e-3),
            weight_decay=kwargs.get("weight_decay", 1e-4),
            patience=kwargs.get("early_stopping_patience", 5),
            gradient_clip=kwargs.get("gradient_clip", 1.0),
            device=kwargs.get("device", "auto"),
            inference_device=kwargs.get("inference_device", "cpu"),
            num_workers=kwargs.get("num_workers", 0),
            predict_batch_size=kwargs.get("predict_batch_size", 512),
            random_state=kwargs.get("random_state", 42),
            calibrate=kwargs.get("calibrate", True),
            label_smoothing=kwargs.get("label_smoothing", 0.0),
            input_noise_std=kwargs.get("input_noise_std", 0.0),
            early_stopping_min_delta=kwargs.get("early_stopping_min_delta", 1e-4),
        )
    if model_type == "minirocket":
        return MiniRocketBinaryModel(
            num_kernels=kwargs.get("num_kernels", 10000),
            random_state=kwargs.get("random_state", 42),
            calibrate=kwargs.get("calibrate", True),
        )
    if model_type in ("extratrees", "randomforest", "histgb", "lightgbm"):
        return FlatSklearnBinaryModel(
            model_type=model_type,
            n_estimators=kwargs.get("n_estimators", 300),
            random_state=kwargs.get("random_state", 42),
            calibrate=kwargs.get("calibrate", True),
            learning_rate=kwargs.get("learning_rate", 0.05),
            max_leaf_nodes=kwargs.get("max_leaf_nodes", 31),
            l2_regularization=kwargs.get("l2_regularization", 0.0),
            max_bins=kwargs.get("max_bins", 64),
        )
    if model_type == "tsai_inceptiontime":
        try:
            import tsai  # noqa: F401
        except ImportError as exc:
            raise OptionalModelUnavailable("tsai model is optional and not installed") from exc
        raise OptionalModelUnavailable(
            "tsai_inceptiontime is optional and not implemented in this lightweight script"
        )
    raise ValueError(f"unsupported model_type: {model_type}")
