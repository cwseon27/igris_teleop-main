from __future__ import annotations

import copy
import random
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression


TORCH_INSTALL_HINT = (
    "PyTorch is required for model_type='gru', 'lstm', 'rnn', 'tcn', or 'cnn1d'. "
    "Install it in the training/inference environment with: pip install torch"
)

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
except ImportError:  # Tree models must remain usable without PyTorch.
    torch = None
    nn = None
    F = None
    DataLoader = None


def torch_available() -> bool:
    return torch is not None and nn is not None and DataLoader is not None


def require_torch():
    if not torch_available():
        raise ImportError(TORCH_INSTALL_HINT)


class _NumpySequenceDataset:
    """Normalize one sequence at a time to avoid copying the full dataset."""

    def __init__(
        self,
        X,
        y,
        mean: np.ndarray,
        std: np.ndarray,
        training: bool = False,
        input_noise_std: float = 0.0,
    ):
        require_torch()
        self.X = X
        self.y = np.asarray(y, dtype=np.float32)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.training = bool(training)
        self.input_noise_std = max(0.0, float(input_noise_std))

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, index):
        sequence = np.asarray(self.X[index], dtype=np.float32)
        sequence = (sequence - self.mean) / self.std
        if self.training and self.input_noise_std > 0.0:
            noise = np.random.normal(0.0, self.input_noise_std, size=sequence.shape).astype(np.float32)
            sequence = sequence + noise
        return torch.from_numpy(sequence), torch.tensor(self.y[index], dtype=torch.float32)


if nn is not None:

    class _RecurrentBinaryNetwork(nn.Module):
        def __init__(
            self,
            model_type: str,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
            bidirectional: bool,
        ):
            super().__init__()
            self.model_type = str(model_type).lower()
            self.bidirectional = bool(bidirectional)
            recurrent_dropout = float(dropout) if int(num_layers) > 1 else 0.0

            recurrent_kwargs = dict(
                input_size=int(input_size),
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                batch_first=True,
                dropout=recurrent_dropout,
                bidirectional=self.bidirectional,
            )
            if self.model_type == "gru":
                self.recurrent = nn.GRU(**recurrent_kwargs)
            elif self.model_type == "lstm":
                self.recurrent = nn.LSTM(**recurrent_kwargs)
            elif self.model_type == "rnn":
                self.recurrent = nn.RNN(nonlinearity="tanh", **recurrent_kwargs)
            else:
                raise ValueError(f"unsupported recurrent model_type: {self.model_type}")

            directions = 2 if self.bidirectional else 1
            head_size = int(hidden_size) * directions
            self.head = nn.Sequential(
                nn.LayerNorm(head_size),
                nn.Dropout(float(dropout)),
                nn.Linear(head_size, 1),
            )

        def forward(self, X):
            _, hidden = self.recurrent(X)
            if self.model_type == "lstm":
                hidden = hidden[0]

            directions = 2 if self.bidirectional else 1
            batch_size = X.shape[0]
            hidden_size = hidden.shape[-1]
            hidden = hidden.reshape(-1, directions, batch_size, hidden_size)[-1]
            if directions == 2:
                hidden = torch.cat([hidden[0], hidden[1]], dim=1)
            else:
                hidden = hidden[0]
            return self.head(hidden).squeeze(1)


    class _CausalConv1d(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
            super().__init__()
            self.left_padding = int(dilation) * (int(kernel_size) - 1)
            self.conv = nn.Conv1d(
                int(in_channels),
                int(out_channels),
                kernel_size=int(kernel_size),
                dilation=int(dilation),
            )

        def forward(self, X):
            return self.conv(F.pad(X, (self.left_padding, 0)))


    class _TemporalResidualBlock(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int,
            dropout: float,
        ):
            super().__init__()
            self.conv1 = _CausalConv1d(in_channels, out_channels, kernel_size, dilation)
            self.norm1 = nn.GroupNorm(1, out_channels)
            self.conv2 = _CausalConv1d(out_channels, out_channels, kernel_size, dilation)
            self.norm2 = nn.GroupNorm(1, out_channels)
            self.dropout = nn.Dropout(float(dropout))
            self.residual = (
                nn.Identity()
                if int(in_channels) == int(out_channels)
                else nn.Conv1d(int(in_channels), int(out_channels), kernel_size=1)
            )

        def forward(self, X):
            residual = self.residual(X)
            output = self.dropout(F.gelu(self.norm1(self.conv1(X))))
            output = self.dropout(F.gelu(self.norm2(self.conv2(output))))
            return F.gelu(output + residual)


    class _TCNBinaryNetwork(nn.Module):
        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            kernel_size: int,
            dropout: float,
        ):
            super().__init__()
            blocks = []
            in_channels = int(input_size)
            for layer_index in range(int(num_layers)):
                blocks.append(
                    _TemporalResidualBlock(
                        in_channels=in_channels,
                        out_channels=int(hidden_size),
                        kernel_size=int(kernel_size),
                        dilation=2**layer_index,
                        dropout=float(dropout),
                    )
                )
                in_channels = int(hidden_size)
            self.blocks = nn.Sequential(*blocks)
            self.head = nn.Sequential(
                nn.LayerNorm(int(hidden_size)),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_size), 1),
            )

        def forward(self, X):
            features = self.blocks(X.transpose(1, 2))
            return self.head(features[:, :, -1]).squeeze(1)


    class _CNN1DBinaryNetwork(nn.Module):
        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            kernel_size: int,
            dropout: float,
        ):
            super().__init__()
            layers = []
            in_channels = int(input_size)
            padding = int(kernel_size) // 2
            for _ in range(int(num_layers)):
                layers.extend(
                    [
                        nn.Conv1d(
                            in_channels,
                            int(hidden_size),
                            kernel_size=int(kernel_size),
                            padding=padding,
                        ),
                        nn.GroupNorm(1, int(hidden_size)),
                        nn.GELU(),
                        nn.Dropout(float(dropout)),
                    ]
                )
                in_channels = int(hidden_size)
            self.features = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(
                nn.LayerNorm(int(hidden_size)),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_size), 1),
            )

        def forward(self, X):
            features = self.features(X.transpose(1, 2))
            pooled = self.pool(features).squeeze(-1)
            return self.head(pooled).squeeze(1)

else:

    class _RecurrentBinaryNetwork:
        def __init__(self, *args, **kwargs):
            require_torch()


    class _TCNBinaryNetwork:
        def __init__(self, *args, **kwargs):
            require_torch()


    class _CNN1DBinaryNetwork:
        def __init__(self, *args, **kwargs):
            require_torch()


class TorchSequenceBinaryModel:
    """Recurrent/convolutional sequence classifier with the shared confidence API."""

    def __init__(
        self,
        model_type: str = "gru",
        hidden_size: int = 128,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = False,
        epochs: int = 30,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 5,
        gradient_clip: float = 1.0,
        device: str = "auto",
        inference_device: str = "cpu",
        num_workers: int = 0,
        predict_batch_size: int = 512,
        random_state: int = 42,
        calibrate: bool = True,
        label_smoothing: float = 0.0,
        input_noise_std: float = 0.0,
        early_stopping_min_delta: float = 1e-4,
    ):
        self.model_type = str(model_type).lower()
        if self.model_type not in ("gru", "lstm", "rnn", "tcn", "cnn1d"):
            raise ValueError("model_type must be one of: gru, lstm, rnn, tcn, cnn1d")

        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.gradient_clip = float(gradient_clip)
        self.device = str(device).lower()
        self.inference_device = str(inference_device).lower()
        self.num_workers = int(num_workers)
        self.predict_batch_size = int(predict_batch_size)
        self.random_state = int(random_state)
        self.calibrate = bool(calibrate)
        self.label_smoothing = float(label_smoothing)
        self.input_noise_std = float(input_noise_std)
        self.early_stopping_min_delta = float(early_stopping_min_delta)

        self.input_size = None
        self.window_size = None
        self.feature_mean = None
        self.feature_std = None
        self.network = None
        self.calibrator = None
        self.constant_class = None
        self.training_history = []
        self.training_device = "cpu"
        self.active_inference_device = "cpu"
        self.parameter_count = 0

    def _validate_hyperparameters(self):
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.model_type in ("tcn", "cnn1d"):
            if self.kernel_size < 3 or self.kernel_size % 2 == 0:
                raise ValueError("kernel_size must be an odd integer >= 3 for tcn/cnn1d")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.predict_batch_size < 1:
            raise ValueError("predict_batch_size must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.label_smoothing < 0.5:
            raise ValueError("label_smoothing must be in [0, 0.5)")
        if self.input_noise_std < 0.0:
            raise ValueError("input_noise_std must be >= 0")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be >= 0")

    def _set_random_seed(self):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _resolve_device(self):
        requested = self.device
        return self._resolve_named_device(requested)

    def _resolve_inference_device(self):
        requested = getattr(self, "inference_device", "cpu")
        return self._resolve_named_device(requested)

    def _resolve_named_device(self, requested):
        requested = str(requested).lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if requested != "cpu" and not requested.startswith("cuda"):
            raise ValueError("device must be 'auto', 'cpu', 'cuda', or a CUDA device such as 'cuda:0'")
        return torch.device(requested)

    def _ensure_network_device(self, device):
        if self.network is None:
            return
        try:
            current_device = next(self.network.parameters()).device
        except StopIteration:
            current_device = torch.device("cpu")
        if current_device != device:
            self.network.to(device)
        self.active_inference_device = str(device)

    def _validate_X(self, X):
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"expected X shape (N, T, C), got {X.shape}")
        if self.window_size is not None and X.shape[1] != self.window_size:
            raise ValueError(f"window_size mismatch: {X.shape[1]} != {self.window_size}")
        if self.input_size is not None and X.shape[2] != self.input_size:
            raise ValueError(f"feature_dim mismatch: {X.shape[2]} != {self.input_size}")
        return X

    def _make_loader(self, X, y, shuffle: bool, device, training: bool = False):
        dataset = _NumpySequenceDataset(
            X,
            y,
            self.feature_mean,
            self.feature_std,
            training=training,
            input_noise_std=self.input_noise_std if training else 0.0,
        )
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=bool(shuffle),
            num_workers=self.num_workers,
            pin_memory=device.type == "cuda",
            generator=generator,
        )

    def _compute_feature_stats(self, X, chunk_size: int = 1024):
        feature_sum = np.zeros(X.shape[2], dtype=np.float64)
        feature_sum_sq = np.zeros(X.shape[2], dtype=np.float64)
        count = 0

        for start in range(0, X.shape[0], int(chunk_size)):
            stop = min(start + int(chunk_size), X.shape[0])
            chunk = np.asarray(X[start:stop], dtype=np.float32)
            feature_sum += np.sum(chunk, axis=(0, 1), dtype=np.float64)
            squared = np.square(chunk, dtype=np.float64)
            feature_sum_sq += np.sum(squared, axis=(0, 1), dtype=np.float64)
            count += int(chunk.shape[0] * chunk.shape[1])

        if count <= 0:
            raise ValueError("cannot compute normalization statistics from empty X")
        mean = feature_sum / count
        variance = np.maximum((feature_sum_sq / count) - np.square(mean), 0.0)
        std = np.sqrt(variance)
        return mean.astype(np.float32), np.maximum(std.astype(np.float32), np.float32(1e-6))

    def _epoch_loss(self, loader, criterion, device, optimizer=None):
        training = optimizer is not None
        self.network.train(training)
        total_loss = 0.0
        total_count = 0

        for sequences, labels in loader:
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training and self.label_smoothing > 0.0:
                smooth = float(self.label_smoothing)
                labels = labels * (1.0 - smooth) + 0.5 * smooth

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                logits = self.network(sequences)
                loss = criterion(logits, labels)
                if training:
                    loss.backward()
                    if self.gradient_clip > 0.0:
                        nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
                    optimizer.step()

            batch_count = int(labels.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_count
            total_count += batch_count

        return total_loss / max(total_count, 1)

    def fit(self, X, y, X_val=None, y_val=None):
        require_torch()
        self._validate_hyperparameters()
        X = self._validate_X(X)
        y = np.asarray(y, dtype=np.int64)
        if X.shape[0] != y.shape[0] or y.size == 0:
            raise ValueError("X and y must contain the same non-zero number of samples")

        unique = np.unique(y)
        if unique.size < 2:
            self.constant_class = int(unique[0])
            return self

        self.window_size = int(X.shape[1])
        self.input_size = int(X.shape[2])
        self.feature_mean, self.feature_std = self._compute_feature_stats(X)

        self._set_random_seed()
        device = self._resolve_device()
        self.training_device = str(device)
        if self.model_type in ("gru", "lstm", "rnn"):
            self.network = _RecurrentBinaryNetwork(
                model_type=self.model_type,
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
            ).to(device)
        elif self.model_type == "tcn":
            self.network = _TCNBinaryNetwork(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
            ).to(device)
        else:
            self.network = _CNN1DBinaryNetwork(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
            ).to(device)

        n_positive = int(np.sum(y == 1))
        n_negative = int(np.sum(y == 0))
        pos_weight = float(n_negative / max(n_positive, 1))
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
        )
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        train_loader = self._make_loader(X, y, shuffle=True, device=device, training=True)
        has_validation = X_val is not None and y_val is not None and len(y_val) > 0
        val_loader = None
        if has_validation:
            X_val = self._validate_X(X_val)
            y_val = np.asarray(y_val, dtype=np.int64)
            val_loader = self._make_loader(X_val, y_val, shuffle=False, device=device, training=False)

        self.parameter_count = sum(parameter.numel() for parameter in self.network.parameters())
        print(
            f"[{self.model_type}] device={device}, parameters={self.parameter_count:,}, "
            f"train_samples={len(y)}, val_samples={0 if y_val is None else len(y_val)}"
        )

        best_loss = float("inf")
        best_state = None
        stale_epochs = 0
        self.training_history = []

        for epoch in range(1, self.epochs + 1):
            train_loss = self._epoch_loss(train_loader, criterion, device, optimizer=optimizer)
            if val_loader is not None:
                with torch.no_grad():
                    val_loss = self._epoch_loss(val_loader, criterion, device, optimizer=None)
                monitored_loss = val_loss
            else:
                val_loss = None
                monitored_loss = train_loss

            history_item = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": None if val_loss is None else float(val_loss),
            }
            self.training_history.append(history_item)
            val_text = "n/a" if val_loss is None else f"{val_loss:.6f}"
            print(
                f"[{self.model_type}] epoch={epoch:03d}/{self.epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_text}"
            )

            if monitored_loss < best_loss - self.early_stopping_min_delta:
                best_loss = monitored_loss
                best_state = copy.deepcopy(
                    {name: tensor.detach().cpu() for name, tensor in self.network.state_dict().items()}
                )
                stale_epochs = 0
            else:
                stale_epochs += 1
                if self.patience > 0 and stale_epochs >= self.patience:
                    print(f"[{self.model_type}] early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            self.network.load_state_dict(best_state)
        self._ensure_network_device(self._resolve_inference_device())
        self.network.eval()

        if self.calibrate and has_validation and np.unique(y_val).size >= 2:
            try:
                logits = self._predict_logits(X_val)
                self.calibrator = LogisticRegression(max_iter=1000, random_state=self.random_state)
                self.calibrator.fit(logits.reshape(-1, 1), y_val)
            except Exception as exc:
                warnings.warn(f"probability calibration failed for {self.model_type}: {exc}")
                self.calibrator = None

        return self

    def _predict_logits(self, X) -> np.ndarray:
        require_torch()
        if self.network is None:
            raise RuntimeError("model has not been fitted")
        X = self._validate_X(X)
        outputs = []
        device = self._resolve_inference_device()
        self._ensure_network_device(device)
        self.network.eval()

        with torch.inference_mode():
            for start in range(0, X.shape[0], self.predict_batch_size):
                stop = min(start + self.predict_batch_size, X.shape[0])
                batch = np.asarray(X[start:stop], dtype=np.float32)
                batch = (batch - self.feature_mean) / self.feature_std
                batch_tensor = torch.from_numpy(batch).to(device, non_blocking=device.type == "cuda")
                logits = self.network(batch_tensor)
                outputs.append(logits.detach().cpu().numpy().astype(np.float32))

        if not outputs:
            return np.asarray([], dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    def predict_confidence(self, X) -> np.ndarray:
        X = self._validate_X(X)
        if self.constant_class is not None:
            return np.full(X.shape[0], float(self.constant_class), dtype=np.float32)

        logits = self._predict_logits(X)
        if self.calibrator is not None:
            confidence = self.calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
        else:
            logits64 = logits.astype(np.float64)
            confidence = np.empty_like(logits64)
            positive = logits64 >= 0.0
            confidence[positive] = 1.0 / (1.0 + np.exp(-logits64[positive]))
            exp_values = np.exp(logits64[~positive])
            confidence[~positive] = exp_values / (1.0 + exp_values)
        return np.clip(np.asarray(confidence, dtype=np.float32), 0.0, 1.0)

    def predict_proba(self, X) -> np.ndarray:
        confidence = self.predict_confidence(X)
        return np.column_stack([1.0 - confidence, confidence]).astype(np.float32)

    def __getstate__(self):
        state = self.__dict__.copy()
        if torch_available() and state.get("network") is not None:
            state["network"].to("cpu")
            state["active_inference_device"] = "cpu"
        return state
