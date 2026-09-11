# # utils/kf_tau.py
# import numpy as np
# import threading

# class KalmanFilter:
#     """1D 칼만 필터 (상태=토크)"""
#     def __init__(self, F, H, Q, R, x0, P0):
#         self.F = F; self.H = H; self.Q = Q; self.R = R
#         self.x = x0  # (1,1)
#         self.P = P0  # (1,1)

#     def Step(self, y):
#         # Predict
#         x_k = self.F @ self.x
#         P_k = self.F @ self.P @ self.F.T + self.Q
#         # Update
#         S = self.H @ P_k @ self.H.T + self.R
#         K = P_k @ self.H.T @ np.linalg.inv(S)
#         self.x = x_k + K @ (y - self.H @ x_k)
#         self.P = P_k - K @ self.H @ P_k


# class TorqueKalmanBank:
#     """
#     N개 조인트 토크(tau_est) 전용 칼만 필터 뱅크.
#     모델: x_k = x_{k-1} + w   (F=[1])
#     관측: z_k = x_k + v       (H=[1])
#     """
#     def __init__(self, n_joints: int, q: float = 5.0, r: float = 20.0, P0: float = 1e3):
#         self._lock = threading.Lock()
#         self.n = int(n_joints)
#         self.filters = []

#         F = np.array([[1.0]], dtype=np.float64)
#         H = np.array([[1.0]], dtype=np.float64)
#         Q = np.array([[float(q)]], dtype=np.float64)
#         R = np.array([[float(r)]], dtype=np.float64)

#         x0 = np.array([[0.0]], dtype=np.float64)
#         P0m = np.array([[float(P0)]], dtype=np.float64)

#         for _ in range(self.n):
#             self.filters.append(KalmanFilter(F, H, Q, R, x0.copy(), P0m.copy()))

#         self._initialized = False

#     def reset_with_measurement(self, z: np.ndarray):
#         """첫 측정값으로 필터 상태를 동기화(초기 과도 응답 방지)."""
#         z = np.asarray(z, dtype=np.float64).reshape(-1)
#         if z.size != self.n:
#             raise ValueError(f"reset z length {z.size} != {self.n}")

#         with self._lock:
#             for i in range(self.n):
#                 self.filters[i].x[0, 0] = float(z[i])
#                 # P는 그대로 두거나(=빠른 수렴) 필요 시 작게 리셋 가능
#             self._initialized = True

#     def step(self, z: np.ndarray) -> np.ndarray:
#         z = np.asarray(z, dtype=np.float32).reshape(-1)
#         if z.size != self.n:
#             raise ValueError(f"z length {z.size} != {self.n}")

#         out = np.zeros(self.n, dtype=np.float32)
#         with self._lock:
#             if not self._initialized:
#                 # 첫 호출에서는 측정값으로 상태를 맞추고 그대로 반환
#                 self.reset_with_measurement(z)
#                 return z.astype(np.float32, copy=False)

#             for i in range(self.n):
#                 y = np.array([[float(z[i])]], dtype=np.float64)
#                 self.filters[i].Step(y)
#                 out[i] = float(self.filters[i].x[0, 0])

#         return out


import numpy as np
import threading


class KalmanFilter1D:
    def __init__(self, q: float, r: float, x0: float = 0.0, p0: float = 1e3):
        self.Q = float(q)
        self.R = float(r)
        self.x = float(x0)
        self.P = float(p0)

    def step(self, z: float) -> float:
        # Predict
        x_pred = self.x
        P_pred = self.P + self.Q

        # Update
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (z - x_pred)
        self.P = (1.0 - K) * P_pred
        return self.x


class TorqueKalmanBank:
    def __init__(self, n_joints: int, q: float = 5.0, r: float = 20.0, P0: float = 1e3):
        self._lock = threading.Lock()
        self.n = int(n_joints)
        self.filters = [KalmanFilter1D(q=q, r=r, x0=0.0, p0=P0) for _ in range(self.n)]
        self._initialized = False

    def reset_with_measurement(self, z: np.ndarray):
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        if z.size != self.n:
            raise ValueError(f"reset z length {z.size} != {self.n}")

        with self._lock:
            for i in range(self.n):
                self.filters[i].x = float(z[i])
            self._initialized = True

    def step(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        if z.size != self.n:
            raise ValueError(f"z length {z.size} != {self.n}")

        out = np.zeros(self.n, dtype=np.float32)

        with self._lock:
            if not self._initialized:
                for i in range(self.n):
                    self.filters[i].x = float(z[i])
                self._initialized = True
                return z.astype(np.float32, copy=False)

            for i in range(self.n):
                out[i] = self.filters[i].step(float(z[i]))

        return out