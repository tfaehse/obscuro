from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter

from .common import tlwh_to_xyah, xyah_to_tlwh

STATE_SIZE = 8
MEAS_SIZE = 4


def create_kalman_filter() -> KalmanFilter:
    kf = KalmanFilter(dim_x=STATE_SIZE, dim_z=MEAS_SIZE)
    dt = 1.0

    kf.F = np.eye(STATE_SIZE)
    for i in range(MEAS_SIZE):
        kf.F[i, i + MEAS_SIZE] = dt

    kf.H = np.zeros((MEAS_SIZE, STATE_SIZE))
    kf.H[:MEAS_SIZE, :MEAS_SIZE] = np.eye(MEAS_SIZE)

    kf.R *= 0.01
    kf.P *= 10.0
    kf.Q = np.eye(STATE_SIZE)
    q_pos = 1.0
    q_vel = 0.01
    for i in range(MEAS_SIZE):
        kf.Q[i, i] = q_pos
        kf.Q[i + MEAS_SIZE, i + MEAS_SIZE] = q_vel

    return kf


def initiate(tlwh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kf = create_kalman_filter()
    measurement = tlwh_to_xyah(tlwh)
    kf.x[:MEAS_SIZE] = measurement.reshape(MEAS_SIZE, 1)
    return kf.x.flatten(), kf.P.copy()


def predict(mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kf = create_kalman_filter()
    kf.x = mean.reshape(-1, 1)
    kf.P = covariance.copy()
    kf.predict()
    return kf.x.flatten(), kf.P.copy()


def project(mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kf = create_kalman_filter()
    kf.x = mean.reshape(-1, 1)
    kf.P = covariance.copy()
    return kf.x[:MEAS_SIZE].flatten(), (kf.H @ kf.P @ kf.H.T + kf.R)


def update(
    mean: np.ndarray,
    covariance: np.ndarray,
    measurement_tlwh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    measurement = tlwh_to_xyah(measurement_tlwh)
    kf = create_kalman_filter()
    kf.x = mean.reshape(-1, 1)
    kf.P = covariance.copy()
    kf.update(measurement)
    return kf.x.flatten(), kf.P.copy()


def mean_to_tlwh(mean: np.ndarray) -> np.ndarray:
    return xyah_to_tlwh(mean[:MEAS_SIZE])
