"""Sleep-wake-specific Garmin data loader.

I/O MODULE: reads CSV files from disk.
Model-specific: Garmin CSV format, 4 observation channels.

Date:    15 April 2026
Version: 5.0 (model-agnostic framework)
"""

import os
import numpy as np
import pandas as pd
from typing import NamedTuple

# Grid constants (could be passed as args, but fixed for sleep-wake)
DATA_START = "2026-04-05"
DATA_END = "2026-04-14"
T_DAYS = 9
DT_HOURS = 5.0 / 60.0
T_HOURS = T_DAYS * 24.0
T_STEPS = int(T_HOURS / DT_HOURS)

_DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "my_intraday_data")


class SleepWakeObsData(NamedTuple):
    """Raw observation data from Garmin watch."""
    t_hours: np.ndarray
    sleep_t_hours: np.ndarray
    sleep_labels: np.ndarray
    hr_t_idx: np.ndarray
    hr_bpm: np.ndarray
    hr_exercise: np.ndarray
    stress_t_idx: np.ndarray
    stress_score: np.ndarray
    step_t_idx: np.ndarray
    step_counts: np.ndarray
    step_is_running: np.ndarray
    daily_active_bins: np.ndarray
    daily_start_idx: np.ndarray
    n_hr: int
    n_stress: int
    n_sleep_epochs: int
    n_step_bins: int
    n_days: int


def _utc_to_hours(ts, t0):
    return (ts - t0).total_seconds() / 3600.0


def _hours_to_idx(t_h):
    idx = np.round(t_h / DT_HOURS).astype(np.int32)
    return np.clip(idx, 0, T_STEPS - 1)


def _thin(t_idx, values, extras=None):
    unique = np.unique(t_idx)
    vals = np.zeros(len(unique), dtype=np.float32)
    ext_out = [np.zeros(len(unique), dtype=np.float32) for _ in (extras or [])]
    for i, u in enumerate(unique):
        m = (t_idx == u)
        vals[i] = np.mean(values[m])
        for j, arr in enumerate(extras or []):
            ext_out[j][i] = np.mean(arr[m])
    if extras:
        return unique, vals, ext_out
    return unique, vals


def load_data(data_dir=None):
    """Load Garmin CSV data for the sleep-wake model.

    Args:
        data_dir: Path to CSV directory. Defaults to ./my_intraday_data/.

    Returns:
        SleepWakeObsData NamedTuple.
    """
    if data_dir is None:
        data_dir = os.environ.get("DATA_DIR", _DEFAULT_DIR)

    t0 = pd.Timestamp(DATA_START, tz='UTC')
    t_end = pd.Timestamp(DATA_END, tz='UTC')
    last_date = (t_end - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    # Sleep
    stg = pd.read_csv(f"{data_dir}/sleep_stages.csv",
                       parse_dates=["epoch_start", "epoch_end"])
    stg = stg[(stg.is_nap == False) & (stg.epoch_end > t0) & (stg.epoch_start < t_end)]
    lmap = {'awake': 0, 'light': 1, 'rem': 2, 'deep': 3}
    sl_t, sl_l = [], []
    for _, ep in stg.iterrows():
        mid = ep.epoch_start + (ep.epoch_end - ep.epoch_start) / 2
        th = _utc_to_hours(mid, t0)
        if 0 <= th < T_HOURS:
            sl_t.append(th); sl_l.append(lmap.get(ep.sleep_level, 0))

    # Steps (load before HR for exercise indicator)
    sb = pd.read_csv(f"{data_dir}/sdk_steps_binned.csv",
                      parse_dates=["bin_start_bst", "bin_end_bst"])
    sb = sb[(sb.date_bst >= DATA_START) & (sb.date_bst <= last_date)].copy()
    sb['mid_utc'] = sb.bin_start_bst.dt.tz_convert('UTC') + \
        (sb.bin_end_bst.dt.tz_convert('UTC') - sb.bin_start_bst.dt.tz_convert('UTC')) / 2

    n_bins = 96 * T_DAYS
    bins_start = pd.date_range(t0, periods=n_bins, freq='15min', tz='UTC')
    bins_mid_h = np.array([_utc_to_hours(ts + pd.Timedelta(minutes=7.5), t0)
                            for ts in bins_start], dtype=np.float32)
    step_counts = np.zeros(n_bins, dtype=np.float32)
    step_running = np.zeros(n_bins, dtype=bool)
    for _, r in sb.iterrows():
        idx = np.argmin(np.abs(bins_mid_h - _utc_to_hours(r.mid_utc, t0)))
        step_counts[idx] = r.steps
        if r.activity_type == 'running':
            step_running[idx] = True

    step_t_idx = _hours_to_idx(bins_mid_h)

    # Exercise mask
    ex_mask = np.zeros(T_STEPS, dtype=np.float32)
    for i, (ts, cnt) in enumerate(zip(bins_start, step_counts)):
        if cnt > 200:
            sh = _utc_to_hours(ts, t0)
            ex_mask[max(0, int(sh / DT_HOURS)):min(T_STEPS, int((sh + 0.25) / DT_HOURS) + 1)] = 1.0

    # HR
    hr = pd.read_csv(f"{data_dir}/sdk_intraday_hr.csv", parse_dates=["timestamp_utc"])
    hr = hr[(hr.timestamp_utc >= t0) & (hr.timestamp_utc < t_end) & (hr.heart_rate_bpm > 0)]
    hr_th = hr.timestamp_utc.apply(lambda ts: _utc_to_hours(ts, t0)).values
    hr_idx_raw = _hours_to_idx(hr_th)
    hr_ex_raw = ex_mask[hr_idx_raw]
    hr_idx, hr_bpm, (hr_ex,) = _thin(hr_idx_raw, hr.heart_rate_bpm.values.astype(np.float32), [hr_ex_raw])
    hr_ex = (hr_ex > 0.5).astype(np.float32)

    # Stress
    st = pd.read_csv(f"{data_dir}/sdk_intraday_stress.csv", parse_dates=["timestamp_utc"])
    st = st[(st.timestamp_utc >= t0) & (st.timestamp_utc < t_end) & (st.stress_score >= 0)]
    st_th = st.timestamp_utc.apply(lambda ts: _utc_to_hours(ts, t0)).values
    st_idx, st_score = _thin(_hours_to_idx(st_th), st.stress_score.values.astype(np.float32))

    # Daily active bins
    dab = np.zeros(T_DAYS, dtype=np.float32)
    dsi = np.zeros(T_DAYS, dtype=np.int32)
    for d in range(T_DAYS):
        s, e = d * 96, min((d + 1) * 96, n_bins)
        dab[d] = np.sum(step_counts[s:e] > 200)
        dsi[d] = int(d * 24.0 / DT_HOURS)

    return SleepWakeObsData(
        t_hours=np.arange(T_STEPS) * DT_HOURS,
        sleep_t_hours=np.array(sl_t, np.float32), sleep_labels=np.array(sl_l, np.int32),
        hr_t_idx=hr_idx, hr_bpm=hr_bpm, hr_exercise=hr_ex,
        stress_t_idx=st_idx, stress_score=st_score,
        step_t_idx=step_t_idx, step_counts=step_counts, step_is_running=step_running,
        daily_active_bins=dab, daily_start_idx=dsi,
        n_hr=len(hr_bpm), n_stress=len(st_score),
        n_sleep_epochs=len(sl_t), n_step_bins=n_bins, n_days=T_DAYS)
