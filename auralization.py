import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal.windows import hann
from numpy.fft import ifft
from scipy.io.wavfile import write as write_wav


# =========================================================
# Public constants
# =========================================================
SAMPLE_TRAJECTORY_OPTIONS = [
    "Takeoff",
    "Overfly",
    "Landing",
    "Cosine Z",
    "Wind Turbine Tip",
]


# =========================================================
# Basic utilities
# =========================================================
def convert_db_to_amplitude(spl_db, pref=20e-6):
    """Convert dB SPL to pressure amplitude in Pa."""
    return pref * 10.0 ** (spl_db / 20.0)


def compute_doppler_shift(source_pos, observer_pos, source_vel, observer_vel, c=343.0):
    """
    Frequency Doppler factor based on radial velocities.
    """
    direction = observer_pos - source_pos
    distance = np.linalg.norm(direction)

    if distance < 1e-12:
        return 1.0

    unit_dir = direction / distance
    v_s_radial = np.dot(source_vel, unit_dir)
    v_o_radial = np.dot(observer_vel, unit_dir)

    denom = c - v_s_radial
    if abs(denom) < 1e-9:
        denom = 1e-9

    doppler_factor = (c + v_o_radial) / denom
    return max(doppler_factor, 1e-6)


def compute_convective_amplification(source_pos, observer_pos, source_vel, c=343.0):
    """
    Motion-dependent amplitude correction:
        1 / (1 - v_radial/c)^2
    """
    direction = observer_pos - source_pos
    distance = np.linalg.norm(direction)

    if distance < 1e-12:
        return 1.0

    unit_dir = direction / distance
    v_s_radial = np.dot(source_vel, unit_dir)

    denom = 1.0 - (v_s_radial / c)
    if abs(denom) < 1e-9:
        denom = 1e-9

    amplification = 1.0 / (denom ** 2)

    if not np.isfinite(amplification) or amplification <= 0.0:
        amplification = 1.0

    return amplification


def _power_sum_db(db_a, db_b):
    pa = 10.0 ** (db_a / 10.0)
    pb = 10.0 ** (db_b / 10.0)
    return 10.0 * np.log10(np.maximum(pa + pb, 1e-30))


def _normalize_signal(signal, scale):
    if scale < 1e-12:
        return np.zeros_like(signal)
    return signal / scale


def _write_wav(path, fs, signal):
    signal_clip = np.clip(signal, -1.0, 1.0)
    write_wav(path, fs, (signal_clip * 32767).astype(np.int16))


def _safe_norm(vec):
    return float(np.linalg.norm(vec))


def _require_librosa():
    try:
        import librosa
    except ImportError as e:
        raise ImportError(
            "This feature requires librosa. Install it with: pip install librosa soundfile"
        ) from e
    return librosa


def _load_audio_mono(audio_path, sr=None):
    librosa = _require_librosa()
    y, sr_out = librosa.load(audio_path, sr=sr, mono=True)
    if y.size == 0:
        raise ValueError("Loaded audio is empty.")
    return y.astype(float), int(sr_out)


# =========================================================
# Propagation model
# =========================================================
def default_propagation_settings():
    """
    Lightweight propagation settings.

    This is intentionally simpler than full atmospheric ray tracing.
    """
    return {
        "temperature_c": 20.0,
        "relative_humidity": 70.0,
        "reference_distance_m": 1.0,
        "enable_atmospheric_absorption": True,
        "enable_ground_reflection": True,
        "ground_reflection_gain": 0.65,
        "ground_lowpass_hz": 1800.0,
        "sound_speed_m_s": None,
    }


def _merge_propagation_settings(user_settings=None):
    settings = default_propagation_settings()
    if user_settings is not None:
        settings.update(user_settings)
    return settings


def _sound_speed_m_s(temperature_c):
    """
    Simple temperature-dependent speed of sound.
    """
    return 331.3 + 0.606 * float(temperature_c)


def _approx_air_absorption_db_per_m(freqs_hz, temperature_c=20.0, relative_humidity=70.0):
    """
    Lightweight frequency-dependent atmospheric absorption model.
    """
    f_khz = np.maximum(np.asarray(freqs_hz, dtype=float) / 1000.0, 1e-9)

    humidity = np.clip(float(relative_humidity), 0.0, 100.0)
    humidity_scale = 0.6 + 0.8 * (humidity / 100.0)
    temperature_scale = 1.0 + 0.003 * (float(temperature_c) - 20.0)

    alpha_db_per_m = 2.0e-4 * humidity_scale * temperature_scale * (f_khz ** 1.7)
    return alpha_db_per_m


def _compute_propagation_state(source_pos, observer_pos, freqs_hz, propagation_settings):
    settings = _merge_propagation_settings(propagation_settings)

    c = settings["sound_speed_m_s"]
    if c is None:
        c = _sound_speed_m_s(settings["temperature_c"])

    ref_dist = max(float(settings["reference_distance_m"]), 1e-6)

    direct_vec = observer_pos - source_pos
    direct_distance = max(_safe_norm(direct_vec), ref_dist)
    direct_delay_s = direct_distance / c

    if settings["enable_atmospheric_absorption"]:
        alpha_db_per_m = _approx_air_absorption_db_per_m(
            freqs_hz,
            temperature_c=settings["temperature_c"],
            relative_humidity=settings["relative_humidity"],
        )
    else:
        alpha_db_per_m = np.zeros_like(freqs_hz, dtype=float)

    direct_spreading_amp = ref_dist / direct_distance
    direct_air_amp = 10.0 ** (-(alpha_db_per_m * direct_distance) / 20.0)
    direct_total_amp = direct_spreading_amp * direct_air_amp

    state = {
        "c_m_s": c,
        "direct_distance_m": direct_distance,
        "direct_delay_s": direct_delay_s,
        "direct_amp_per_band": direct_total_amp,
        "direct_alpha_db_per_m": alpha_db_per_m,
        "reflection_enabled": bool(settings["enable_ground_reflection"]),
        "mirrored_source_pos": None,
        "reflected_distance_m": None,
        "reflected_delay_s": None,
        "reflected_amp_per_band": None,
    }

    if settings["enable_ground_reflection"]:
        mirrored_source = np.array([source_pos[0], source_pos[1], -source_pos[2]], dtype=float)
        reflected_vec = observer_pos - mirrored_source
        reflected_distance = max(_safe_norm(reflected_vec), ref_dist)
        reflected_delay_s = reflected_distance / c

        grazing_factor = np.clip(abs(source_pos[2] + observer_pos[2]) / reflected_distance, 0.0, 1.0)

        base_reflection = float(settings["ground_reflection_gain"]) * (0.5 + 0.5 * grazing_factor)

        lowpass_hz = float(settings["ground_lowpass_hz"])
        if lowpass_hz > 0.0:
            refl_lpf = 1.0 / np.sqrt(1.0 + (np.asarray(freqs_hz, dtype=float) / lowpass_hz) ** 2)
        else:
            refl_lpf = np.ones_like(freqs_hz, dtype=float)

        reflected_spreading_amp = ref_dist / reflected_distance
        reflected_air_amp = 10.0 ** (-(alpha_db_per_m * reflected_distance) / 20.0)

        reflected_total_amp = base_reflection * refl_lpf * reflected_spreading_amp * reflected_air_amp

        state["mirrored_source_pos"] = mirrored_source
        state["reflected_distance_m"] = reflected_distance
        state["reflected_delay_s"] = reflected_delay_s
        state["reflected_amp_per_band"] = reflected_total_amp

    return state


def _estimate_max_delay_seconds(source_positions, observer_positions, apply_propagation, propagation_settings):
    if not apply_propagation:
        return 0.0

    settings = _merge_propagation_settings(propagation_settings)
    c = settings["sound_speed_m_s"]
    if c is None:
        c = _sound_speed_m_s(settings["temperature_c"])

    direct_max = 0.0
    reflected_max = 0.0

    for src, obs in zip(source_positions, observer_positions):
        direct_max = max(direct_max, _safe_norm(obs - src))
        if settings["enable_ground_reflection"]:
            mirrored_src = np.array([src[0], src[1], -src[2]], dtype=float)
            reflected_max = max(reflected_max, _safe_norm(obs - mirrored_src))

    return max(direct_max, reflected_max) / c


def _build_propagation_summary(source_positions, observer_positions, times, apply_propagation, propagation_settings):
    n = len(times)

    summary = {
        "enabled": bool(apply_propagation),
        "settings": _merge_propagation_settings(propagation_settings),
        "direct_distance_m": None,
        "direct_delay_s": None,
        "mirrored_source_positions": None,
        "reflected_distance_m": None,
        "reflected_delay_s": None,
    }

    if not apply_propagation:
        return summary

    settings = _merge_propagation_settings(propagation_settings)

    direct_distances = np.zeros(n, dtype=float)
    direct_delays = np.zeros(n, dtype=float)

    mirrored_positions = None
    reflected_distances = None
    reflected_delays = None

    if settings["enable_ground_reflection"]:
        mirrored_positions = np.zeros_like(source_positions, dtype=float)
        reflected_distances = np.zeros(n, dtype=float)
        reflected_delays = np.zeros(n, dtype=float)

    for i in range(n):
        state = _compute_propagation_state(
            source_positions[i],
            observer_positions[i],
            freqs_hz=np.array([1000.0]),
            propagation_settings=settings,
        )

        direct_distances[i] = state["direct_distance_m"]
        direct_delays[i] = state["direct_delay_s"]

        if settings["enable_ground_reflection"]:
            mirrored_positions[i] = state["mirrored_source_pos"]
            reflected_distances[i] = state["reflected_distance_m"]
            reflected_delays[i] = state["reflected_delay_s"]

    summary["direct_distance_m"] = direct_distances
    summary["direct_delay_s"] = direct_delays
    summary["mirrored_source_positions"] = mirrored_positions
    summary["reflected_distance_m"] = reflected_distances
    summary["reflected_delay_s"] = reflected_delays
    return summary


# =========================================================
# CSV / positions
# =========================================================
def load_spectrogram_csv(csv_path):
    """
    Expected format:
    - first column: frequencies
    - first row (header): times
    """
    df = pd.read_csv(csv_path, index_col=0)

    freqs = np.array([float(v) for v in df.index], dtype=float)
    times = np.array([float(v) for v in df.columns], dtype=float)
    spl_db = df.to_numpy(dtype=float)

    if spl_db.shape != (len(freqs), len(times)):
        raise ValueError(f"Inconsistent CSV shape in file: {csv_path}")

    return freqs, times, spl_db


def get_spectrogram_duration(csv_path):
    _, times, _ = load_spectrogram_csv(csv_path)
    if len(times) == 0:
        return 0.0
    if len(times) == 1:
        return float(times[0])
    return float(times[-1] - times[0])


def get_audio_duration(audio_path, analysis_fs=None):
    y, sr = _load_audio_mono(audio_path, sr=analysis_fs)
    return len(y) / float(sr)


def load_and_interpolate_positions(position_csv_path, target_times):
    """
    Expected columns:
    time, emitter_x, emitter_y, emitter_z, observer_x, observer_y, observer_z
    """
    pos_df = pd.read_csv(position_csv_path)

    required_cols = [
        "time",
        "emitter_x", "emitter_y", "emitter_z",
        "observer_x", "observer_y", "observer_z",
    ]
    for col in required_cols:
        if col not in pos_df.columns:
            raise ValueError(f"Missing required column in positions CSV: {col}")

    t_pos = pos_df["time"].to_numpy(dtype=float)

    if len(t_pos) < 2:
        raise ValueError("Positions CSV must contain at least two rows.")

    interp_pos = {}
    for col in required_cols[1:]:
        interp_pos[col] = interp1d(
            t_pos,
            pos_df[col].to_numpy(dtype=float),
            bounds_error=False,
            fill_value="extrapolate",
        )

    vel_df = pos_df.copy()
    for col in required_cols[1:]:
        vel_df[col] = np.gradient(pos_df[col].to_numpy(dtype=float), t_pos)

    interp_vel = {}
    for col in required_cols[1:]:
        interp_vel[col] = interp1d(
            t_pos,
            vel_df[col].to_numpy(dtype=float),
            bounds_error=False,
            fill_value="extrapolate",
        )

    source_positions = np.column_stack([
        interp_pos["emitter_x"](target_times),
        interp_pos["emitter_y"](target_times),
        interp_pos["emitter_z"](target_times),
    ])

    observer_positions = np.column_stack([
        interp_pos["observer_x"](target_times),
        interp_pos["observer_y"](target_times),
        interp_pos["observer_z"](target_times),
    ])

    source_velocities = np.column_stack([
        interp_vel["emitter_x"](target_times),
        interp_vel["emitter_y"](target_times),
        interp_vel["emitter_z"](target_times),
    ])

    observer_velocities = np.column_stack([
        interp_vel["observer_x"](target_times),
        interp_vel["observer_y"](target_times),
        interp_vel["observer_z"](target_times),
    ])

    return source_positions, observer_positions, source_velocities, observer_velocities


def _prepare_positions(position_csv_path, times):
    n_times = len(times)

    if position_csv_path:
        return load_and_interpolate_positions(position_csv_path, times)

    source_positions = np.zeros((n_times, 3))
    observer_positions = np.zeros((n_times, 3))
    source_velocities = np.zeros((n_times, 3))
    observer_velocities = np.zeros((n_times, 3))
    return source_positions, observer_positions, source_velocities, observer_velocities


# =========================================================
# Sample positions generator
# =========================================================
def generate_sample_positions_csv(output_csv_path, duration_s, scenario="Overfly", n_points=500):
    duration_s = float(duration_s)
    if duration_s <= 0.0:
        raise ValueError("Duration must be positive to generate sample positions.")

    n_points = max(int(n_points), 2)
    t = np.linspace(0.0, duration_s, n_points)
    tau = t / duration_s

    observer_x = np.zeros_like(t)
    observer_y = np.zeros_like(t)
    observer_z = np.full_like(t, 1.7)

    scenario_key = scenario.strip().lower()

    if scenario_key == "takeoff":
        emitter_x = -250.0 + 700.0 * tau
        emitter_y = np.zeros_like(t)
        emitter_z = 2.0 + 260.0 * (tau ** 1.35)

    elif scenario_key == "overfly":
        emitter_x = -700.0 + 1400.0 * tau
        emitter_y = np.zeros_like(t)
        emitter_z = np.full_like(t, 120.0)

    elif scenario_key == "landing":
        emitter_x = -700.0 + 800.0 * tau
        emitter_y = np.zeros_like(t)
        emitter_z = 260.0 * ((1.0 - tau) ** 1.25) + 2.0

    elif scenario_key == "cosine z":
        emitter_x = -600.0 + 1200.0 * tau
        emitter_y = np.zeros_like(t)
        emitter_z = 220.0 - 120.0 * np.cos(2.0 * np.pi * tau)

    elif scenario_key == "wind turbine tip":
        hub_x = 0.0
        hub_y = 0.0
        hub_z = 110.0
        radius = 55.0
        rotation_period_s = 1.0

        theta = 2.0 * np.pi * t / rotation_period_s

        emitter_x = hub_x + radius * np.cos(theta)
        emitter_y = np.full_like(t, hub_y)
        emitter_z = hub_z + radius * np.sin(theta)

    else:
        raise ValueError(f"Unknown sample scenario: {scenario}")

    df = pd.DataFrame({
        "time": t,
        "emitter_x": emitter_x,
        "emitter_y": emitter_y,
        "emitter_z": emitter_z,
        "observer_x": observer_x,
        "observer_y": observer_y,
        "observer_z": observer_z,
    })

    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    return output_csv_path


# =========================================================
# Spectrogram-based synthesis
# =========================================================
def _synthesize_signal_from_spl(
    freqs,
    times,
    spl_db,
    fs,
    fft_block_size,
    source_positions,
    observer_positions,
    source_velocities,
    observer_velocities,
    apply_doppler=True,
    apply_propagation=False,
    propagation_settings=None,
    random_seed_offset=0,
):
    """
    Synthesize a signal directly from a spectrogram-like SPL grid.

    Optional propagation model:
    - travel-time delay
    - geometric spreading
    - atmospheric absorption
    - optional flat-ground reflection
    """
    hop_size = fft_block_size // 4
    n_blocks = spl_db.shape[1]

    max_delay_s = _estimate_max_delay_seconds(
        source_positions,
        observer_positions,
        apply_propagation=apply_propagation,
        propagation_settings=propagation_settings,
    )
    max_delay_samples = int(np.ceil(max_delay_s * fs))

    output_length = hop_size * (n_blocks - 1) + fft_block_size + max_delay_samples + 2

    signal = np.zeros(output_length, dtype=float)
    window = hann(fft_block_size, sym=False)

    df_bin = fs / fft_block_size
    n_positive_bins = fft_block_size // 2 + 1

    for i in range(n_blocks):
        direct_interp_ampl = np.zeros(n_positive_bins, dtype=float)
        reflected_interp_ampl = np.zeros(n_positive_bins, dtype=float)

        band_ampl = convert_db_to_amplitude(spl_db[:, i])

        if apply_doppler:
            doppler_factor = compute_doppler_shift(
                source_positions[i],
                observer_positions[i],
                source_velocities[i],
                observer_velocities[i],
            )

            amplitude_factor = compute_convective_amplification(
                source_positions[i],
                observer_positions[i],
                source_velocities[i],
            )

            freqs_used = freqs * doppler_factor
            band_ampl = band_ampl * amplitude_factor
        else:
            freqs_used = freqs

        if apply_propagation:
            prop_state = _compute_propagation_state(
                source_positions[i],
                observer_positions[i],
                freqs_used,
                propagation_settings,
            )
            direct_delay_samples = int(round(prop_state["direct_delay_s"] * fs))
            reflected_delay_samples = (
                int(round(prop_state["reflected_delay_s"] * fs))
                if prop_state["reflection_enabled"] and prop_state["reflected_delay_s"] is not None
                else None
            )
        else:
            prop_state = None
            direct_delay_samples = 0
            reflected_delay_samples = None

        for f_idx, f_center in enumerate(freqs_used):
            bin_idx = int(np.round(f_center / df_bin))
            if 0 <= bin_idx < n_positive_bins:
                if apply_propagation:
                    direct_interp_ampl[bin_idx] += band_ampl[f_idx] * prop_state["direct_amp_per_band"][f_idx]

                    if prop_state["reflection_enabled"] and prop_state["reflected_amp_per_band"] is not None:
                        reflected_interp_ampl[bin_idx] += (
                            band_ampl[f_idx] * prop_state["reflected_amp_per_band"][f_idx]
                        )
                else:
                    direct_interp_ampl[bin_idx] += band_ampl[f_idx]

        rng = np.random.default_rng(seed=i + random_seed_offset)
        phase = np.exp(1j * 2.0 * np.pi * rng.random(n_positive_bins))
        phase[0] = 1.0 + 0j
        if n_positive_bins > 1:
            phase[-1] = 1.0 + 0j

        direct_spectrum = direct_interp_ampl * phase
        direct_spectrum_full = np.concatenate([direct_spectrum, np.conj(direct_spectrum[-2:0:-1])])
        direct_block = np.real(ifft(direct_spectrum_full))
        direct_block *= window

        direct_start = i * hop_size + direct_delay_samples
        direct_end = direct_start + fft_block_size
        signal[direct_start:direct_end] += direct_block

        if apply_propagation and prop_state["reflection_enabled"] and reflected_delay_samples is not None:
            reflected_spectrum = reflected_interp_ampl * phase
            reflected_spectrum_full = np.concatenate([reflected_spectrum, np.conj(reflected_spectrum[-2:0:-1])])
            reflected_block = np.real(ifft(reflected_spectrum_full))
            reflected_block *= window

            reflected_start = i * hop_size + reflected_delay_samples
            reflected_end = reflected_start + fft_block_size
            signal[reflected_start:reflected_end] += reflected_block

    return signal


# =========================================================
# Audio-through-trajectory synthesis
# =========================================================
def _synthesize_moving_audio_source(
    source_signal,
    fs,
    block_times,
    source_positions,
    observer_positions,
    source_velocities,
    observer_velocities,
    fft_block_size=2048,
    apply_doppler=True,
    apply_propagation=False,
    propagation_settings=None,
):
    """
    Treat an existing time-domain audio file as the emitted source signal and
    auralize it along a moving trajectory.
    """
    hop_size = fft_block_size // 4
    window = hann(fft_block_size, sym=False)
    local_t = (np.arange(fft_block_size) - fft_block_size / 2) / float(fs)

    source_signal = np.asarray(source_signal, dtype=float)
    source_time = np.arange(len(source_signal)) / float(fs)

    max_delay_s = _estimate_max_delay_seconds(
        source_positions,
        observer_positions,
        apply_propagation=apply_propagation,
        propagation_settings=propagation_settings,
    )
    max_delay_samples = int(np.ceil(max_delay_s * fs))

    output_length = len(source_signal) + max_delay_samples + fft_block_size + 2
    output = np.zeros(output_length, dtype=float)
    weight = np.zeros(output_length, dtype=float)

    sample_freqs_for_prop = np.array([1000.0], dtype=float)

    for i, t_obs in enumerate(block_times):
        src_pos = source_positions[i]
        obs_pos = observer_positions[i]
        src_vel = source_velocities[i]
        obs_vel = observer_velocities[i]

        if apply_doppler:
            doppler_factor = compute_doppler_shift(src_pos, obs_pos, src_vel, obs_vel)
            conv_amp = compute_convective_amplification(src_pos, obs_pos, src_vel)
        else:
            doppler_factor = 1.0
            conv_amp = 1.0

        if apply_propagation:
            prop_state = _compute_propagation_state(
                src_pos,
                obs_pos,
                freqs_hz=sample_freqs_for_prop,
                propagation_settings=propagation_settings,
            )
            direct_delay_s = float(prop_state["direct_delay_s"])
            direct_amp = float(prop_state["direct_amp_per_band"][0])

            if prop_state["reflection_enabled"] and prop_state["reflected_delay_s"] is not None:
                reflected_delay_s = float(prop_state["reflected_delay_s"])
                reflected_amp = float(prop_state["reflected_amp_per_band"][0])
            else:
                reflected_delay_s = None
                reflected_amp = None
        else:
            direct_delay_s = 0.0
            direct_amp = 1.0
            reflected_delay_s = None
            reflected_amp = None

        start_out = i * hop_size
        end_out = start_out + fft_block_size

        direct_emission_center = t_obs - direct_delay_s
        direct_sample_times = direct_emission_center + local_t * doppler_factor
        direct_block = np.interp(
            direct_sample_times,
            source_time,
            source_signal,
            left=0.0,
            right=0.0,
        )
        direct_block = direct_block * (conv_amp * direct_amp) * window

        output[start_out:end_out] += direct_block
        weight[start_out:end_out] += window ** 2

        if reflected_delay_s is not None and reflected_amp is not None and reflected_amp > 0.0:
            reflected_emission_center = t_obs - reflected_delay_s
            reflected_sample_times = reflected_emission_center + local_t * doppler_factor
            reflected_block = np.interp(
                reflected_sample_times,
                source_time,
                source_signal,
                left=0.0,
                right=0.0,
            )
            reflected_block = reflected_block * (conv_amp * reflected_amp) * window

            output[start_out:end_out] += reflected_block
            weight[start_out:end_out] += window ** 2

    valid = weight > 1e-12
    output[valid] /= weight[valid]
    output[~valid] = 0.0

    return output


# =========================================================
# Spectrogram-input API
# =========================================================
def auralize_from_csv(
    csv_path,
    position_csv_path=None,
    fs=44100,
    fft_block_size=2048,
    apply_doppler=True,
    apply_propagation=False,
    propagation_settings=None,
    output_wav_path="combined_output.wav",
):
    """
    Combined spectrogram input mode.
    """
    freqs, times, spl_db = load_spectrogram_csv(csv_path)

    (
        source_positions,
        observer_positions,
        source_velocities,
        observer_velocities,
    ) = _prepare_positions(position_csv_path, times)

    raw_signal = _synthesize_signal_from_spl(
        freqs=freqs,
        times=times,
        spl_db=spl_db,
        fs=fs,
        fft_block_size=fft_block_size,
        source_positions=source_positions,
        observer_positions=observer_positions,
        source_velocities=source_velocities,
        observer_velocities=observer_velocities,
        apply_doppler=apply_doppler,
        apply_propagation=apply_propagation,
        propagation_settings=propagation_settings,
        random_seed_offset=0,
    )

    max_abs = np.max(np.abs(raw_signal))
    signal = _normalize_signal(raw_signal, max_abs)
    _write_wav(output_wav_path, fs, signal)

    propagation_summary = _build_propagation_summary(
        source_positions,
        observer_positions,
        times,
        apply_propagation=apply_propagation,
        propagation_settings=propagation_settings,
    )

    result = {
        "mode": "combined_input",
        "times": times,
        "freqs": freqs,
        "combined_input_db": spl_db,
        "source_positions": source_positions,
        "observer_positions": observer_positions,
        "apply_doppler": apply_doppler,
        "apply_propagation": apply_propagation,
        "propagation": propagation_summary,
        "duration_seconds": len(signal) / fs,
    }

    return signal, fs, output_wav_path, result


def auralize_from_separate_csv(
    broadband_csv_path,
    tonal_csv_path,
    position_csv_path=None,
    fs=44100,
    fft_block_size=2048,
    apply_doppler=True,
    apply_propagation=False,
    propagation_settings=None,
    output_dir=".",
):
    """
    Separate broadband + tonal input mode.
    """
    bb_freqs, bb_times, broadband_db = load_spectrogram_csv(broadband_csv_path)
    tn_freqs, tn_times, tonal_db = load_spectrogram_csv(tonal_csv_path)

    if len(bb_freqs) != len(tn_freqs) or not np.allclose(bb_freqs, tn_freqs):
        raise ValueError("Broadband and tonal CSV files must have the same frequency axis.")

    if len(bb_times) != len(tn_times) or not np.allclose(bb_times, tn_times):
        raise ValueError("Broadband and tonal CSV files must have the same time axis.")

    freqs = bb_freqs
    times = bb_times
    combined_input_db = _power_sum_db(broadband_db, tonal_db)

    (
        source_positions,
        observer_positions,
        source_velocities,
        observer_velocities,
    ) = _prepare_positions(position_csv_path, times)

    broadband_raw = _synthesize_signal_from_spl(
        freqs=freqs,
        times=times,
        spl_db=broadband_db,
        fs=fs,
        fft_block_size=fft_block_size,
        source_positions=source_positions,
        observer_positions=observer_positions,
        source_velocities=source_velocities,
        observer_velocities=observer_velocities,
        apply_doppler=apply_doppler,
        apply_propagation=apply_propagation,
        propagation_settings=propagation_settings,
        random_seed_offset=1000,
    )

    tonal_raw = _synthesize_signal_from_spl(
        freqs=freqs,
        times=times,
        spl_db=tonal_db,
        fs=fs,
        fft_block_size=fft_block_size,
        source_positions=source_positions,
        observer_positions=observer_positions,
        source_velocities=source_velocities,
        observer_velocities=observer_velocities,
        apply_doppler=apply_doppler,
        apply_propagation=apply_propagation,
        propagation_settings=propagation_settings,
        random_seed_offset=2000,
    )

    combined_raw = broadband_raw + tonal_raw

    global_scale = max(
        np.max(np.abs(broadband_raw)),
        np.max(np.abs(tonal_raw)),
        np.max(np.abs(combined_raw)),
        1e-12,
    )

    broadband_signal = _normalize_signal(broadband_raw, global_scale)
    tonal_signal = _normalize_signal(tonal_raw, global_scale)
    combined_signal = _normalize_signal(combined_raw, global_scale)

    os.makedirs(output_dir, exist_ok=True)
    broadband_wav = os.path.join(output_dir, "broadband_output.wav")
    tonal_wav = os.path.join(output_dir, "tonal_output.wav")
    combined_wav = os.path.join(output_dir, "combined_output.wav")

    _write_wav(broadband_wav, fs, broadband_signal)
    _write_wav(tonal_wav, fs, tonal_signal)
    _write_wav(combined_wav, fs, combined_signal)

    propagation_summary = _build_propagation_summary(
        source_positions,
        observer_positions,
        times,
        apply_propagation=apply_propagation,
        propagation_settings=propagation_settings,
    )

    signals = {
        "broadband": broadband_signal,
        "tonal": tonal_signal,
        "combined": combined_signal,
    }

    wav_paths = {
        "broadband": broadband_wav,
        "tonal": tonal_wav,
        "combined": combined_wav,
    }

    result = {
        "mode": "separate_input",
        "times": times,
        "freqs": freqs,
        "broadband_db": broadband_db,
        "tonal_db": tonal_db,
        "combined_input_db": combined_input_db,
        "source_positions": source_positions,
        "observer_positions": observer_positions,
        "apply_doppler": apply_doppler,
        "apply_propagation": apply_propagation,
        "propagation": propagation_summary,
        "duration_seconds": len(combined_signal) / fs,
    }

    return signals, fs, wav_paths, result


def auralize_audio_file_with_trajectory(
    audio_path,
    position_csv_path=None,
    analysis_fs=None,
    fft_block_size=2048,
    apply_doppler=True,
    apply_propagation=False,
    propagation_settings=None,
    output_dir=".",
):
    """
    Use an existing WAV/MP3 as the emitted source signal and play it through
    a moving source/observer trajectory.
    """
    y, sr = _load_audio_mono(audio_path, sr=analysis_fs)

    hop_size = fft_block_size // 4
    if hop_size <= 0:
        raise ValueError("FFT block size must be positive and large enough.")

    n_blocks = max(1, int(np.ceil(max(len(y) - fft_block_size, 0) / hop_size)) + 1)
    block_times = (np.arange(n_blocks) * hop_size + fft_block_size / 2) / float(sr)

    (
        source_positions,
        observer_positions,
        source_velocities,
        observer_velocities,
    ) = _prepare_positions(position_csv_path, block_times)

    propagated_raw = _synthesize_moving_audio_source(
        source_signal=y,
        fs=sr,
        block_times=block_times,
        source_positions=source_positions,
        observer_positions=observer_positions,
        source_velocities=source_velocities,
        observer_velocities=observer_velocities,
        fft_block_size=fft_block_size,
        apply_doppler=apply_doppler,
        apply_propagation=apply_propagation,
        propagation_settings=propagation_settings,
    )

    global_scale = max(
        np.max(np.abs(y)),
        np.max(np.abs(propagated_raw)),
        1e-12,
    )

    original_signal = _normalize_signal(y, global_scale)
    propagated_signal = _normalize_signal(propagated_raw, global_scale)

    os.makedirs(output_dir, exist_ok=True)
    original_wav = os.path.join(output_dir, "audio_original.wav")
    trajectory_wav = os.path.join(output_dir, "audio_trajectory_output.wav")

    _write_wav(original_wav, sr, original_signal)
    _write_wav(trajectory_wav, sr, propagated_signal)

    propagation_summary = _build_propagation_summary(
        source_positions,
        observer_positions,
        block_times,
        apply_propagation=apply_propagation,
        propagation_settings=propagation_settings,
    )

    signals = {
        "original": original_signal,
        "trajectory": propagated_signal,
    }

    wav_paths = {
        "original": original_wav,
        "trajectory": trajectory_wav,
    }

    result = {
        "mode": "audio_trajectory",
        "times": block_times,
        "source_positions": source_positions,
        "observer_positions": observer_positions,
        "apply_doppler": apply_doppler,
        "apply_propagation": apply_propagation,
        "propagation": propagation_summary,
        "duration_seconds": len(propagated_signal) / float(sr),
        "analysis_fs": sr,
        "fft_block_size": fft_block_size,
    }

    return signals, sr, wav_paths, result


# =========================================================
# Audio-input learning module
# =========================================================
def _amplitude_to_db_relative(amplitude):
    amplitude = np.maximum(amplitude, 1e-12)
    ref = np.max(amplitude)
    if ref < 1e-12:
        ref = 1.0
    return 20.0 * np.log10(amplitude / ref)


def analyze_audio_input(
    audio_path,
    analysis_fs=None,
    fft_block_size=2048,
    griffinlim_iterations=32,
    output_dir=".",
):
    """
    WAV / MP3 learning module.
    """
    librosa = _require_librosa()

    if fft_block_size <= 0:
        raise ValueError("FFT block size must be positive.")

    if griffinlim_iterations <= 0:
        raise ValueError("Griffin-Lim iterations must be positive.")

    hop_length = fft_block_size // 4
    if hop_length <= 0:
        raise ValueError("FFT block size is too small.")

    y, sr = librosa.load(audio_path, sr=analysis_fs, mono=True)

    if y.size == 0:
        raise ValueError("Loaded audio is empty.")

    stft_matrix = librosa.stft(
        y,
        n_fft=fft_block_size,
        hop_length=hop_length,
        win_length=fft_block_size,
        window="hann",
        center=True,
    )

    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)

    y_phase_reconstructed = librosa.istft(
        stft_matrix,
        hop_length=hop_length,
        win_length=fft_block_size,
        window="hann",
        center=True,
        length=len(y),
    )

    y_magnitude_only = librosa.griffinlim(
        magnitude,
        n_iter=griffinlim_iterations,
        hop_length=hop_length,
        win_length=fft_block_size,
        window="hann",
        center=True,
        length=len(y),
        init="random",
        random_state=0,
    )

    global_scale = max(
        np.max(np.abs(y)),
        np.max(np.abs(y_phase_reconstructed)),
        np.max(np.abs(y_magnitude_only)),
        1e-12,
    )

    original_signal = _normalize_signal(y, global_scale)
    phase_reconstructed_signal = _normalize_signal(y_phase_reconstructed, global_scale)
    magnitude_only_signal = _normalize_signal(y_magnitude_only, global_scale)

    os.makedirs(output_dir, exist_ok=True)
    original_wav = os.path.join(output_dir, "audio_original.wav")
    phase_reconstructed_wav = os.path.join(output_dir, "audio_phase_reconstructed.wav")
    magnitude_only_wav = os.path.join(output_dir, "audio_magnitude_only.wav")

    _write_wav(original_wav, sr, original_signal)
    _write_wav(phase_reconstructed_wav, sr, phase_reconstructed_signal)
    _write_wav(magnitude_only_wav, sr, magnitude_only_signal)

    magnitude_db = _amplitude_to_db_relative(magnitude)

    magonly_stft = librosa.stft(
        magnitude_only_signal,
        n_fft=fft_block_size,
        hop_length=hop_length,
        win_length=fft_block_size,
        window="hann",
        center=True,
    )
    griffinlim_db = _amplitude_to_db_relative(np.abs(magonly_stft))

    freqs = librosa.fft_frequencies(sr=sr, n_fft=fft_block_size)
    times = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=sr, hop_length=hop_length)
    wave_time = np.arange(len(original_signal)) / float(sr)

    signals = {
        "original": original_signal,
        "phase_reconstructed": phase_reconstructed_signal,
        "magnitude_only": magnitude_only_signal,
    }

    wav_paths = {
        "original": original_wav,
        "phase_reconstructed": phase_reconstructed_wav,
        "magnitude_only": magnitude_only_wav,
    }

    result = {
        "mode": "audio_input",
        "wave_time": wave_time,
        "original_signal": original_signal,
        "phase_reconstructed_signal": phase_reconstructed_signal,
        "magnitude_only_signal": magnitude_only_signal,
        "times": times,
        "freqs": freqs,
        "magnitude_db": magnitude_db,
        "phase_rad": phase,
        "griffinlim_db": griffinlim_db,
        "duration_seconds": len(original_signal) / float(sr),
        "analysis_fs": sr,
        "fft_block_size": fft_block_size,
        "griffinlim_iterations": griffinlim_iterations,
    }

    return signals, sr, wav_paths, result