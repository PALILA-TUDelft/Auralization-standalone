import os
import time
import shutil
import traceback
import threading
import subprocess
import platform
import numpy as np
import customtkinter as ctk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from auralization import (
    auralize_from_csv,
    auralize_from_separate_csv,
    analyze_audio_input,
    auralize_audio_file_with_trajectory,
    get_spectrogram_duration,
    get_audio_duration,
    generate_sample_positions_csv,
    SAMPLE_TRAJECTORY_OPTIONS,
)


# ---------------------------------
# CustomTkinter setup
# ---------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Auralization Framework")
root.geometry("1500x940")
root.minsize(1250, 800)


# ---------------------------------
# Global state
# ---------------------------------
content_frame = None
message_label = None

dashboard_canvas = None
dashboard_figure = None
dashboard_handles = {}

current_result = None
current_signals = {}
current_fs = None
current_wav_paths = {}
current_play_key = "combined"
current_play_duration = 0.0

is_processing = False
is_playing = False
playback_start_time = None
audio_process = None


# ---------------------------------
# Helpers
# ---------------------------------
def update_message(msg, color="white"):
    global message_label
    if message_label is not None and message_label.winfo_exists():
        message_label.configure(text=msg, text_color=color)


def browse_file(entry_widget, filetypes):
    path = filedialog.askopenfilename(filetypes=filetypes)
    if path:
        entry_widget.delete(0, "end")
        entry_widget.insert(0, path)
        update_message(f"Loaded: {os.path.basename(path)}", "lightblue")


def copy_file(src, default_ext):
    if not src or not os.path.isfile(src):
        update_message("File not available.", "orange")
        return

    save_path = filedialog.asksaveasfilename(defaultextension=default_ext)
    if save_path:
        shutil.copy(src, save_path)
        update_message(f"Saved file to: {save_path}", "lightgreen")


def save_current_figure():
    global dashboard_figure
    if dashboard_figure is None:
        update_message("No figure available.", "orange")
        return

    save_path = filedialog.asksaveasfilename(defaultextension=".png")
    if save_path:
        dashboard_figure.savefig(save_path, dpi=200, bbox_inches="tight")
        update_message(f"Saved figure to: {save_path}", "lightgreen")


def clear_content():
    global content_frame
    stop_audio(reset_cursor=True)
    if content_frame is not None and content_frame.winfo_exists():
        content_frame.destroy()
    content_frame = ctk.CTkFrame(root, corner_radius=0)
    content_frame.pack(fill="both", expand=True, padx=12, pady=12)


def reset_dashboard():
    global dashboard_canvas, dashboard_figure, dashboard_handles
    dashboard_canvas = None
    dashboard_figure = None
    dashboard_handles = {}


def show_help():
    help_window = ctk.CTkToplevel(root)
    help_window.title("Help")
    help_window.geometry("680x500")

    text = (
        "Auralization Framework\n\n"
        "This interface supports:\n"
        "1. Combined spectrogram input\n"
        "2. Separate broadband + tonal spectrogram input\n"
        "3. WAV / MP3 input for STFT-based learning and reconstruction\n"
        "4. WAV / MP3 input as a moving emitted source along a trajectory\n\n"
        "You can also generate sample position CSV files automatically:\n"
        "- Takeoff\n"
        "- Overfly\n"
        "- Landing\n"
        "- Cosine Z\n"
        "- Wind Turbine Tip\n\n"
        "Playback shows a moving vertical cursor through the plots.\n"
        "For trajectory-based modes, the source/observer motion is also animated."
    )

    label = ctk.CTkLabel(help_window, text=text, justify="left", wraplength=620)
    label.pack(padx=22, pady=22, anchor="w")


def _safe_get_audio_fs_value(fs_var):
    fs_text = fs_var.get().strip()
    if fs_text.lower() == "original":
        return None
    return int(fs_text)


def generate_sample_positions_for_entry(entry_widget, duration_seconds, scenario_name):
    if duration_seconds <= 0.0:
        update_message("Could not determine a valid duration for the sample trajectory.", "red")
        return

    suggested_name = f"{scenario_name.lower().replace(' ', '_')}_positions.csv"
    save_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        initialfile=suggested_name,
        filetypes=[("CSV files", "*.csv")],
    )

    if not save_path:
        return

    try:
        generate_sample_positions_csv(save_path, duration_seconds, scenario=scenario_name)
        entry_widget.delete(0, "end")
        entry_widget.insert(0, save_path)
        update_message(
            f"Generated sample trajectory: {scenario_name} ({duration_seconds:.2f} s)",
            "lightgreen",
        )
    except Exception as e:
        traceback.print_exc()
        update_message(f"Could not generate sample trajectory: {e}", "red")


# ---------------------------------
# Playback cursor helpers
# ---------------------------------
def _add_time_cursor(ax, initial_time=0.0, color="white", linewidth=1.6, linestyle="-"):
    line = ax.axvline(initial_time, color=color, linewidth=linewidth, linestyle=linestyle, alpha=0.95)
    return line


def _set_cursor_position(line, t):
    if line is None:
        return
    line.set_xdata([t, t])


def _add_waveform_cursor(ax, initial_time=0.0):
    return _add_time_cursor(ax, initial_time=initial_time, color="white", linewidth=1.6, linestyle="-")


def _add_spectrogram_cursor(ax, initial_time=0.0):
    return _add_time_cursor(ax, initial_time=initial_time, color="white", linewidth=1.6, linestyle="-")


def update_dashboard_playback(elapsed_time):
    global dashboard_canvas, dashboard_handles

    if not dashboard_handles:
        return

    playback_lines = dashboard_handles.get("playback_lines", [])
    for line in playback_lines:
        _set_cursor_position(line, elapsed_time)

    required = ["times", "source_positions", "observer_positions"]
    if all(key in dashboard_handles for key in required):
        times = dashboard_handles["times"]
        source_positions = dashboard_handles["source_positions"]
        observer_positions = dashboard_handles["observer_positions"]
        mirrored_source_positions = dashboard_handles.get("mirrored_source_positions", None)

        idx = np.searchsorted(times, elapsed_time, side="right") - 1
        idx = max(0, min(idx, len(times) - 1))

        sx, sy, sz = source_positions[idx]
        ox, oy, oz = observer_positions[idx]

        dashboard_handles["src_marker"]._offsets3d = ([sx], [sy], [sz])
        dashboard_handles["obs_marker"]._offsets3d = ([ox], [oy], [oz])

        if dashboard_handles.get("direct_line") is not None:
            dashboard_handles["direct_line"].set_data_3d(
                [sx, ox],
                [sy, oy],
                [sz, oz],
            )

        if mirrored_source_positions is not None:
            mx, my, mz = mirrored_source_positions[idx]

            if dashboard_handles.get("mirrored_marker") is not None:
                dashboard_handles["mirrored_marker"]._offsets3d = ([mx], [my], [mz])

            if dashboard_handles.get("reflected_line") is not None:
                dashboard_handles["reflected_line"].set_data_3d(
                    [mx, ox],
                    [my, oy],
                    [mz, oz],
                )

        if dashboard_handles.get("time_text") is not None:
            dashboard_handles["time_text"].set_text(f"t = {times[idx]:.2f} s")

    if dashboard_canvas is not None:
        dashboard_canvas.draw_idle()


# ---------------------------------
# Audio playback
# ---------------------------------
def stop_audio(reset_cursor=True):
    global is_playing, audio_process
    is_playing = False

    if audio_process is not None:
        try:
            if audio_process.poll() is None:
                audio_process.terminate()
        except Exception:
            pass

    audio_process = None

    if reset_cursor:
        update_dashboard_playback(0.0)


def _play_file(path):
    global audio_process

    if platform.system() == "Darwin":
        audio_process = subprocess.Popen(["afplay", path])
    elif platform.system() == "Windows":
        os.startfile(path)
        audio_process = None
    else:
        audio_process = subprocess.Popen(["xdg-open", path])


def _get_play_duration(play_key):
    global current_signals, current_fs, current_result

    if current_fs is not None and play_key in current_signals:
        signal = current_signals[play_key]
        if signal is not None and len(signal) > 0:
            return len(signal) / float(current_fs)

    if current_result is not None:
        return float(current_result.get("duration_seconds", 0.0))

    return 0.0


def play_audio(play_key="combined"):
    global is_playing, playback_start_time, current_wav_paths, current_result
    global current_play_key, current_play_duration

    if play_key not in current_wav_paths:
        update_message(f"No {play_key} audio available.", "orange")
        return

    wav_path = current_wav_paths[play_key]
    if not wav_path or not os.path.isfile(wav_path):
        update_message(f"No {play_key} WAV available.", "orange")
        return

    stop_audio(reset_cursor=True)

    try:
        _play_file(wav_path)
        current_play_key = play_key
        current_play_duration = _get_play_duration(play_key)
        is_playing = True
        playback_start_time = time.monotonic()
        update_message(f"Playing {play_key} audio...", "lightblue")
        update_dashboard_playback(0.0)
        animate_playback()
    except Exception as e:
        traceback.print_exc()
        update_message(f"Playback error: {e}", "red")


def restart_audio():
    play_key = current_play_key if current_play_key in current_wav_paths else "combined"
    stop_audio(reset_cursor=True)
    play_audio(play_key)


def animate_playback():
    global is_playing, playback_start_time, current_play_duration, audio_process

    if not is_playing:
        return

    elapsed = time.monotonic() - playback_start_time
    duration = max(float(current_play_duration), 0.0)

    update_dashboard_playback(elapsed)

    process_finished = (audio_process is not None and audio_process.poll() is not None)

    if duration > 0.0 and (elapsed >= duration or process_finished):
        is_playing = False
        update_dashboard_playback(duration)
        update_message("Playback finished.", "white")
        return

    root.after(50, animate_playback)


# ---------------------------------
# Plotting
# ---------------------------------
def _build_motion_plot(ax_motion, result, current_time_idx=0):
    source_positions = result["source_positions"]
    observer_positions = result["observer_positions"]
    times = result["times"]

    ax_motion.plot(
        source_positions[:, 0],
        source_positions[:, 1],
        source_positions[:, 2],
        "--",
        linewidth=1.2,
        label="Source path",
    )
    ax_motion.plot(
        observer_positions[:, 0],
        observer_positions[:, 1],
        observer_positions[:, 2],
        "--",
        linewidth=1.2,
        label="Observer path",
    )

    src_marker = ax_motion.scatter(
        [source_positions[current_time_idx, 0]],
        [source_positions[current_time_idx, 1]],
        [source_positions[current_time_idx, 2]],
        s=60,
        label="Source",
    )
    obs_marker = ax_motion.scatter(
        [observer_positions[current_time_idx, 0]],
        [observer_positions[current_time_idx, 1]],
        [observer_positions[current_time_idx, 2]],
        s=60,
        label="Observer",
    )

    direct_line, = ax_motion.plot(
        [source_positions[current_time_idx, 0], observer_positions[current_time_idx, 0]],
        [source_positions[current_time_idx, 1], observer_positions[current_time_idx, 1]],
        [source_positions[current_time_idx, 2], observer_positions[current_time_idx, 2]],
        linestyle=":",
        linewidth=1.5,
        label="Direct path",
    )

    mirrored_marker = None
    reflected_line = None
    mirrored_source_positions = None

    if result.get("apply_propagation", False):
        prop = result.get("propagation", {})
        mirrored_source_positions = prop.get("mirrored_source_positions", None)

        if mirrored_source_positions is not None:
            ax_motion.plot(
                mirrored_source_positions[:, 0],
                mirrored_source_positions[:, 1],
                mirrored_source_positions[:, 2],
                ":",
                linewidth=1.0,
                alpha=0.8,
                label="Mirrored source",
            )

            mirrored_marker = ax_motion.scatter(
                [mirrored_source_positions[current_time_idx, 0]],
                [mirrored_source_positions[current_time_idx, 1]],
                [mirrored_source_positions[current_time_idx, 2]],
                s=40,
                marker="x",
                label="Image source",
            )

            reflected_line, = ax_motion.plot(
                [mirrored_source_positions[current_time_idx, 0], observer_positions[current_time_idx, 0]],
                [mirrored_source_positions[current_time_idx, 1], observer_positions[current_time_idx, 1]],
                [mirrored_source_positions[current_time_idx, 2], observer_positions[current_time_idx, 2]],
                linestyle="--",
                linewidth=1.3,
                label="Reflected path",
            )

    all_points = np.vstack([source_positions, observer_positions])
    if mirrored_source_positions is not None:
        all_points = np.vstack([all_points, mirrored_source_positions])

    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = max(np.max(maxs - mins), 10.0)
    pad = 0.2 * span

    ax_motion.set_xlim(center[0] - span / 2 - pad, center[0] + span / 2 + pad)
    ax_motion.set_ylim(center[1] - span / 2 - pad, center[1] + span / 2 + pad)
    ax_motion.set_zlim(center[2] - span / 2 - pad, center[2] + span / 2 + pad)

    ax_motion.set_title("Source / Observer Motion")
    ax_motion.set_xlabel("x [m]")
    ax_motion.set_ylabel("y [m]")
    ax_motion.set_zlabel("z [m]")
    ax_motion.legend(
        loc="center left",
        bbox_to_anchor=(1.3, 0.7),
        borderaxespad=0.0,
        frameon=True,
    )

    time_text = ax_motion.text2D(
        0.03,
        0.95,
        f"t = {times[current_time_idx]:.2f} s",
        transform=ax_motion.transAxes,
    )

    handles = {
        "motion_axis": ax_motion,
        "src_marker": src_marker,
        "obs_marker": obs_marker,
        "direct_line": direct_line,
        "mirrored_marker": mirrored_marker,
        "reflected_line": reflected_line,
        "time_text": time_text,
        "times": times,
        "source_positions": source_positions,
        "observer_positions": observer_positions,
        "mirrored_source_positions": mirrored_source_positions,
    }
    return handles


def build_combined_dashboard_figure(signal, fs, result, current_time_idx=0):
    fig = Figure(figsize=(9.5, 8.5), dpi=100, constrained_layout=True)

    ax_motion = fig.add_subplot(2, 1, 1, projection="3d")
    ax_combined = fig.add_subplot(2, 1, 2)

    motion_handles = _build_motion_plot(ax_motion, result, current_time_idx=current_time_idx)

    _, _, _, im = ax_combined.specgram(
        signal,
        NFFT=1024,
        Fs=fs,
        noverlap=768,
        cmap="viridis",
    )
    ax_combined.set_title("Combined Spectrogram")
    ax_combined.set_xlabel("Time [s]")
    ax_combined.set_ylabel("Frequency [Hz]")
    cbar = fig.colorbar(im, ax=ax_combined)
    cbar.set_label("Amplitude [dB]")

    playback_lines = [
        _add_spectrogram_cursor(ax_combined, initial_time=0.0),
    ]

    handles = {
        "figure": fig,
        "playback_lines": playback_lines,
        **motion_handles,
    }
    return fig, handles


def build_separate_dashboard_figure(combined_signal, fs, result, current_time_idx=0):
    fig = Figure(figsize=(12.5, 8.5), dpi=100, constrained_layout=True)

    ax_motion = fig.add_subplot(2, 2, 1, projection="3d")
    ax_combined = fig.add_subplot(2, 2, 2)
    ax_broadband = fig.add_subplot(2, 2, 3)
    ax_tonal = fig.add_subplot(2, 2, 4)

    motion_handles = _build_motion_plot(ax_motion, result, current_time_idx=current_time_idx)

    _, _, _, im1 = ax_combined.specgram(
        combined_signal,
        NFFT=1024,
        Fs=fs,
        noverlap=768,
        cmap="viridis",
    )
    ax_combined.set_title("Combined Spectrogram")
    ax_combined.set_xlabel("Time [s]")
    ax_combined.set_ylabel("Frequency [Hz]")
    cbar1 = fig.colorbar(im1, ax=ax_combined)
    cbar1.set_label("Amplitude [dB]")

    times = result["times"]
    freqs = result["freqs"]
    broadband_db = result["broadband_db"]
    tonal_db = result["tonal_db"]

    im2 = ax_broadband.pcolormesh(
        times,
        freqs,
        broadband_db,
        shading="auto",
        cmap="viridis",
    )
    ax_broadband.set_yscale("log")
    ax_broadband.set_title("Broadband Spectrogram")
    ax_broadband.set_xlabel("Time [s]")
    ax_broadband.set_ylabel("Frequency [Hz]")
    cbar2 = fig.colorbar(im2, ax=ax_broadband)
    cbar2.set_label("SPL [dB]")

    im3 = ax_tonal.pcolormesh(
        times,
        freqs,
        tonal_db,
        shading="auto",
        cmap="viridis",
    )
    ax_tonal.set_yscale("log")
    ax_tonal.set_title("Tonal Spectrogram")
    ax_tonal.set_xlabel("Time [s]")
    ax_tonal.set_ylabel("Frequency [Hz]")
    cbar3 = fig.colorbar(im3, ax=ax_tonal)
    cbar3.set_label("SPL [dB]")

    playback_lines = [
        _add_spectrogram_cursor(ax_combined, initial_time=0.0),
        _add_spectrogram_cursor(ax_broadband, initial_time=0.0),
        _add_spectrogram_cursor(ax_tonal, initial_time=0.0),
    ]

    handles = {
        "figure": fig,
        "playback_lines": playback_lines,
        **motion_handles,
    }
    return fig, handles


def build_audio_learning_figure(result):
    fig = Figure(figsize=(12.5, 8.5), dpi=100, constrained_layout=True)

    ax_wave = fig.add_subplot(2, 2, 1)
    ax_mag = fig.add_subplot(2, 2, 2)
    ax_phase = fig.add_subplot(2, 2, 3)
    ax_gl = fig.add_subplot(2, 2, 4)

    wave_time = result["wave_time"]
    original_signal = result["original_signal"]
    phase_reconstructed_signal = result["phase_reconstructed_signal"]

    ax_wave.plot(wave_time, original_signal, linewidth=1.0, label="Original")
    ax_wave.plot(
        wave_time,
        phase_reconstructed_signal,
        linewidth=0.8,
        alpha=0.75,
        label="Reconstructed from saved phase",
    )
    ax_wave.set_title("Waveform")
    ax_wave.set_xlabel("Time [s]")
    ax_wave.set_ylabel("Amplitude [-]")
    ax_wave.legend(loc="upper right", fontsize=8)

    times = result["times"]
    freqs = result["freqs"]
    max_display_freq = min(float(freqs.max()), 8000.0)

    im1 = ax_mag.pcolormesh(
        times,
        freqs,
        result["magnitude_db"],
        shading="auto",
        cmap="viridis",
    )
    ax_mag.set_title("Magnitude Spectrogram")
    ax_mag.set_xlabel("Time [s]")
    ax_mag.set_ylabel("Frequency [Hz]")
    ax_mag.set_ylim(0.0, max_display_freq)
    cbar1 = fig.colorbar(im1, ax=ax_mag)
    cbar1.set_label("Magnitude [dB]")

    im2 = ax_phase.pcolormesh(
        times,
        freqs,
        result["phase_rad"],
        shading="auto",
        cmap="twilight",
    )
    ax_phase.set_title("Phase Spectrogram")
    ax_phase.set_xlabel("Time [s]")
    ax_phase.set_ylabel("Frequency [Hz]")
    ax_phase.set_ylim(0.0, max_display_freq)
    cbar2 = fig.colorbar(im2, ax=ax_phase)
    cbar2.set_label("Phase [rad]")

    im3 = ax_gl.pcolormesh(
        times,
        freqs,
        result["griffinlim_db"],
        shading="auto",
        cmap="viridis",
    )
    ax_gl.set_title("Magnitude-Only Reconstruction")
    ax_gl.set_xlabel("Time [s]")
    ax_gl.set_ylabel("Frequency [Hz]")
    ax_gl.set_ylim(0.0, max_display_freq)
    cbar3 = fig.colorbar(im3, ax=ax_gl)
    cbar3.set_label("Magnitude [dB]")

    playback_lines = [
        _add_waveform_cursor(ax_wave, initial_time=0.0),
        _add_spectrogram_cursor(ax_mag, initial_time=0.0),
        _add_spectrogram_cursor(ax_phase, initial_time=0.0),
        _add_spectrogram_cursor(ax_gl, initial_time=0.0),
    ]

    handles = {
        "figure": fig,
        "playback_lines": playback_lines,
    }
    return fig, handles


def build_audio_trajectory_figure(signal, fs, result, current_time_idx=0):
    fig = Figure(figsize=(9.8, 8.5), dpi=100, constrained_layout=True)

    ax_motion = fig.add_subplot(2, 1, 1, projection="3d")
    ax_spec = fig.add_subplot(2, 1, 2)

    motion_handles = _build_motion_plot(ax_motion, result, current_time_idx=current_time_idx)

    _, _, _, im = ax_spec.specgram(
        signal,
        NFFT=1024,
        Fs=fs,
        noverlap=768,
        cmap="viridis",
    )
    ax_spec.set_title("Trajectory-Auralized Audio Spectrogram")
    ax_spec.set_xlabel("Time [s]")
    ax_spec.set_ylabel("Frequency [Hz]")
    cbar = fig.colorbar(im, ax=ax_spec)
    cbar.set_label("Amplitude [dB]")

    playback_lines = [
        _add_spectrogram_cursor(ax_spec, initial_time=0.0),
    ]

    handles = {
        "figure": fig,
        "playback_lines": playback_lines,
        **motion_handles,
    }
    return fig, handles


def show_dashboard_in_frame(parent, fig, handles):
    global dashboard_canvas, dashboard_figure, dashboard_handles

    for widget in parent.winfo_children():
        widget.destroy()

    dashboard_figure = fig
    dashboard_handles = handles
    dashboard_canvas = FigureCanvasTkAgg(fig, master=parent)
    dashboard_canvas.draw()
    dashboard_canvas.get_tk_widget().pack(fill="both", expand=True)


# ---------------------------------
# Workers
# ---------------------------------
def run_combined_auralization(
    spec_entry,
    pos_entry,
    fs_var,
    fft_entry,
    doppler_var,
    propagation_var,
    dashboard_parent,
):
    global is_processing, current_result, current_signals, current_fs, current_wav_paths, current_play_key

    if is_processing:
        update_message("Auralization already running...", "orange")
        return

    csv_path = spec_entry.get().strip()
    pos_path = pos_entry.get().strip()

    if not os.path.isfile(csv_path):
        update_message("Please upload a valid combined spectrogram CSV.", "red")
        return

    if pos_path and not os.path.isfile(pos_path):
        update_message("Please upload a valid positions CSV.", "red")
        return

    try:
        fs = int(fs_var.get())
        fft_size = int(fft_entry.get())
    except ValueError:
        update_message("Sampling rate and FFT block size must be integers.", "red")
        return

    apply_doppler = doppler_var.get()
    is_processing = True
    update_message("Starting auralization...", "orange")

    def worker():
        global is_processing, current_result, current_signals, current_fs, current_wav_paths, current_play_key
        try:
            signal, fs_out, wav_path, result = auralize_from_csv(
                csv_path=csv_path,
                position_csv_path=pos_path if pos_path else None,
                fs=fs,
                fft_block_size=fft_size,
                apply_doppler=apply_doppler,
                apply_propagation=propagation_var.get(),
                propagation_settings=None,
                output_wav_path="combined_output.wav",
            )

            current_result = result
            current_signals = {"combined": signal}
            current_fs = fs_out
            current_wav_paths = {"combined": wav_path}
            current_play_key = "combined"

            def finish():
                fig, handles = build_combined_dashboard_figure(signal, fs_out, result, current_time_idx=0)
                show_dashboard_in_frame(dashboard_parent, fig, handles)
                update_dashboard_playback(0.0)
                update_message("Auralization complete.", "lightgreen")

            root.after(0, finish)

        except Exception as e:
            traceback.print_exc()
            root.after(0, lambda: update_message(f"Error: {e}", "red"))
        finally:
            is_processing = False

    threading.Thread(target=worker, daemon=True).start()


def run_separate_auralization(
    broadband_entry,
    tonal_entry,
    pos_entry,
    fs_var,
    fft_entry,
    doppler_var,
    propagation_var,
    dashboard_parent,
):
    global is_processing, current_result, current_signals, current_fs, current_wav_paths, current_play_key

    if is_processing:
        update_message("Auralization already running...", "orange")
        return

    broadband_csv = broadband_entry.get().strip()
    tonal_csv = tonal_entry.get().strip()
    pos_path = pos_entry.get().strip()

    if not os.path.isfile(broadband_csv):
        update_message("Please upload a valid broadband CSV.", "red")
        return

    if not os.path.isfile(tonal_csv):
        update_message("Please upload a valid tonal CSV.", "red")
        return

    if pos_path and not os.path.isfile(pos_path):
        update_message("Please upload a valid positions CSV.", "red")
        return

    try:
        fs = int(fs_var.get())
        fft_size = int(fft_entry.get())
    except ValueError:
        update_message("Sampling rate and FFT block size must be integers.", "red")
        return

    apply_doppler = doppler_var.get()
    is_processing = True
    update_message("Starting auralization...", "orange")

    def worker():
        global is_processing, current_result, current_signals, current_fs, current_wav_paths, current_play_key
        try:
            signals, fs_out, wav_paths, result = auralize_from_separate_csv(
                broadband_csv_path=broadband_csv,
                tonal_csv_path=tonal_csv,
                position_csv_path=pos_path if pos_path else None,
                fs=fs,
                fft_block_size=fft_size,
                apply_doppler=apply_doppler,
                apply_propagation=propagation_var.get(),
                propagation_settings=None,
                output_dir=".",
            )

            current_result = result
            current_signals = signals
            current_fs = fs_out
            current_wav_paths = wav_paths
            current_play_key = "combined"

            def finish():
                fig, handles = build_separate_dashboard_figure(
                    signals["combined"], fs_out, result, current_time_idx=0
                )
                show_dashboard_in_frame(dashboard_parent, fig, handles)
                update_dashboard_playback(0.0)
                update_message("Auralization complete.", "lightgreen")

            root.after(0, finish)

        except Exception as e:
            traceback.print_exc()
            root.after(0, lambda: update_message(f"Error: {e}", "red"))
        finally:
            is_processing = False

    threading.Thread(target=worker, daemon=True).start()


def run_audio_analysis(
    audio_entry,
    fs_var,
    fft_entry,
    griffinlim_entry,
    dashboard_parent,
):
    global is_processing, current_result, current_signals, current_fs, current_wav_paths, current_play_key

    if is_processing:
        update_message("Another process is already running...", "orange")
        return

    audio_path = audio_entry.get().strip()
    if not os.path.isfile(audio_path):
        update_message("Please upload a valid WAV or MP3 file.", "red")
        return

    try:
        fft_size = int(fft_entry.get())
        griffinlim_iterations = int(griffinlim_entry.get())
    except ValueError:
        update_message("FFT block size and Griffin-Lim iterations must be integers.", "red")
        return

    try:
        analysis_fs = _safe_get_audio_fs_value(fs_var)
    except ValueError:
        update_message("Invalid analysis sampling rate.", "red")
        return

    is_processing = True
    update_message("Analyzing audio file...", "orange")

    def worker():
        global is_processing, current_result, current_signals, current_fs, current_wav_paths, current_play_key
        try:
            signals, fs_out, wav_paths, result = analyze_audio_input(
                audio_path=audio_path,
                analysis_fs=analysis_fs,
                fft_block_size=fft_size,
                griffinlim_iterations=griffinlim_iterations,
                output_dir=".",
            )

            current_result = result
            current_signals = signals
            current_fs = fs_out
            current_wav_paths = wav_paths
            current_play_key = "original"

            def finish():
                fig, handles = build_audio_learning_figure(result)
                show_dashboard_in_frame(dashboard_parent, fig, handles)
                update_dashboard_playback(0.0)
                update_message("Audio analysis complete.", "lightgreen")

            root.after(0, finish)

        except Exception as e:
            traceback.print_exc()
            root.after(0, lambda: update_message(f"Error: {e}", "red"))
        finally:
            is_processing = False

    threading.Thread(target=worker, daemon=True).start()


def run_audio_trajectory_auralization(
    audio_entry,
    pos_entry,
    fs_var,
    fft_entry,
    doppler_var,
    propagation_var,
    dashboard_parent,
):
    global is_processing, current_result, current_signals, current_fs, current_wav_paths, current_play_key

    if is_processing:
        update_message("Another process is already running...", "orange")
        return

    audio_path = audio_entry.get().strip()
    pos_path = pos_entry.get().strip()

    if not os.path.isfile(audio_path):
        update_message("Please upload a valid WAV or MP3 file.", "red")
        return

    if pos_path and not os.path.isfile(pos_path):
        update_message("Please upload a valid positions CSV.", "red")
        return

    try:
        fft_size = int(fft_entry.get())
        analysis_fs = _safe_get_audio_fs_value(fs_var)
    except ValueError:
        update_message("Invalid sampling rate or FFT block size.", "red")
        return

    is_processing = True
    update_message("Auralizing audio through trajectory...", "orange")

    def worker():
        global is_processing, current_result, current_signals, current_fs, current_wav_paths, current_play_key
        try:
            signals, fs_out, wav_paths, result = auralize_audio_file_with_trajectory(
                audio_path=audio_path,
                position_csv_path=pos_path if pos_path else None,
                analysis_fs=analysis_fs,
                fft_block_size=fft_size,
                apply_doppler=doppler_var.get(),
                apply_propagation=propagation_var.get(),
                propagation_settings=None,
                output_dir=".",
            )

            current_result = result
            current_signals = signals
            current_fs = fs_out
            current_wav_paths = wav_paths
            current_play_key = "trajectory"

            def finish():
                fig, handles = build_audio_trajectory_figure(
                    signals["trajectory"], fs_out, result, current_time_idx=0
                )
                show_dashboard_in_frame(dashboard_parent, fig, handles)
                update_dashboard_playback(0.0)
                update_message("Trajectory auralization complete.", "lightgreen")

            root.after(0, finish)

        except Exception as e:
            traceback.print_exc()
            root.after(0, lambda: update_message(f"Error: {e}", "red"))
        finally:
            is_processing = False

    threading.Thread(target=worker, daemon=True).start()


# ---------------------------------
# Screen builders
# ---------------------------------
def show_home_screen():
    global message_label
    clear_content()
    reset_dashboard()

    outer = ctk.CTkFrame(content_frame)
    outer.pack(fill="both", expand=True, padx=10, pady=10)

    title = ctk.CTkLabel(
        outer,
        text="Auralization Framework",
        font=ctk.CTkFont(size=30, weight="bold"),
    )
    title.pack(anchor="center", pady=(18, 10))

    intro = (
        "The interface supports the auralization of source sound fields from "
        "spectrogram-based inputs. It allows the user to provide either a combined "
        "spectrogram, separate broadband and tonal spectrograms, or an audio file "
        "for STFT analysis and trajectory-based auralization."
    )
    intro_label = ctk.CTkLabel(
        outer,
        text=intro,
        wraplength=980,
        justify="center",
        font=ctk.CTkFont(size=16),
    )
    intro_label.pack(padx=24, pady=(0, 24))

    cards_row = ctk.CTkFrame(outer, fg_color="transparent")
    cards_row.pack(expand=True, fill="both", padx=40, pady=20)

    left_card = ctk.CTkFrame(cards_row)
    left_card.pack(side="left", expand=True, fill="both", padx=(0, 10), pady=10)

    right_card = ctk.CTkFrame(cards_row)
    right_card.pack(side="left", expand=True, fill="both", padx=(10, 0), pady=10)

    ctk.CTkLabel(
        left_card,
        text="Spectrogram as input",
        font=ctk.CTkFont(size=22, weight="bold"),
    ).pack(pady=(28, 16))

    ctk.CTkButton(
        left_card,
        text="Open",
        width=220,
        height=42,
        command=show_spectrogram_menu_screen,
    ).pack(pady=(0, 24))

    ctk.CTkLabel(
        right_card,
        text="WAV / MP3 as input",
        font=ctk.CTkFont(size=22, weight="bold"),
    ).pack(pady=(28, 16))

    ctk.CTkButton(
        right_card,
        text="Open",
        width=220,
        height=42,
        command=show_audio_input_screen,
    ).pack(pady=(0, 24))

    footer = ctk.CTkLabel(
        outer,
        text=(
            "Developed by Ricardo Rocha under the supervision of "
            "Dr. R. Merino Martinez at TU Delft for the "
            "Sustainable Air Transport Modelling Project."
        ),
        font=ctk.CTkFont(size=12),
        text_color="gray75",
    )
    footer.pack(pady=(6, 10))

    message_label = ctk.CTkLabel(outer, text="Ready.", justify="left")
    message_label.pack(anchor="w", padx=20, pady=(0, 10))


def show_spectrogram_menu_screen():
    global message_label
    clear_content()
    reset_dashboard()

    outer = ctk.CTkFrame(content_frame)
    outer.pack(fill="both", expand=True, padx=10, pady=10)

    topbar = ctk.CTkFrame(outer, fg_color="transparent")
    topbar.pack(fill="x", padx=10, pady=(10, 20))

    ctk.CTkButton(topbar, text="Back", width=120, command=show_home_screen).pack(side="left")

    title = ctk.CTkLabel(
        outer,
        text="Spectrogram input",
        font=ctk.CTkFont(size=28, weight="bold"),
    )
    title.pack(pady=(10, 24))

    cards_row = ctk.CTkFrame(outer, fg_color="transparent")
    cards_row.pack(expand=True, fill="both", padx=40, pady=10)

    combined_card = ctk.CTkFrame(cards_row)
    combined_card.pack(side="left", expand=True, fill="both", padx=(0, 10), pady=10)

    separate_card = ctk.CTkFrame(cards_row)
    separate_card.pack(side="left", expand=True, fill="both", padx=(10, 0), pady=10)

    ctk.CTkLabel(
        combined_card,
        text="Combined spectrogram input",
        font=ctk.CTkFont(size=22, weight="bold"),
    ).pack(pady=(30, 16))

    ctk.CTkButton(
        combined_card,
        text="Open",
        width=230,
        height=42,
        command=show_combined_input_screen,
    ).pack(pady=(0, 26))

    ctk.CTkLabel(
        separate_card,
        text="Separate broadband + tonal input",
        font=ctk.CTkFont(size=22, weight="bold"),
    ).pack(pady=(30, 16))

    ctk.CTkButton(
        separate_card,
        text="Open",
        width=230,
        height=42,
        command=show_separate_input_screen,
    ).pack(pady=(0, 26))

    message_label = ctk.CTkLabel(outer, text="Choose an input mode.", justify="left")
    message_label.pack(anchor="w", padx=20, pady=10)


def show_audio_input_screen():
    global message_label
    clear_content()
    reset_dashboard()

    main = ctk.CTkFrame(content_frame)
    main.pack(fill="both", expand=True, padx=10, pady=10)

    left_outer = ctk.CTkFrame(main, width=390)
    left_outer.pack(side="left", fill="y", padx=(0, 10))
    left_outer.pack_propagate(False)

    left_panel = ctk.CTkScrollableFrame(left_outer, fg_color="transparent")
    left_panel.pack(fill="both", expand=True, padx=0, pady=0)

    right_panel = ctk.CTkFrame(main)
    right_panel.pack(side="left", fill="both", expand=True)

    dashboard_frame = ctk.CTkFrame(right_panel)
    dashboard_frame.pack(fill="both", expand=True, padx=10, pady=10)

    ctk.CTkLabel(
        left_panel,
        text="WAV / MP3 input",
        font=ctk.CTkFont(size=24, weight="bold"),
    ).pack(anchor="w", padx=16, pady=(18, 18))

    ctk.CTkLabel(left_panel, text="Audio file (.wav / .mp3)").pack(anchor="w", padx=16, pady=(4, 4))
    audio_entry = ctk.CTkEntry(left_panel, width=320)
    audio_entry.pack(padx=16)
    ctk.CTkButton(
        left_panel,
        text="Browse",
        width=150,
        command=lambda: browse_file(audio_entry, [("Audio files", "*.wav *.mp3")]),
    ).pack(padx=16, pady=(6, 12), anchor="w")

    ctk.CTkLabel(left_panel, text="Analysis / Output Sampling Rate [Hz]").pack(anchor="w", padx=16, pady=(8, 4))
    audio_fs_var = ctk.StringVar(value="44100")
    audio_fs_menu = ctk.CTkComboBox(
        left_panel,
        values=["Original", "8000", "11025", "16000", "22050", "32000", "44100", "48000"],
        variable=audio_fs_var,
        width=320,
    )
    audio_fs_menu.pack(padx=16)

    ctk.CTkLabel(left_panel, text="FFT Block Size").pack(anchor="w", padx=16, pady=(12, 4))
    audio_fft_entry = ctk.CTkEntry(left_panel, width=320)
    audio_fft_entry.insert(0, "2048")
    audio_fft_entry.pack(padx=16)

    ctk.CTkLabel(left_panel, text="Griffin-Lim Iterations").pack(anchor="w", padx=16, pady=(12, 4))
    griffinlim_entry = ctk.CTkEntry(left_panel, width=320)
    griffinlim_entry.insert(0, "32")
    griffinlim_entry.pack(padx=16)

    ctk.CTkLabel(left_panel, text="Positions (.csv) for trajectory mode").pack(anchor="w", padx=16, pady=(18, 4))
    pos_entry = ctk.CTkEntry(left_panel, width=320)
    pos_entry.pack(padx=16)

    row_pos = ctk.CTkFrame(left_panel, fg_color="transparent")
    row_pos.pack(fill="x", padx=16, pady=(6, 10))
    ctk.CTkButton(
        row_pos,
        text="Browse Positions",
        width=150,
        command=lambda: browse_file(pos_entry, [("CSV files", "*.csv")]),
    ).pack(side="left", padx=(0, 8))
    ctk.CTkButton(
        row_pos,
        text="Clear",
        width=80,
        command=lambda: pos_entry.delete(0, "end"),
    ).pack(side="left")

    ctk.CTkLabel(left_panel, text="Sample trajectory").pack(anchor="w", padx=16, pady=(4, 4))
    sample_traj_var = ctk.StringVar(value=SAMPLE_TRAJECTORY_OPTIONS[1])
    sample_traj_menu = ctk.CTkComboBox(
        left_panel,
        values=SAMPLE_TRAJECTORY_OPTIONS,
        variable=sample_traj_var,
        width=320,
    )
    sample_traj_menu.pack(padx=16)

    ctk.CTkButton(
        left_panel,
        text="Generate Sample Positions CSV",
        width=320,
        command=lambda: _generate_audio_sample_positions(audio_entry, audio_fs_var, pos_entry, sample_traj_var),
    ).pack(padx=16, pady=(8, 14))

    doppler_var = ctk.BooleanVar(value=True)
    ctk.CTkCheckBox(
        left_panel,
        text="Apply Doppler effect",
        variable=doppler_var,
    ).pack(anchor="w", padx=16, pady=(6, 8))

    propagation_var = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(
        left_panel,
        text="Apply propagation",
        variable=propagation_var,
    ).pack(anchor="w", padx=16, pady=(0, 12))

    ctk.CTkLabel(
        left_panel,
        text="Audio analysis mode",
        font=ctk.CTkFont(size=18, weight="bold"),
    ).pack(anchor="w", padx=16, pady=(10, 6))

    ctk.CTkButton(
        left_panel,
        text="Analyze audio",
        width=320,
        fg_color="#d48a00",
        hover_color="#b57600",
        command=lambda: run_audio_analysis(
            audio_entry, audio_fs_var, audio_fft_entry, griffinlim_entry, dashboard_frame
        ),
    ).pack(padx=16, pady=(4, 12))

    ctk.CTkLabel(
        left_panel,
        text="Trajectory auralization mode",
        font=ctk.CTkFont(size=18, weight="bold"),
    ).pack(anchor="w", padx=16, pady=(10, 6))

    ctk.CTkButton(
        left_panel,
        text="Auralize audio through trajectory",
        width=320,
        fg_color="#d48a00",
        hover_color="#b57600",
        command=lambda: run_audio_trajectory_auralization(
            audio_entry, pos_entry, audio_fs_var, audio_fft_entry, doppler_var, propagation_var, dashboard_frame
        ),
    ).pack(padx=16, pady=(4, 12))

    ctk.CTkButton(
        left_panel,
        text="Back",
        width=320,
        command=show_home_screen,
    ).pack(padx=16, pady=(0, 16))

    ctk.CTkLabel(left_panel, text="Playback").pack(anchor="w", padx=16, pady=(6, 6))

    row1 = ctk.CTkFrame(left_panel, fg_color="transparent")
    row1.pack(fill="x", padx=16, pady=(0, 6))
    ctk.CTkButton(row1, text="Play Original", width=150, command=lambda: play_audio("original")).pack(side="left", padx=(0, 6))
    ctk.CTkButton(row1, text="Play Phase Recon", width=150, command=lambda: play_audio("phase_reconstructed")).pack(side="left", padx=(6, 0))

    row2 = ctk.CTkFrame(left_panel, fg_color="transparent")
    row2.pack(fill="x", padx=16, pady=(0, 6))
    ctk.CTkButton(row2, text="Play Mag-Only", width=150, command=lambda: play_audio("magnitude_only")).pack(side="left", padx=(0, 6))
    ctk.CTkButton(row2, text="Play Trajectory", width=150, command=lambda: play_audio("trajectory")).pack(side="left", padx=(6, 0))

    row3 = ctk.CTkFrame(left_panel, fg_color="transparent")
    row3.pack(fill="x", padx=16, pady=(0, 8))
    ctk.CTkButton(row3, text="Stop", width=100, command=lambda: stop_audio(reset_cursor=True)).pack(side="left", padx=(0, 6))
    ctk.CTkButton(row3, text="Restart", width=100, command=restart_audio).pack(side="left", padx=(6, 0))

    ctk.CTkLabel(left_panel, text="Save output").pack(anchor="w", padx=16, pady=(10, 6))

    row4 = ctk.CTkFrame(left_panel, fg_color="transparent")
    row4.pack(fill="x", padx=16, pady=(0, 6))
    ctk.CTkButton(
        row4,
        text="Save Original",
        width=150,
        command=lambda: copy_file(current_wav_paths.get("original", ""), ".wav"),
    ).pack(side="left", padx=(0, 6))
    ctk.CTkButton(
        row4,
        text="Save Phase Recon",
        width=150,
        command=lambda: copy_file(current_wav_paths.get("phase_reconstructed", ""), ".wav"),
    ).pack(side="left", padx=(6, 0))

    row5 = ctk.CTkFrame(left_panel, fg_color="transparent")
    row5.pack(fill="x", padx=16, pady=(0, 6))
    ctk.CTkButton(
        row5,
        text="Save Mag-Only",
        width=150,
        command=lambda: copy_file(current_wav_paths.get("magnitude_only", ""), ".wav"),
    ).pack(side="left", padx=(0, 6))
    ctk.CTkButton(
        row5,
        text="Save Trajectory",
        width=150,
        command=lambda: copy_file(current_wav_paths.get("trajectory", ""), ".wav"),
    ).pack(side="left", padx=(6, 0))

    ctk.CTkButton(
        left_panel,
        text="Save Figure",
        width=320,
        command=save_current_figure,
    ).pack(padx=16, pady=(4, 8))

    ctk.CTkButton(left_panel, text="Help", width=320, command=show_help).pack(padx=16, pady=(4, 8))

    message_label = ctk.CTkLabel(left_panel, text="Ready.", wraplength=320, justify="left")
    message_label.pack(anchor="w", padx=16, pady=(12, 12))


def _generate_audio_sample_positions(audio_entry, fs_var, pos_entry, scenario_var):
    audio_path = audio_entry.get().strip()
    if not os.path.isfile(audio_path):
        update_message("Please load a valid audio file first.", "red")
        return

    try:
        analysis_fs = _safe_get_audio_fs_value(fs_var)
        duration = get_audio_duration(audio_path, analysis_fs=analysis_fs)
    except Exception as e:
        traceback.print_exc()
        update_message(f"Could not determine audio duration: {e}", "red")
        return

    generate_sample_positions_for_entry(pos_entry, duration, scenario_var.get())


def show_combined_input_screen():
    global message_label
    clear_content()
    reset_dashboard()

    main = ctk.CTkFrame(content_frame)
    main.pack(fill="both", expand=True, padx=10, pady=10)

    left_outer = ctk.CTkFrame(main, width=390)
    left_outer.pack(side="left", fill="y", padx=(0, 10))
    left_outer.pack_propagate(False)

    left_panel = ctk.CTkScrollableFrame(left_outer, fg_color="transparent")
    left_panel.pack(fill="both", expand=True, padx=0, pady=0)

    right_panel = ctk.CTkFrame(main)
    right_panel.pack(side="left", fill="both", expand=True)

    dashboard_frame = ctk.CTkFrame(right_panel)
    dashboard_frame.pack(fill="both", expand=True, padx=10, pady=10)

    ctk.CTkLabel(
        left_panel,
        text="Combined spectrogram input",
        font=ctk.CTkFont(size=24, weight="bold"),
    ).pack(anchor="w", padx=16, pady=(18, 18))

    ctk.CTkLabel(left_panel, text="Combined spectrogram (.csv)").pack(anchor="w", padx=16, pady=(4, 4))
    spec_entry = ctk.CTkEntry(left_panel, width=320)
    spec_entry.pack(padx=16)
    ctk.CTkButton(
        left_panel,
        text="Browse",
        width=150,
        command=lambda: browse_file(spec_entry, [("CSV files", "*.csv")]),
    ).pack(padx=16, pady=(6, 12), anchor="w")

    ctk.CTkLabel(left_panel, text="Positions (.csv)").pack(anchor="w", padx=16, pady=(4, 4))
    pos_entry = ctk.CTkEntry(left_panel, width=320)
    pos_entry.pack(padx=16)

    row_pos = ctk.CTkFrame(left_panel, fg_color="transparent")
    row_pos.pack(fill="x", padx=16, pady=(6, 10))
    ctk.CTkButton(
        row_pos,
        text="Browse Positions",
        width=150,
        command=lambda: browse_file(pos_entry, [("CSV files", "*.csv")]),
    ).pack(side="left", padx=(0, 8))
    ctk.CTkButton(
        row_pos,
        text="Clear",
        width=80,
        command=lambda: pos_entry.delete(0, "end"),
    ).pack(side="left")

    ctk.CTkLabel(left_panel, text="Sample trajectory").pack(anchor="w", padx=16, pady=(4, 4))
    sample_traj_var = ctk.StringVar(value=SAMPLE_TRAJECTORY_OPTIONS[1])
    sample_traj_menu = ctk.CTkComboBox(
        left_panel,
        values=SAMPLE_TRAJECTORY_OPTIONS,
        variable=sample_traj_var,
        width=320,
    )
    sample_traj_menu.pack(padx=16)

    ctk.CTkButton(
        left_panel,
        text="Generate Sample Positions CSV",
        width=320,
        command=lambda: _generate_combined_sample_positions(spec_entry, pos_entry, sample_traj_var),
    ).pack(padx=16, pady=(8, 14))

    ctk.CTkLabel(left_panel, text="Sampling Rate [Hz]").pack(anchor="w", padx=16, pady=(8, 4))
    fs_var = ctk.StringVar(value="44100")
    fs_menu = ctk.CTkComboBox(
        left_panel,
        values=["8000", "11025", "16000", "22050", "32000", "44100", "48000", "88200", "96000"],
        variable=fs_var,
        width=320,
    )
    fs_menu.pack(padx=16)

    ctk.CTkLabel(left_panel, text="FFT Block Size").pack(anchor="w", padx=16, pady=(12, 4))
    fft_entry = ctk.CTkEntry(left_panel, width=320)
    fft_entry.insert(0, "2048")
    fft_entry.pack(padx=16)

    doppler_var = ctk.BooleanVar(value=True)
    ctk.CTkCheckBox(
        left_panel,
        text="Apply Doppler effect",
        variable=doppler_var,
    ).pack(anchor="w", padx=16, pady=(14, 12))

    propagation_var = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(
        left_panel,
        text="Apply propagation",
        variable=propagation_var,
    ).pack(anchor="w", padx=16, pady=(0, 12))

    ctk.CTkButton(
        left_panel,
        text="Auralize",
        width=320,
        fg_color="#d48a00",
        hover_color="#b57600",
        command=lambda: run_combined_auralization(
            spec_entry, pos_entry, fs_var, fft_entry, doppler_var, propagation_var, dashboard_frame
        ),
    ).pack(padx=16, pady=(8, 8))

    ctk.CTkButton(
        left_panel,
        text="Back",
        width=320,
        command=show_spectrogram_menu_screen,
    ).pack(padx=16, pady=(0, 16))

    playback_row = ctk.CTkFrame(left_panel, fg_color="transparent")
    playback_row.pack(fill="x", padx=16, pady=(4, 8))

    ctk.CTkButton(playback_row, text="Play Combined", width=150, command=lambda: play_audio("combined")).pack(side="left", padx=(0, 6))
    ctk.CTkButton(playback_row, text="Stop", width=80, command=lambda: stop_audio(reset_cursor=True)).pack(side="left", padx=6)
    ctk.CTkButton(playback_row, text="Restart", width=80, command=restart_audio).pack(side="left", padx=(6, 0))

    save_row = ctk.CTkFrame(left_panel, fg_color="transparent")
    save_row.pack(fill="x", padx=16, pady=(8, 8))

    ctk.CTkButton(
        save_row,
        text="Save Combined WAV",
        width=155,
        command=lambda: copy_file(current_wav_paths.get("combined", ""), ".wav"),
    ).pack(side="left", padx=(0, 5))

    ctk.CTkButton(
        save_row,
        text="Save Figure",
        width=155,
        command=save_current_figure,
    ).pack(side="left", padx=(5, 0))

    ctk.CTkButton(left_panel, text="Help", width=320, command=show_help).pack(padx=16, pady=(8, 8))

    message_label = ctk.CTkLabel(left_panel, text="Ready.", wraplength=320, justify="left")
    message_label.pack(anchor="w", padx=16, pady=(12, 12))


def _generate_combined_sample_positions(spec_entry, pos_entry, scenario_var):
    csv_path = spec_entry.get().strip()
    if not os.path.isfile(csv_path):
        update_message("Please load a valid combined spectrogram CSV first.", "red")
        return

    try:
        duration = get_spectrogram_duration(csv_path)
    except Exception as e:
        traceback.print_exc()
        update_message(f"Could not determine spectrogram duration: {e}", "red")
        return

    generate_sample_positions_for_entry(pos_entry, duration, scenario_var.get())


def show_separate_input_screen():
    global message_label
    clear_content()
    reset_dashboard()

    main = ctk.CTkFrame(content_frame)
    main.pack(fill="both", expand=True, padx=10, pady=10)

    left_outer = ctk.CTkFrame(main, width=390)
    left_outer.pack(side="left", fill="y", padx=(0, 10))
    left_outer.pack_propagate(False)

    left_panel = ctk.CTkScrollableFrame(left_outer, fg_color="transparent")
    left_panel.pack(fill="both", expand=True, padx=0, pady=0)

    right_panel = ctk.CTkFrame(main)
    right_panel.pack(side="left", fill="both", expand=True)

    dashboard_frame = ctk.CTkFrame(right_panel)
    dashboard_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    ctk.CTkLabel(
        left_panel,
        text="Separate broadband + tonal input",
        font=ctk.CTkFont(size=22, weight="bold"),
    ).pack(anchor="w", padx=16, pady=(18, 18))

    ctk.CTkLabel(left_panel, text="Broadband spectrogram (.csv)").pack(anchor="w", padx=16, pady=(4, 4))
    broadband_entry = ctk.CTkEntry(left_panel, width=320)
    broadband_entry.pack(padx=16)
    ctk.CTkButton(
        left_panel,
        text="Browse",
        width=150,
        command=lambda: browse_file(broadband_entry, [("CSV files", "*.csv")]),
    ).pack(padx=16, pady=(6, 10), anchor="w")

    ctk.CTkLabel(left_panel, text="Tonal spectrogram (.csv)").pack(anchor="w", padx=16, pady=(4, 4))
    tonal_entry = ctk.CTkEntry(left_panel, width=320)
    tonal_entry.pack(padx=16)
    ctk.CTkButton(
        left_panel,
        text="Browse",
        width=150,
        command=lambda: browse_file(tonal_entry, [("CSV files", "*.csv")]),
    ).pack(padx=16, pady=(6, 10), anchor="w")

    ctk.CTkLabel(left_panel, text="Positions (.csv)").pack(anchor="w", padx=16, pady=(4, 4))
    pos_entry = ctk.CTkEntry(left_panel, width=320)
    pos_entry.pack(padx=16)

    row_pos = ctk.CTkFrame(left_panel, fg_color="transparent")
    row_pos.pack(fill="x", padx=16, pady=(6, 10))
    ctk.CTkButton(
        row_pos,
        text="Browse Positions",
        width=150,
        command=lambda: browse_file(pos_entry, [("CSV files", "*.csv")]),
    ).pack(side="left", padx=(0, 8))
    ctk.CTkButton(
        row_pos,
        text="Clear",
        width=80,
        command=lambda: pos_entry.delete(0, "end"),
    ).pack(side="left")

    ctk.CTkLabel(left_panel, text="Sample trajectory").pack(anchor="w", padx=16, pady=(4, 4))
    sample_traj_var = ctk.StringVar(value=SAMPLE_TRAJECTORY_OPTIONS[1])
    sample_traj_menu = ctk.CTkComboBox(
        left_panel,
        values=SAMPLE_TRAJECTORY_OPTIONS,
        variable=sample_traj_var,
        width=320,
    )
    sample_traj_menu.pack(padx=16)

    ctk.CTkButton(
        left_panel,
        text="Generate Sample Positions CSV",
        width=320,
        command=lambda: _generate_separate_sample_positions(
            broadband_entry, tonal_entry, pos_entry, sample_traj_var
        ),
    ).pack(padx=16, pady=(8, 14))

    ctk.CTkLabel(left_panel, text="Sampling Rate [Hz]").pack(anchor="w", padx=16, pady=(8, 4))
    fs_var = ctk.StringVar(value="44100")
    fs_menu = ctk.CTkComboBox(
        left_panel,
        values=["8000", "11025", "16000", "22050", "32000", "44100", "48000", "88200", "96000"],
        variable=fs_var,
        width=320,
    )
    fs_menu.pack(padx=16)

    ctk.CTkLabel(left_panel, text="FFT Block Size").pack(anchor="w", padx=16, pady=(12, 4))
    fft_entry = ctk.CTkEntry(left_panel, width=320)
    fft_entry.insert(0, "2048")
    fft_entry.pack(padx=16)

    doppler_var = ctk.BooleanVar(value=True)
    ctk.CTkCheckBox(
        left_panel,
        text="Apply Doppler effect",
        variable=doppler_var,
    ).pack(anchor="w", padx=16, pady=(14, 12))

    propagation_var = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(
        left_panel,
        text="Apply propagation",
        variable=propagation_var,
    ).pack(anchor="w", padx=16, pady=(0, 12))

    ctk.CTkButton(
        left_panel,
        text="Auralize",
        width=320,
        fg_color="#d48a00",
        hover_color="#b57600",
        command=lambda: run_separate_auralization(
            broadband_entry, tonal_entry, pos_entry, fs_var, fft_entry, doppler_var, propagation_var, dashboard_frame
        ),
    ).pack(padx=16, pady=(8, 8))

    ctk.CTkButton(
        left_panel,
        text="Back",
        width=320,
        command=show_spectrogram_menu_screen,
    ).pack(padx=16, pady=(0, 16))

    ctk.CTkLabel(left_panel, text="Playback").pack(anchor="w", padx=16, pady=(6, 6))

    row1 = ctk.CTkFrame(left_panel, fg_color="transparent")
    row1.pack(fill="x", padx=16, pady=(0, 6))
    ctk.CTkButton(row1, text="Play Combined", width=150, command=lambda: play_audio("combined")).pack(side="left", padx=(0, 6))
    ctk.CTkButton(row1, text="Play Broadband", width=150, command=lambda: play_audio("broadband")).pack(side="left", padx=(6, 0))

    row2 = ctk.CTkFrame(left_panel, fg_color="transparent")
    row2.pack(fill="x", padx=16, pady=(0, 6))
    ctk.CTkButton(row2, text="Play Tonal", width=150, command=lambda: play_audio("tonal")).pack(side="left", padx=(0, 6))
    ctk.CTkButton(row2, text="Stop", width=80, command=lambda: stop_audio(reset_cursor=True)).pack(side="left", padx=(6, 6))
    ctk.CTkButton(row2, text="Restart", width=80, command=restart_audio).pack(side="left", padx=(0, 0))

    ctk.CTkLabel(left_panel, text="Save output").pack(anchor="w", padx=16, pady=(10, 6))

    row3 = ctk.CTkFrame(left_panel, fg_color="transparent")
    row3.pack(fill="x", padx=16, pady=(0, 6))
    ctk.CTkButton(
        row3,
        text="Save Combined",
        width=150,
        command=lambda: copy_file(current_wav_paths.get("combined", ""), ".wav"),
    ).pack(side="left", padx=(0, 6))
    ctk.CTkButton(
        row3,
        text="Save Broadband",
        width=150,
        command=lambda: copy_file(current_wav_paths.get("broadband", ""), ".wav"),
    ).pack(side="left", padx=(6, 0))

    row4 = ctk.CTkFrame(left_panel, fg_color="transparent")
    row4.pack(fill="x", padx=16, pady=(0, 6))
    ctk.CTkButton(
        row4,
        text="Save Tonal",
        width=150,
        command=lambda: copy_file(current_wav_paths.get("tonal", ""), ".wav"),
    ).pack(side="left", padx=(0, 6))
    ctk.CTkButton(
        row4,
        text="Save Figure",
        width=150,
        command=save_current_figure,
    ).pack(side="left", padx=(6, 0))

    ctk.CTkButton(left_panel, text="Help", width=320, command=show_help).pack(padx=16, pady=(8, 8))

    message_label = ctk.CTkLabel(left_panel, text="Ready.", wraplength=320, justify="left")
    message_label.pack(anchor="w", padx=16, pady=(12, 12))


def _generate_separate_sample_positions(broadband_entry, tonal_entry, pos_entry, scenario_var):
    broadband_csv = broadband_entry.get().strip()
    tonal_csv = tonal_entry.get().strip()

    csv_for_duration = None
    if os.path.isfile(broadband_csv):
        csv_for_duration = broadband_csv
    elif os.path.isfile(tonal_csv):
        csv_for_duration = tonal_csv

    if csv_for_duration is None:
        update_message("Please load at least one valid spectrogram CSV first.", "red")
        return

    try:
        duration = get_spectrogram_duration(csv_for_duration)
    except Exception as e:
        traceback.print_exc()
        update_message(f"Could not determine spectrogram duration: {e}", "red")
        return

    generate_sample_positions_for_entry(pos_entry, duration, scenario_var.get())


# ---------------------------------
# Start
# ---------------------------------
show_home_screen()
root.mainloop()