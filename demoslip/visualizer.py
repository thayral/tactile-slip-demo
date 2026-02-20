from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend

from .custom_pltrcparams import update_pltstuff

def preprocess_fft_data(data: np.ndarray) -> np.ndarray:
    """Match old behavior: convert power -> dB."""
    data = np.asarray(data, dtype=np.float32)
    # old script did 10*log10(data) without clipping; keep safe clip to avoid -inf.
    return 10.0 * np.log10(np.maximum(data, 1e-12))


def visu_demo(res_dict: dict, visu_sample: dict, cfg: dict | None = None) -> None:


    default_cfg = dict(
        # displayed frequency bins
        minfreq=0,
        maxfreq=128,
        # legacy (kept because demo snippet includes them)
        global_min=-12.3,
        global_max=9.3,
        # synchronize signals and predictions (10 ms frame)
        shift_pred_by_one=False,
        # plotting sizing
        scale_factor=2,
        # recording constants
        record_interval_ms=10,
        pze_fs_hz=10_000,
        pze_per_welch_step=100,
        fft_size=256,
        visu_maxfreq=5000, # hz
        pze_synchro_offset  = 1, # 10ms synchronization sensor to bench ground truth
    )

    cfg = {**default_cfg, **(cfg or {})}

    # ---- from res_dict ----
    slip_run_id = res_dict.get("slip_run_id", "") # unique  id  of sample
    seg_type = res_dict.get("seg_type", "")
    clip_start, clip_end = res_dict["clip_wlech_limits"]
    clip_start = int(clip_start)
    clip_end = int(clip_end)




    # pze column name: title / user key
    pze_col_name = res_dict.get("pze_col_name", "") # sensor id 

    labels_dense = np.asarray(res_dict.get("labels_dense", []), dtype=np.float32)
    predicted_dense = np.asarray(res_dict.get("predicted_dense", []), dtype=np.float32)

    # ---- from visu_sample ----
    pze_full = np.asarray(visu_sample["pze"], dtype=np.float32)
    welch_full = np.asarray(visu_sample["tokyo_fft"], dtype=np.float32)



    # slide_* are loaded to keep the new data bundle consistent with other visus, feel free to plot it, synchronization might be  slightly off
    _slide_vel_full = np.asarray(visu_sample.get("slide_vel", []), dtype=np.float32)
    _slide_pos_full = np.asarray(visu_sample.get("slide_pos", []), dtype=np.float32)




    # ---- clip PzE ( slicing + detrend) ----
    pze_start = (clip_start+cfg["pze_synchro_offset"]) * int(cfg["pze_per_welch_step"])
    pze_end = (clip_end+cfg["pze_synchro_offset"]) * int(cfg["pze_per_welch_step"])
    pze_start = max(0, min(pze_start, pze_full.shape[0]))
    pze_end = min(pze_end, pze_full.shape[0])

    pze_clip = detrend(pze_full[pze_start:pze_end].copy())




    welch_db = preprocess_fft_data(welch_full)

    # ---- clip the spectrogram, time x freq ----
    minfreq = int(cfg["minfreq"])
    maxfreq = int(cfg["maxfreq"])
    minfreq = max(0, minfreq)
    maxfreq = max(minfreq + 1, maxfreq)

    # Cap to available freq bins
    maxfreq_eff = min(maxfreq, welch_db.shape[0])
    minfreq_eff = min(minfreq, maxfreq_eff - 1)


    welch_clip = welch_db[
        minfreq_eff:maxfreq_eff,
        clip_start : clip_end + 1,
    ]

    # ---- time axes ----
    dt_slide = cfg["record_interval_ms"] / 1000.0

    time_in_seconds = np.arange(labels_dense.shape[0], dtype=np.float32) * dt_slide

    dt_pze = 1.0 / float(cfg["pze_fs_hz"])
    time_in_seconds_10khz = (
        np.arange(pze_clip.shape[0], dtype=np.float32) * dt_pze 
    )


    # ---- frequency axis ----
    fft_size = int(cfg["fft_size"])
    sampling_rate = float(cfg["pze_fs_hz"])
    freq_resolution = sampling_rate / fft_size
    freq_bins = np.arange(fft_size // 2, dtype=np.float32) * freq_resolution



    # Map bin indices to Hz for extent.
    # If user requests more bins than we have in freq_bins, clamp.
    maxfreq_hz_idx = min(maxfreq_eff, freq_bins.shape[0])
    minfreq_hz_idx = min(minfreq_eff, maxfreq_hz_idx - 1)



    # ---- plotting (kept same as old) ----
    scale_factor = float(cfg.get("scale_factor", 2))
    update_pltstuff(scale_factor)

    final_width = 3.5  # inches
    aspect_ratio = 1.75
    working_width = final_width * scale_factor

    fig, axs = plt.subplots(
        3,
        sharex=True,
        sharey=False,
        gridspec_kw={"height_ratios": [1.5, 1.5, 1.0]},
        figsize=(working_width, working_width / aspect_ratio),
    )

    # ---- PzE ----
    axi = 0
    axs[axi].plot(time_in_seconds_10khz, pze_clip, color="midnightblue", linewidth=1.5)
    axs[axi].set_ylabel("PzE")
    axs[axi].tick_params(labelbottom=False)
    axs[axi].set_ylim(ymin=-100, ymax=100)

    # ---- Spectrogram ----
    axi += 1
    vmin = -80
    vmax = float(np.max(welch_clip)) 



    extent = [
        time_in_seconds[0] + cfg["num_welch_pad"] * cfg["record_interval_ms"]/1000 - 0.005 + 0.005, # align/center the 10ms resolution and cfg["num_welch_pad"] to align the fft sliding window 256 pze data points
        time_in_seconds[-1] + cfg["num_welch_pad"] * cfg["record_interval_ms"]/1000 + 0.005 + 0.005,
        float(freq_bins[minfreq_hz_idx]) ,
        float(freq_bins[maxfreq_hz_idx - 1]) ,
    ]



    im = axs[axi].imshow(
        welch_clip,
        aspect="auto",
        origin="lower",
        cmap="cividis",
        extent=extent,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    axs[axi].set_ylabel("Freq. (Hz)")
    axs[axi].set_ylim(ymin=0, ymax=cfg["visu_maxfreq"])

    if cfg["visu_maxfreq"]== 5000:
        axs[axi].set_yticks([0, 5000])
        axs[axi].set_yticklabels(["0", "5k"])

    axs[axi].tick_params(labelbottom=False)

    # ---- Labels / predictions ----
    axi += 1
    if labels_dense.size and predicted_dense.size:
        assert labels_dense.shape[0] == predicted_dense.shape[0]

    if labels_dense.size:
        axs[axi].step(
            time_in_seconds[: labels_dense.shape[0]],
            labels_dense,
            where="pre",
            color="firebrick",
            linestyle="--",
            linewidth=3.2,
            label="Ground-truth",
            alpha=1,
        )
    if predicted_dense.size:
        axs[axi].step(
            time_in_seconds[: predicted_dense.shape[0]],
            predicted_dense,
            where="pre",
            color="dodgerblue",
            linestyle="-",
            linewidth=3.2,
            label="Detection",
            alpha=0.8,
        )

    axs[axi].legend(loc="upper right", prop={"size": 8})
    axs[axi].set_ylim(-0.03, 1.03)
    axs[axi].set_yticks([0, 1])
    axs[axi].set_yticklabels(["0", "1"])
    axs[axi].set_ylabel("Classif.")
    axs[axi].set_xlabel("Time (s)")

    # X axis zoom
    # axs[0].set_xlim(xmin=1.0, xmax=1.6)

    fig.align_ylabels(axs[:])
    fig.suptitle(f"{slip_run_id}")
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.45)
    plt.show()
