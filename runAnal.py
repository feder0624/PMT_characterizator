#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import cygno as cy
import midas.file_reader


def mean_and_sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return x[0], 0.0
    return np.mean(x), np.std(x, ddof=1) / np.sqrt(len(x))


def load_waveforms(run_number, channel=2,
                   base_path="/jupyter-workspace/cnaf-storage/cygno-data/NMV/WC/WC25",
                   max_events=None,
                   verbose=True):
    mpath = os.path.join(base_path, f"run{run_number:05d}.mid.gz")
    if not os.path.exists(mpath):
        raise FileNotFoundError(f"File not found: {mpath}")

    if verbose:
        print(f"Opening {mpath}")

    mfile = midas.file_reader.MidasFile(mpath)
    odb = cy.get_bor_odb(mfile)
    corrected = odb.data["Configurations"]["DRS4Correction"]
    channels_offsets = odb.data["Configurations"]["DigitizerOffset"]

    waveforms = []
    nevents = 0

    for event in mfile:
        if event.header.is_midas_internal_event():
            continue
        if "DGH0" not in event.banks:
            continue

        full_header = cy.daq_dgz_full2header(event.banks["DGH0"], verbose=False)
        w_fast, w_slow = cy.daq_dgz_full2array(
            event.banks["DIG0"],
            full_header,
            verbose=False,
            corrected=corrected,
            ch_offset=channels_offsets
        )

        if channel >= len(w_fast):
            raise IndexError(f"Requested channel {channel}, but only {len(w_fast)} channels are available.")

        waveforms.append(np.asarray(w_fast[channel], dtype=float))
        nevents += 1

        if max_events is not None and nevents >= max_events:
            break

    if verbose:
        print(f"Loaded {len(waveforms)} waveforms from run {run_number}")

    return waveforms


def find_pulse_bounds(wf_bs, min_idx, rms, return_threshold_sigma=1.0):
    thr = -return_threshold_sigma * rms
    n = len(wf_bs)

    left = min_idx
    while left > 0 and wf_bs[left] < thr:
        left -= 1

    right = min_idx
    while right < n - 1 and wf_bs[right] < thr:
        right += 1

    return left, right


def choose_noise_window(signal_left, signal_right, baseline_bins, total_bins, margin=20):
    width = signal_right - signal_left + 1

    start = max(margin, baseline_bins // 2 - width // 2)
    end = start + width - 1

    if end < baseline_bins - margin:
        return start, end

    start = margin
    end = start + width - 1
    if end < min(baseline_bins, total_bins - margin):
        return start, end

    end = min(total_bins - margin - 1, baseline_bins - margin - 1)
    start = max(margin, end - width + 1)
    return start, end


def analyze_waveforms(
    waveforms,
    dt=400e-12,
    impedance=50.0,
    baseline_bins=500,
    trigger_window=(600, 800),
    threshold_sigma=5.0,
    return_threshold_sigma=1.0,
    adc_to_volt=1.0/4096.0,
    lower_time_cut=2.5
):
    records = []
    selected_debug = []

    x0, x1 = trigger_window
    n_total = len(waveforms)
    n_selected = 0

    for i, wf in enumerate(tqdm(waveforms, desc="Analyzing")):
        wf = np.asarray(wf, dtype=float)

        if len(wf) <= max(baseline_bins, x1):
            continue

        wf_V = wf * adc_to_volt
        baseline = np.mean(wf_V[:baseline_bins])
        rms = np.std(wf_V[:baseline_bins], ddof=1)
        wf_bs = wf_V - baseline

        window = wf_bs[x0:x1 + 1]
        threshold = -threshold_sigma * rms
        has_signal = np.any(window < threshold)

        record = {
            "event": i,
            "baseline": baseline,
            "rms": rms,
            "selected": has_signal,
        }

        if has_signal:
            #n_selected += 1

            local_min_idx = np.argmin(window)
            min_idx = x0 + local_min_idx
            min_amp = wf_bs[min_idx]
            min_time_ns = min_idx * dt * 1e9

            left, right = find_pulse_bounds(
                wf_bs, min_idx, rms, return_threshold_sigma=return_threshold_sigma
            )

            duration_bins = right - left + 1
            duration_ns = duration_bins * dt * 1e9
            if duration_ns > lower_time_cut:
                n_selected+=1
            else:
                continue

            signal_area_vs = np.sum(wf_bs[left:right + 1]) * dt
            charge_pc = (-signal_area_vs / impedance) * 1e12

            nleft, nright = choose_noise_window(
                left, right, baseline_bins, len(wf_bs), margin=20
            )
            noise_area_vs = np.sum(wf_bs[nleft:nright + 1]) * dt
            noise_charge_pc = (-noise_area_vs / impedance) * 1e12

            record.update({
                "min_idx": min_idx,
                "min_time_ns": min_time_ns,
                "min_amp": min_amp,
                "left_idx": left,
                "right_idx": right,
                "duration_bins": duration_bins,
                "duration_ns": duration_ns,
                "charge_pC": charge_pc,
                "noise_charge_pC": noise_charge_pc,
            })

            selected_debug.append({
                "event": i,
                "wf_bs": wf_bs.copy(),
                "rms": rms,
                "threshold": threshold,
                "min_idx": min_idx,
                "left": left,
                "right": right,
                "nleft": nleft,
                "nright": nright,
            })

        records.append(record)

    df = pd.DataFrame(records)

    trigger_rate = n_selected / n_total if n_total > 0 else np.nan
    trigger_err = np.sqrt(trigger_rate * (1.0 - trigger_rate) / n_total) if n_total > 0 else np.nan

    sel = df[df["selected"] == True].copy()

    mean_charge, err_charge = mean_and_sem(sel["charge_pC"].values if "charge_pC" in sel.columns else [])
    mean_duration, err_duration = mean_and_sem(sel["duration_ns"].values if "duration_ns" in sel.columns else [])
    mean_baseline, err_baseline = mean_and_sem(df["baseline"].values if "baseline" in df.columns else [])
    mean_rms, err_rms = mean_and_sem(df["rms"].values if "rms" in df.columns else [])

    summary = {
        "n_total": n_total,
        "n_selected": n_selected,
        "trigger_rate": trigger_rate,
        "trigger_rate_err": trigger_err,
        "mean_charge_pC": mean_charge,
        "mean_charge_pC_err": err_charge,
        "mean_duration_ns": mean_duration,
        "mean_duration_ns_err": err_duration,
        "mean_baseline_V": mean_baseline,
        "mean_baseline_V_err": err_baseline,
        "mean_rms_V": mean_rms,
        "mean_rms_V_err": err_rms,
        "SPE gain": mean_charge * 1e-12 / 1.6e-19
    }

    return summary, df, selected_debug


def make_debug_plots(df, selected_debug, run_number,
                     trigger_window=(600, 800),
                     max_overlay=30,
                     outdir_base="."):
    debug_dir = os.path.join(outdir_base, f"run{run_number:05d}_debug")
    os.makedirs(debug_dir, exist_ok=True)

    sel = df[df["selected"] == True].copy()
    if len(sel) == 0 or len(selected_debug) == 0:
        print(f"No selected events. Debug folder created but no plots saved: {debug_dir}")
        return debug_dir

    x0, x1 = trigger_window

    # 1) Overlay of selected waveforms
    nplot = min(max_overlay, len(selected_debug))
    plt.figure(figsize=(12, 6))

    for item in selected_debug[:nplot]:
        plt.plot(item["wf_bs"], alpha=0.30, lw=1)

    ref = selected_debug[0]
    plt.axvspan(x0, x1, alpha=0.15, label="trigger window")
    plt.axhline(ref["threshold"], linestyle="--", label="-5 RMS (example)")
    plt.axhline(-ref["rms"], linestyle=":", label="-1 RMS (example)")
    plt.axvspan(ref["left"], ref["right"], alpha=0.20, label="signal ROI")
    plt.axvspan(ref["nleft"], ref["nright"], alpha=0.20, label="noise ROI")
    plt.axvline(ref["min_idx"], linestyle="--", label="pulse minimum")

    plt.xlabel("Sample")
    plt.ylabel("Baseline-subtracted amplitude [V]")
    plt.title(f"Overlay of selected waveforms ({nplot} events)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(debug_dir, "overlay_selected_waveforms.png"), dpi=150)
    plt.close()

    # 2) Signal and noise integrated spectra on same histogram (common bins)
    signal = sel["charge_pC"].dropna().values
    noise = sel["noise_charge_pC"].dropna().values

    if len(signal) > 0 and len(noise) > 0:
        xmin = min(signal.min(), noise.min())
        xmax = max(signal.max(), noise.max())
        if xmin == xmax:
            xmin -= 0.5
            xmax += 0.5
        bins = np.linspace(xmin, xmax, 101)

        plt.figure(figsize=(8, 5))
        plt.hist(signal, bins=bins, alpha=0.6, label="signal charge")
        plt.hist(noise, bins=bins, alpha=0.6, label="noise charge")
        plt.xlabel("Integrated charge [pC]")
        plt.ylabel("Counts")
        plt.title("Signal and noise integrated spectra")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, "signal_noise_integrated_spectra.png"), dpi=150)
        plt.close()

    # 3) Charge vs duration scatter
    if "duration_ns" in sel.columns and "charge_pC" in sel.columns:
        plt.figure(figsize=(8, 5))
        plt.scatter(sel["duration_ns"], sel["charge_pC"], s=10, alpha=0.5)
        plt.xlabel("Pulse duration [ns]")
        plt.ylabel("Integrated signal charge [pC]")
        plt.title("Charge vs duration")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, "charge_vs_duration.png"), dpi=150)
        plt.close()

    # 4) Duration spectrum
    if "duration_ns" in sel.columns:
        plt.figure(figsize=(8, 5))
        plt.hist(sel["duration_ns"].dropna(), bins=80)
        plt.xlabel("Pulse duration [ns]")
        plt.ylabel("Counts")
        plt.title("Pulse duration spectrum")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, "duration_spectrum.png"), dpi=150)
        plt.close()

    return debug_dir


def save_summary_csv(summary, run_number, channel, outdir="."):
    os.makedirs(outdir, exist_ok=True)
    out_csv = os.path.join(outdir, f"run{run_number:05d}_ch{channel}_summary.csv")
    pd.DataFrame([summary]).to_csv(out_csv, index=False)
    return out_csv


def run_analysis(
    run_number,
    channel=2,
    base_path="/jupyter-workspace/cnaf-storage/cygno-data/NMV/WC/WC25",
    max_events=None,
    baseline_bins=500,
    trigger_window=(600, 800),
    threshold_sigma=5.0,
    return_threshold_sigma=1.0,
    dt=400e-12,
    impedance=50.0,
    debug=False,
    max_overlay=100,
    outdir=".",
    pmt=None,
    meas_type=None,
    vmon=None,
    imon=None,
    led_pulse=None,
    r_divider_mohm=None,
    nevent_meta=None,
    lower_time_cut=2.5
):
    waveforms = load_waveforms(
        run_number=run_number,
        channel=channel,
        base_path=base_path,
        max_events=max_events,
        verbose=True
    )

    summary, df, selected_debug = analyze_waveforms(
        waveforms,
        dt=dt,
        impedance=impedance,
        baseline_bins=baseline_bins,
        trigger_window=trigger_window,
        threshold_sigma=threshold_sigma,
        return_threshold_sigma=return_threshold_sigma,
        lower_time_cut=lower_time_cut
    )

    summary["run"] = run_number
    summary["channel"] = channel
    summary["trigger_window_min"] = trigger_window[0]
    summary["trigger_window_max"] = trigger_window[1]
    summary["baseline_bins"] = baseline_bins
    summary["threshold_sigma"] = threshold_sigma
    summary["return_threshold_sigma"] = return_threshold_sigma
    summary["dt_s"] = dt
    summary["impedance_ohm"] = impedance

    if pmt is not None:
        summary["PMT"] = pmt
    if meas_type is not None:
        summary["Type Meas"] = meas_type
    if vmon is not None:
        summary["Vmon (V)"] = vmon
    if imon is not None:
        summary["Imon (uA)"] = imon
    if led_pulse is not None:
        summary["LED pulse"] = led_pulse
    if r_divider_mohm is not None:
        summary["R_divider (Mohm)"] = r_divider_mohm
    if nevent_meta is not None:
        summary["Nevent_meta"] = nevent_meta

    out_csv = save_summary_csv(summary, run_number, channel, outdir=outdir)

    if debug:
        debug_dir = make_debug_plots(
            df,
            selected_debug,
            run_number=run_number,
            trigger_window=trigger_window,
            max_overlay=max_overlay,
            outdir_base=outdir
        )
    else:
        debug_dir = None

    return summary, out_csv, debug_dir


def main():
    parser = argparse.ArgumentParser(description="PMT waveform analysis for high-light and SPE runs.")
    parser.add_argument("--run", type=int, required=True, help="Run number, e.g. 379")
    parser.add_argument("--channel", type=int, default=2, help="Fast channel index, default: 2")
    parser.add_argument("--base-path", type=str,
                        default="/jupyter-workspace/cnaf-storage/cygno-data/NMV/WC/WC25",
                        help="Directory containing runXXXXX.mid.gz")
    parser.add_argument("--max-events", type=int, default=None, help="Maximum number of events to read")
    parser.add_argument("--baseline-bins", type=int, default=500, help="Bins for baseline/RMS estimation")
    parser.add_argument("--window-min", type=int, default=600, help="Trigger window start")
    parser.add_argument("--window-max", type=int, default=800, help="Trigger window end")
    parser.add_argument("--threshold-sigma", type=float, default=5.0, help="Selection threshold in RMS")
    parser.add_argument("--return-threshold-sigma", type=float, default=3.0,
                        help="Threshold used to define pulse boundaries")
    parser.add_argument("--dt", type=float, default=400e-12, help="Sample spacing in seconds")
    parser.add_argument("--impedance", type=float, default=50.0, help="Input impedance in ohm")
    parser.add_argument("--debug", action="store_true", help="Save debug plots")
    parser.add_argument("--max-overlay", type=int, default=1, help="Max selected waveforms in overlay")
    parser.add_argument("--outdir", type=str, default=".", help="Base output directory")

    # Optional metadata for batch-friendly standalone runs
    parser.add_argument("--pmt", type=str, default=None, help="PMT name")
    parser.add_argument("--meas-type", type=str, default=None, help="Measurement type, e.g. SPE or HIGH")
    parser.add_argument("--vmon", type=float, default=None, help="Monitored HV [V]")
    parser.add_argument("--imon", type=float, default=None, help="Monitored current [uA]")
    parser.add_argument("--led-pulse", type=float, default=None, help="LED pulse setting")
    parser.add_argument("--r-divider-mohm", type=float, default=None, help="Divider resistance [Mohm]")
    parser.add_argument("--nevent-meta", type=int, default=None, help="Metadata number of requested events")
    parser.add_argument("--time-cut", type=float, default=2.5, help="select waveforms with larger times")

    args = parser.parse_args()

    summary, out_csv, debug_dir = run_analysis(
        run_number=args.run,
        channel=args.channel,
        base_path=args.base_path,
        max_events=args.max_events,
        baseline_bins=args.baseline_bins,
        trigger_window=(args.window_min, args.window_max),
        threshold_sigma=args.threshold_sigma,
        return_threshold_sigma=args.return_threshold_sigma,
        dt=args.dt,
        impedance=args.impedance,
        debug=args.debug,
        max_overlay=args.max_overlay,
        outdir=args.outdir,
        pmt=args.pmt,
        meas_type=args.meas_type,
        vmon=args.vmon,
        imon=args.imon,
        led_pulse=args.led_pulse,
        r_divider_mohm=args.r_divider_mohm,
        nevent_meta=args.nevent_meta,
        lower_time_cut=args.time_cut
    )

    print("\n=== Summary ===")
    print(f"Total waveforms        : {summary['n_total']}")
    print(f"Selected pulses        : {summary['n_selected']}")
    print(f"Trigger rate           : {summary['trigger_rate']:.6f} ± {summary['trigger_rate_err']:.6f}")
    print(f"Mean charge [pC]       : {summary['mean_charge_pC']:.6f} ± {summary['mean_charge_pC_err']:.6f}")
    print(f"Mean duration [ns]     : {summary['mean_duration_ns']:.6f} ± {summary['mean_duration_ns_err']:.6f}")
    print(f"Mean baseline [V]      : {summary['mean_baseline_V']:.6e} ± {summary['mean_baseline_V_err']:.6e}")
    print(f"Mean RMS [V]           : {summary['mean_rms_V']:.6e} ± {summary['mean_rms_V_err']:.6e}")
    print(f"SPE gain               : {summary['SPE gain']:.6e}")

    print(f"\nSaved summary CSV to: {out_csv}")
    if debug_dir is not None:
        print(f"Saved debug plots in: {debug_dir}")


if __name__ == "__main__":
    main()