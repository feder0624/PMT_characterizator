#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from runAnal import run_analysis


SHEET_ID = "1_ymctPCSqQi_dhAj0rm1txnjqNDXYLwxiHr_jPbpXd8"
GID = "1509220136"


def google_sheet_csv_url(sheet_id, gid):
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"


def sanitize_name(s):
    return str(s).replace(" ", "_").replace("/", "_")


def fit_exp_linearized(x, y):
    """
    Fit y = A * exp(B*x) using linearization:
    ln(y) = ln(A) + B*x
    Returns dict with A, B, and fitted y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return None

    coeffs = np.polyfit(x, np.log(y), 1)
    B = coeffs[0]
    lnA = coeffs[1]
    A = np.exp(lnA)
    yfit = A * np.exp(B * x)

    return {
        "A": A,
        "B": B,
        "x": x,
        "y": y,
        "yfit": yfit,
    }

def fit_pow_linearized(x, y):
    """
    Fit y = A * x^B using linearization:
    ln(y) = ln(A) + B * ln(x)
    Returns dict with A, B, and fitted y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (y > 0) & (x > 0)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return None

    coeffs = np.polyfit(np.log(x), np.log(y), 1)
    B = coeffs[0]
    lnA = coeffs[1]
    A = np.exp(lnA)
    yfit = A * x**B

    return {
        "A": A,
        "B": B,
        "x": x,
        "y": y,
        "yfit": yfit,
    }


def read_run_table(sheet_id=SHEET_ID, gid=GID):
    url = google_sheet_csv_url(sheet_id, gid)
    df = pd.read_csv(url)
    df.columns = [c.strip() for c in df.columns]
    return df


def run_batch(
    sheet_id=SHEET_ID,
    gid=GID,
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
    outdir="batch_output",
    time_cut = 2.5
):
    os.makedirs(outdir, exist_ok=True)

    run_table = read_run_table(sheet_id=sheet_id, gid=gid)
    print(f"Loaded {len(run_table)} rows from Google Sheet")

    all_summaries = []

    for _, row in run_table.iterrows():
        pmt = row["PMT"]
        meas_type = row["Type Meas"]
        run_number = int(row["Run"])
        vmon = float(row["Vmon (V)"])
        imon = float(row["Imon (uA)"])
        led_pulse = float(row["LED pulse"])
        r_divider = float(row["R_divider (Mohm)"])
        nevent_meta = int(row["Nevent"])

        pmt_dir = os.path.join(outdir, sanitize_name(pmt), sanitize_name(meas_type))
        os.makedirs(pmt_dir, exist_ok=True)

        print(f"\nRunning analysis: PMT={pmt}, Type={meas_type}, Run={run_number}, V={vmon}")

        try:
            summary, out_csv, debug_dir = run_analysis(
                run_number=run_number,
                channel=channel,
                base_path=base_path,
                max_events=max_events,
                baseline_bins=baseline_bins,
                trigger_window=trigger_window,
                threshold_sigma=threshold_sigma,
                return_threshold_sigma=return_threshold_sigma,
                dt=dt,
                impedance=impedance,
                debug=debug,
                max_overlay=max_overlay,
                outdir=pmt_dir,
                pmt=pmt,
                meas_type=meas_type,
                vmon=vmon,
                imon=imon,
                led_pulse=led_pulse,
                r_divider_mohm=r_divider,
                nevent_meta=nevent_meta,
                lower_time_cut=time_cut
            )
            all_summaries.append(summary)

        except Exception as e:
            print(f"ERROR on run {run_number}: {e}")
            all_summaries.append({
                "PMT": pmt,
                "Type Meas": meas_type,
                "run": run_number,
                "Vmon (V)": vmon,
                "Imon (uA)": imon,
                "LED pulse": led_pulse,
                "R_divider (Mohm)": r_divider,
                "Nevent_meta": nevent_meta,
                "status": f"ERROR: {e}",
            })

    df_all = pd.DataFrame(all_summaries)
    agg_csv = os.path.join(outdir, "all_runs_summary.csv")
    df_all.to_csv(agg_csv, index=False)
    print(f"\nSaved aggregated summary to: {agg_csv}")

    make_all_plots(df_all, outdir=outdir)

    return df_all


def plot_quantity_vs_voltage(df_sub, quantity, quantity_err, title, ylabel, outfile):
    df_sub = df_sub.sort_values("Vmon (V)").copy()
    x = df_sub["Vmon (V)"].values
    y = df_sub[quantity].values
    yerr = df_sub[quantity_err].values if quantity_err in df_sub.columns else None

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if yerr is not None:
        yerr = yerr[mask]

    if len(x) == 0:
        return

    plt.figure(figsize=(7, 5))
    if yerr is not None:
        plt.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3)
    else:
        plt.plot(x, y, "o-")
    plt.xlabel("Voltage [V]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def make_gain_comparison_plot(df_pmt, outdir, pmt_name):
    df_spe = df_pmt[df_pmt["Type Meas"].astype(str).str.upper() == "SPE"].copy()
    df_high = df_pmt[df_pmt["Type Meas"].astype(str).str.upper() == "HIGH"].copy()

    if len(df_spe) == 0 or len(df_high) == 0:
        return

    df_spe = df_spe.sort_values("Vmon (V)").copy()
    df_high = df_high.sort_values("Vmon (V)").copy()

    # Direct SPE gain
    x_spe = df_spe["Vmon (V)"].values
    y_spe = df_spe["SPE gain"].values
    sy_spe = df_spe["SPE gain err"].values

    # Normalize HIGH charge using highest-voltage SPE point
    idx_anchor = np.argmax(df_spe["Vmon (V)"].values)
    v_anchor = df_spe["Vmon (V)"].values[idx_anchor]
    gain_anchor = df_spe["SPE gain"].values[idx_anchor]

    # Find HIGH point at same voltage
    high_anchor_rows = df_high[np.isclose(df_high["Vmon (V)"], v_anchor)]
    if len(high_anchor_rows) == 0:
        print(f"PMT {pmt_name}: no HIGH point found at anchor voltage {v_anchor} V, skipping gain comparison.")
        return

    q_high_anchor = high_anchor_rows["mean_charge_pC"].values[0]
    if not np.isfinite(q_high_anchor) or q_high_anchor == 0:
        print(f"PMT {pmt_name}: invalid HIGH anchor charge, skipping gain comparison.")
        return

    x_high = df_high["Vmon (V)"].values
    q_high = df_high["mean_charge_pC"].values
    sq_high = df_high["mean_charge_pC_err"].values
    y_high_norm_gain = q_high * (gain_anchor / q_high_anchor)
    sy_high_norm_gain = sq_high * (gain_anchor / q_high_anchor)

    # OLD function
    # fit_spe = fit_exp_linearized(x_spe, y_spe)
    # fit_high = fit_exp_linearized(x_high, y_high_norm_gain)
    fit_spe  = fit_pow_linearized(x_spe, y_spe)
    fit_high = fit_pow_linearized(x_high, y_high_norm_gain)

    plt.figure(figsize=(8, 6))
    plt.errorbar(x_spe, y_spe, yerr = sy_spe, fmt = "o", capsize = 3, label="SPE gain")
    plt.errorbar(x_high, y_high_norm_gain, yerr = sy_high_norm_gain, fmt = "s", capsize = 3, label="HIGH normalized gain")

    if fit_spe is not None:
        xs = np.linspace(np.min(fit_spe["x"]), np.max(fit_spe["x"]), 200)
        # ys = fit_spe["A"] * np.exp(fit_spe["B"] * xs)
        ys = fit_spe["A"] * xs**fit_spe["B"]
        # plt.plot(xs, ys, "--", label=f"SPE fit: A exp(BV), B={fit_spe['B']:.4e}")
        plt.plot(xs, ys, "--", label=f"SPE fit: A x^B, B={fit_spe['B']:.4e}")

    if fit_high is not None:
        xh = np.linspace(np.min(fit_high["x"]), np.max(fit_high["x"]), 200)
        # yh = fit_high["A"] * np.exp(fit_high["B"] * xs)
        yh = fit_high["A"] * xh**fit_high["B"]
        # plt.plot(xh, yh, "--", label=f"HIGH fit: A exp(BV), B={fit_high['B']:.4e}")
        plt.plot(xh, yh, "--", label=f"HIGH fit: A x^B, B={fit_high['B']:.4e}")

    plt.xlabel("Voltage [V]")
    plt.ylabel("Gain")
    plt.title(f"Gain comparison - {pmt_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{sanitize_name(pmt_name)}_gain_comparison.png"), dpi=150)
    plt.close()

    rows = [{
        "PMT": pmt_name,
        "anchor_voltage_V": v_anchor,
        "anchor_spe_gain": gain_anchor,
        "anchor_high_charge_pC": q_high_anchor,
        "spe_fit_A": fit_spe["A"] if fit_spe is not None else np.nan,
        "spe_fit_B": fit_spe["B"] if fit_spe is not None else np.nan,
        "high_fit_A": fit_high["A"] if fit_high is not None else np.nan,
        "high_fit_B": fit_high["B"] if fit_high is not None else np.nan,
    }]
    pd.DataFrame(rows).to_csv(
        os.path.join(outdir, f"{sanitize_name(pmt_name)}_gain_fit_summary.csv"),
        index=False
    )


def make_all_plots(df_all, outdir="batch_output"):
    if "PMT" not in df_all.columns:
        return

    df_ok = df_all.copy()
    if "status" in df_ok.columns:
        df_ok = df_ok[df_ok["status"].isna()].copy()

    for pmt_name in sorted(df_ok["PMT"].dropna().unique()):
        df_pmt = df_ok[df_ok["PMT"] == pmt_name].copy()
        pmt_dir = os.path.join(outdir, sanitize_name(pmt_name))
        os.makedirs(pmt_dir, exist_ok=True)

        for meas_type in sorted(df_pmt["Type Meas"].dropna().unique()):
            df_sub = df_pmt[df_pmt["Type Meas"] == meas_type].copy()
            type_dir = os.path.join(pmt_dir, sanitize_name(meas_type))
            os.makedirs(type_dir, exist_ok=True)

            plot_quantity_vs_voltage(
                df_sub=df_sub,
                quantity="trigger_rate",
                quantity_err="trigger_rate_err",
                title=f"{pmt_name} - {meas_type} - Trigger rate vs Voltage",
                ylabel="Trigger rate",
                outfile=os.path.join(type_dir, "trigger_rate_vs_voltage.png"),
            )

            plot_quantity_vs_voltage(
                df_sub=df_sub,
                quantity="mean_charge_pC",
                quantity_err="mean_charge_pC_err",
                title=f"{pmt_name} - {meas_type} - Mean charge vs Voltage",
                ylabel="Mean charge [pC]",
                outfile=os.path.join(type_dir, "mean_charge_vs_voltage.png"),
            )

            plot_quantity_vs_voltage(
                df_sub=df_sub,
                quantity="mean_duration_ns",
                quantity_err="mean_duration_ns_err",
                title=f"{pmt_name} - {meas_type} - Mean duration vs Voltage",
                ylabel="Mean duration [ns]",
                outfile=os.path.join(type_dir, "mean_duration_vs_voltage.png"),
            )

            plot_quantity_vs_voltage(
                df_sub=df_sub,
                quantity="mean_baseline_V",
                quantity_err="mean_baseline_V_err",
                title=f"{pmt_name} - {meas_type} - Mean baseline vs Voltage",
                ylabel="Mean baseline [V]",
                outfile=os.path.join(type_dir, "mean_baseline_vs_voltage.png"),
            )

            plot_quantity_vs_voltage(
                df_sub=df_sub,
                quantity="mean_rms_V",
                quantity_err="mean_rms_V_err",
                title=f"{pmt_name} - {meas_type} - Mean RMS vs Voltage",
                ylabel="Mean RMS [V]",
                outfile=os.path.join(type_dir, "mean_rms_vs_voltage.png"),
            )

            if str(meas_type).upper() == "SPE":
                plot_quantity_vs_voltage(
                    df_sub=df_sub,
                    quantity="SPE gain",
                    quantity_err="mean_charge_pC_err",
                    title=f"{pmt_name} - SPE gain vs Voltage",
                    ylabel="Gain",
                    outfile=os.path.join(type_dir, "spe_gain_vs_voltage.png"),
                )

        make_gain_comparison_plot(df_pmt, outdir=pmt_dir, pmt_name=pmt_name)


def main():
    parser = argparse.ArgumentParser(description="Batch analysis of PMT runs listed in Google Sheet.")
    parser.add_argument("--sheet-id", type=str, default=SHEET_ID, help="Google Sheet ID")
    parser.add_argument("--gid", type=str, default=GID, help="Google Sheet tab gid")
    parser.add_argument("--channel", type=int, default=2, help="Fast channel index")
    parser.add_argument("--base-path", type=str,
                        default="/jupyter-workspace/cnaf-storage/cygno-data/NMV/WC/WC25",
                        help="Directory containing runXXXXX.mid.gz")
    parser.add_argument("--max-events", type=int, default=None, help="Max events to read per run")
    parser.add_argument("--baseline-bins", type=int, default=500, help="Baseline bins")
    parser.add_argument("--window-min", type=int, default=600, help="Trigger window min")
    parser.add_argument("--window-max", type=int, default=800, help="Trigger window max")
    parser.add_argument("--threshold-sigma", type=float, default=5.0, help="Threshold in RMS")
    parser.add_argument("--return-threshold-sigma", type=float, default=1.0, help="Pulse return threshold in RMS")
    parser.add_argument("--dt", type=float, default=400e-12, help="Sample spacing [s]")
    parser.add_argument("--impedance", type=float, default=50.0, help="Input impedance [ohm]")
    parser.add_argument("--debug", action="store_true", help="Save debug plots for each run")
    parser.add_argument("--max-overlay", type=int, default=100, help="Max waveforms in overlay debug plot")
    parser.add_argument("--outdir", type=str, default="batch_output", help="Output directory")
    parser.add_argument("--time-cut", type=float, default=2.5, help="select waveforms with larger times")

    args = parser.parse_args()

    run_batch(
        sheet_id=args.sheet_id,
        gid=args.gid,
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
        time_cut = 2.5
    )


if __name__ == "__main__":
    main()