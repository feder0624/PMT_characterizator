# PMT analysis tools

Small repo for PMT waveform analysis in CYGNO.

## Files

- `runAnal.py` — analyze one run and save a summary CSV
- `batchAnal.py` — analyze many runs from a Google Sheet and make summary plots
- `PMT_check.ipynb` — debug / interactive checks

## What it does

The scripts load MIDAS waveforms, estimate baseline and RMS, select pulses in a trigger window, integrate the signal charge, estimate pulse duration, and compute the SPE gain.

## Requirements

Python packages used:
- numpy
- pandas
- matplotlib
- tqdm
- cygno
- midas.file_reader

## Usage

### Single run
python3 runAnal.py --run 379 --channel 2 --debug

Useful options:
python3 runAnal.py --run 379 --channel 2 --base-path /path/to/data --max-events 1000 --window-min 600 --window-max 800 --outdir output

Output:
- summary CSV for the run
- optional debug plots in runXXXXX_debug/

### Batch mode
python3 batchAnal.py --channel 2 --debug

Useful options:
python3 batchAnal.py --sheet-id YOUR_SHEET_ID --gid YOUR_GID --base-path /path/to/data --outdir batch_output

Output:
- batch_output/all_runs_summary.csv
- per-PMT and per-type plots
- SPE/high-light gain comparison plots

## Notes

- batchAnal.py reads the run list and metadata from a Google Sheet.
- PMT_check.ipynb is intended for quick debugging and visual inspection.

## Suggested procedure of calibration
- place PMT in the marked area
- Take the SPE first at 1100V. The first run should be analyzed ASAP to check the trigger rate. If <10% is fine, continue with the SPE (set 3 on the LED driver should be fine)
- Take first the HIGH at 1100V and analyze it asap to check if it is not saturating. Continue if not (set 7 on the ELD driver should be fine)
 