#!/usr/bin/env python3
"""
count_spikes.py — extract θ along the interface for one or more PVTU frames
and report spike count + dominant wavelength.

Run with ParaView's pvpython, NOT system python3:

  /Applications/ParaView-6.0.1.app/Contents/bin/pvpython \\
      analysis/count_spikes.py \\
      Results/20260430_235602_hedge_r3_direct_amr/solution__30000.pvtu \\
      [Results/.../solution__40000.pvtu ...]

For each frame, the script:
  1. Loads the .pvtu (auto-stitches all 8 partitions)
  2. Resamples θ on a high-resolution horizontal line at y = y_sample
     (default: 0.15, just above the pool depth of 0.11)
  3. Detects sign changes of θ along that line → x-coordinates of the
     interface intersection
  4. Pairs adjacent intersections to compute spike widths
  5. Reports: number of spikes, mean spacing, dominant wavelength via FFT,
     and writes a CSV with raw θ(x) for plotting

The output CSV can be plotted with system python3 + matplotlib afterwards.
"""
import os
import sys
import csv
import argparse
import numpy as np

# ParaView imports (only available under pvpython)
try:
    from paraview.simple import (
        XMLPartitionedUnstructuredGridReader,
        ResampleToImage,
        servermanager,
        UpdatePipeline,
        Delete,
    )
    import paraview.simple as ps
    PVAVAILABLE = True
except ImportError:
    PVAVAILABLE = False


def extract_theta_line(pvtu_path: str, y_sample: float, nx: int) -> np.ndarray:
    """Resample θ along the horizontal line y = y_sample.
    Returns array shape (nx, 2): columns [x, theta].
    """
    rdr = XMLPartitionedUnstructuredGridReader(FileName=[pvtu_path])
    UpdatePipeline()

    # Compute bounding box
    bds = rdr.GetDataInformation().GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
    xmin, xmax = bds[0], bds[1]
    print(f"    domain x = [{xmin:.4f}, {xmax:.4f}], "
          f"y in [{bds[2]:.4f}, {bds[3]:.4f}]")

    # Use ResampleToImage on a thin slab around y_sample to keep memory tiny
    dy = max(1e-3, (bds[3] - bds[2]) * 1e-3)
    rs = ResampleToImage(Input=rdr)
    rs.SamplingDimensions = [nx, 2, 1]
    rs.UseInputBounds = 0
    rs.SamplingBounds = [xmin, xmax, y_sample - dy, y_sample + dy, 0.0, 0.0]
    UpdatePipeline()

    # Pull data into numpy
    data = servermanager.Fetch(rs)
    pd = data.GetPointData()
    theta_arr = None
    for name in ("theta", "Theta", "T"):
        a = pd.GetArray(name)
        if a is not None:
            theta_arr = a
            break
    if theta_arr is None:
        # list available arrays
        names = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
        raise RuntimeError(f"No θ array found. Available: {names}")

    # Pull only the bottom row (j=0 of the 2-row slab)
    theta = np.array([theta_arr.GetTuple1(i) for i in range(nx)])
    x = np.linspace(xmin, xmax, nx)

    Delete(rs); Delete(rdr)
    return np.column_stack([x, theta])


def count_zero_crossings(x: np.ndarray, theta: np.ndarray) -> list[float]:
    """Return list of x-coordinates where θ changes sign."""
    out = []
    for i in range(len(theta) - 1):
        if theta[i] * theta[i + 1] < 0:
            # linear interpolation to zero
            t = -theta[i] / (theta[i + 1] - theta[i])
            out.append(x[i] + t * (x[i + 1] - x[i]))
    return out


def dominant_wavelength_fft(x: np.ndarray, theta: np.ndarray) -> float:
    """FFT of θ(x); return dominant non-DC wavelength (= 1/peak frequency)."""
    n = len(x)
    if n < 16:
        return float("nan")
    # remove mean, apply Hann window
    th = (theta - theta.mean()) * np.hanning(n)
    spec = np.abs(np.fft.rfft(th))
    spec[0] = 0  # kill DC
    if spec.max() < 1e-12:
        return float("nan")
    k_peak = np.argmax(spec)
    if k_peak == 0:
        return float("nan")
    L = x[-1] - x[0]
    return L / k_peak  # wavelength


def analyze_one_frame(pvtu_path: str, y_sample: float, nx: int,
                      out_dir: str) -> dict:
    name = os.path.splitext(os.path.basename(pvtu_path))[0]
    print(f"  Frame {name}: y_sample = {y_sample}")

    arr = extract_theta_line(pvtu_path, y_sample, nx)
    x, theta = arr[:, 0], arr[:, 1]

    # Save raw line
    csv_path = os.path.join(out_dir, f"{name}__theta_at_y{y_sample:.3f}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "theta"])
        for xi, ti in zip(x, theta):
            w.writerow([f"{xi:.6f}", f"{ti:.6f}"])
    print(f"    → wrote {csv_path}")

    crossings = count_zero_crossings(x, theta)
    n_spikes = max(0, len(crossings) // 2)  # each spike has 2 sign changes

    if len(crossings) >= 2:
        spacings = np.diff(crossings)
        mean_spacing = spacings.mean()
        std_spacing = spacings.std()
    else:
        mean_spacing = float("nan")
        std_spacing = float("nan")

    lam_dom = dominant_wavelength_fft(x, theta)

    summary = {
        "frame": name,
        "y_sample": y_sample,
        "n_zero_crossings": len(crossings),
        "n_spikes": n_spikes,
        "mean_crossing_spacing": mean_spacing,
        "std_crossing_spacing": std_spacing,
        "dominant_wavelength_fft": lam_dom,
    }

    print(f"    crossings = {len(crossings)}, spikes ≈ {n_spikes}, "
          f"mean spacing = {mean_spacing:.4f}, FFT λ = {lam_dom:.4f}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pvtu_files", nargs="+", help="One or more .pvtu files")
    p.add_argument("--y", type=float, default=0.15,
                   help="Horizontal sampling line y-coordinate (default 0.15)")
    p.add_argument("--nx", type=int, default=2048,
                   help="Number of sample points along x (default 2048)")
    p.add_argument("--out", type=str, default=None,
                   help="Output dir for CSVs (default: ./analysis_out/)")
    args = p.parse_args()

    if not PVAVAILABLE:
        sys.exit("ERROR: this script requires pvpython.\n"
                 "Run with: /Applications/ParaView-6.0.1.app/Contents/bin/pvpython "
                 "analysis/count_spikes.py ...")

    out_dir = args.out or os.path.join(os.getcwd(), "analysis_out")
    os.makedirs(out_dir, exist_ok=True)

    summaries = []
    for f in args.pvtu_files:
        if not os.path.exists(f):
            print(f"  SKIP (missing): {f}")
            continue
        try:
            s = analyze_one_frame(f, args.y, args.nx, out_dir)
            summaries.append(s)
        except Exception as e:
            print(f"  FAIL on {f}: {e}")

    # write summary CSV
    summary_path = os.path.join(out_dir, "spike_summary.csv")
    if summaries:
        keys = list(summaries[0].keys())
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(summaries)
        print(f"\nSummary written to {summary_path}")
        print("─" * 64)
        for s in summaries:
            print(f"  {s['frame']:35s}  spikes={s['n_spikes']:2d}  "
                  f"FFT λ = {s['dominant_wavelength_fft']:.4f}")
        print("─" * 64)


if __name__ == "__main__":
    main()
