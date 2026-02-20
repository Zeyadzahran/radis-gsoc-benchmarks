#!/usr/bin/env python3
"""
=============================================================================
RADIS Lazy-Loading Benchmark
=============================================================================
A rigorous benchmark comparing file formats (HDF5, Feather, Parquet) and
dataframe engines (Pandas eager vs. Polars lazy) for memory-efficient handling
of large spectroscopic databases (HITRAN/HITEMP).

Context:
    RADIS currently relies on Pandas or Vaex for processing 50GB+ line-list
    databases. Vaex is unmaintained, and Pandas eagerly loads entire datasets
    into RAM, causing out-of-memory crashes on consumer hardware. This script
    quantifies the trade-offs between storage formats and execution engines to
    identify the best replacement strategy.

    This version uses **real CO2 spectroscopic data** fetched from the HITRAN
    database via ``radis.fetch_hitran``, because real data contains patterns
    (zeros, repeated quantum numbers, clustered wavenumbers) that affect
    columnar compression ratios very differently than synthetic random floats.

Metrics captured:
    1. File size on disk (MB)
    2. Wall-clock read / query time (seconds)
    3. Peak process RSS (Resident Set Size) during the operation (MB)

=============================================================================
"""

import gc
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import psutil

DATA_DIR   = Path("data")               # directory for generated files
RESULT_IMG = "benchmark_results.png"

# HITRAN fetch parameters  (CO2; broad enough range to get a meaningful dataset)
MOLECULE       = "CO2"
WAVENUM_MIN    = 500       # cm⁻¹  – lower bound
WAVENUM_MAX    = 10000     # cm⁻¹  – upper bound  (covers most IR bands)

HDF5_PATH    = DATA_DIR / "co2_hitran.hdf5"
FEATHER_PATH = DATA_DIR / "co2_hitran.feather"
PARQUET_PATH = DATA_DIR / "co2_hitran.parquet"


def get_rss_mb() -> float:
    """Return current process RSS (Resident Set Size) in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def file_size_mb(path: Path) -> float:
    """Return file size in MB."""
    return path.stat().st_size / (1024 ** 2)


def force_gc() -> None:
    """Force garbage collection and allow OS to reclaim memory."""
    gc.collect()
    gc.collect()


def fetch_real_hitran_data() -> pd.DataFrame:
    """
    Download real CO2 line-list data from the HITRAN database using
    ``radis.fetch_hitran``.  The data is cached locally by RADIS after the
    first download, so subsequent runs are fast.

    Key columns returned
    --------------------
    wav     : Transition wavenumber (cm⁻¹)
    int     : Line intensity (cm⁻¹/(molecule·cm⁻²))
    A       : Einstein-A coefficient (s⁻¹)
    airbrd  : Air-broadened half-width (cm⁻¹/atm)
    ... plus additional HITRAN parameters (iso, El, etc.)

    Returns
    -------
    pd.DataFrame
    """
    from radis import fetch_hitran

    print(f"\n{'='*60}")
    print(f"  Fetching real HITRAN data for {MOLECULE}")
    print(f"  Wavenumber range: {WAVENUM_MIN} – {WAVENUM_MAX} cm⁻¹")
    print(f"{'='*60}")

    df = fetch_hitran(
        MOLECULE,
        load_wavenum_min=WAVENUM_MIN,
        load_wavenum_max=WAVENUM_MAX,
        output="pandas",
        verbose=True,
    )

    # Ensure we have a plain Pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    print(f"\n  DataFrame shape : {df.shape}")
    print(f"  Columns         : {list(df.columns)}")
    print(f"  Memory usage    : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return df


def save_all_formats(df: pd.DataFrame) -> dict[str, float]:
    """
    Persist *df* as HDF5, Feather (IPC), and Parquet (snappy).
    Returns a dict of {format_name: size_in_MB}.
    """
    DATA_DIR.mkdir(exist_ok=True)
    sizes: dict[str, float] = {}

    # --- HDF5 (PyTables / tables) ---
    print("\n  Saving HDF5 …", end=" ", flush=True)
    df.to_hdf(str(HDF5_PATH), key="hitran", mode="w", format="table")
    sizes["HDF5"] = file_size_mb(HDF5_PATH)
    print(f"{sizes['HDF5']:.2f} MB")

    # --- Feather (Arrow IPC) ---
    print("  Saving Feather …", end=" ", flush=True)
    df.to_feather(str(FEATHER_PATH))
    sizes["Feather"] = file_size_mb(FEATHER_PATH)
    print(f"{sizes['Feather']:.2f} MB")

    # --- Parquet (snappy compression) ---
    print("  Saving Parquet (snappy) …", end=" ", flush=True)
    df.to_parquet(str(PARQUET_PATH), engine="pyarrow", compression="snappy")
    sizes["Parquet"] = file_size_mb(PARQUET_PATH)
    print(f"{sizes['Parquet']:.2f} MB")

    return sizes


def benchmark_eager_pandas(label: str, read_fn, *args) -> dict:
    """
    Time an *eager* Pandas read, computing max('int') after loading.
    Returns {'time_s': …, 'peak_rss_mb': …, 'result': …}.
    """
    force_gc()
    baseline_rss = get_rss_mb()

    start = time.perf_counter()
    df = read_fn(*args)
    result = df["int"].max()
    elapsed = time.perf_counter() - start

    peak_rss = get_rss_mb()
    del df
    force_gc()

    return {
        "label": label,
        "time_s": round(elapsed, 4),
        "peak_rss_mb": round(peak_rss - baseline_rss, 2),
        "result": result,
    }


def benchmark_lazy_polars(label: str, scan_fn, path: str, query_col: str = "int") -> dict:
    """
    Time a *lazy* Polars scan + aggregation (max of *query_col*).
    The LazyFrame is only materialised for the single aggregate value.
    """
    force_gc()
    baseline_rss = get_rss_mb()

    start = time.perf_counter()
    lazy_frame = scan_fn(path)
    result = lazy_frame.select(pl.col(query_col).max()).collect().item()
    elapsed = time.perf_counter() - start

    peak_rss = get_rss_mb()
    del lazy_frame
    force_gc()

    return {
        "label": label,
        "time_s": round(elapsed, 4),
        "peak_rss_mb": round(peak_rss - baseline_rss, 2),
        "result": result,
    }


def run_benchmarks() -> list[dict]:
    """Execute every benchmark combination and return a list of result dicts."""
    results: list[dict] = []

    print(f"\n{'='*60}")
    print("  Running benchmarks …")
    print(f"{'='*60}")

    # ---- Eager (Pandas) ----
    print("\n  [Pandas Eager] Reading HDF5 …")
    results.append(benchmark_eager_pandas(
        "Pandas + HDF5",
        pd.read_hdf, str(HDF5_PATH), "hitran",
    ))

    print("  [Pandas Eager] Reading Feather …")
    results.append(benchmark_eager_pandas(
        "Pandas + Feather",
        pd.read_feather, str(FEATHER_PATH),
    ))

    print("  [Pandas Eager] Reading Parquet …")
    results.append(benchmark_eager_pandas(
        "Pandas + Parquet",
        pd.read_parquet, str(PARQUET_PATH),
    ))

    # ---- Lazy (Polars) ----
    print("\n  [Polars Lazy] Scanning Feather (IPC) …")
    results.append(benchmark_lazy_polars(
        "Polars Lazy + Feather",
        pl.scan_ipc, str(FEATHER_PATH),
    ))

    print("  [Polars Lazy] Scanning Parquet …")
    results.append(benchmark_lazy_polars(
        "Polars Lazy + Parquet",
        pl.scan_parquet, str(PARQUET_PATH),
    ))

    return results


def print_results(sizes: dict[str, float], benchmarks: list[dict]) -> None:
    """Print a formatted summary table to the terminal."""
    print(f"\n{'='*60}")
    print("  FILE SIZES")
    print(f"{'='*60}")
    for fmt, mb in sizes.items():
        print(f"    {fmt:<12s}: {mb:>10.2f} MB")

    print(f"\n{'='*60}")
    print("  BENCHMARK RESULTS")
    print(f"{'='*60}")
    header = f"  {'Method':<26s} {'Time (s)':>10s} {'Peak ΔRSS (MB)':>16s}"
    print(header)
    print("  " + "-" * len(header.strip()))
    for b in benchmarks:
        print(f"  {b['label']:<26s} {b['time_s']:>10.4f} {b['peak_rss_mb']:>16.2f}")


def plot_results(sizes: dict[str, float], benchmarks: list[dict]) -> None:
    """
    Generate a three-panel bar chart comparing:
        (a) File sizes
        (b) Read / query times
        (c) Peak delta-RSS
    Saved as *RESULT_IMG*.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "RADIS Lazy-Loading Benchmark — Format & Engine Comparison",
        fontsize=14, fontweight="bold", y=1.02,
    )

    bar_color_map = {
        "HDF5":    "#4C72B0",
        "Feather": "#55A868",
        "Parquet": "#C44E52",
    }

    # ----- (a) File Sizes -----
    ax = axes[0]
    formats = list(sizes.keys())
    values  = list(sizes.values())
    colors  = [bar_color_map[f] for f in formats]
    bars = ax.bar(formats, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("File Size on Disk")
    ax.set_ylabel("Size (MB)")
    ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=3)

    # ----- Helper to colour benchmark bars -----
    def _bench_color(label: str) -> str:
        if "HDF5"    in label: return "#4C72B0"
        if "Feather" in label: return "#55A868"
        if "Parquet" in label: return "#C44E52"
        return "#8C8C8C"

    labels = [b["label"] for b in benchmarks]
    bench_colors = [_bench_color(l) for l in labels]

    # Add hatching for lazy (Polars) bars to distinguish engine
    hatch_patterns = ["" if "Pandas" in l else "///" for l in labels]

    # ----- (b) Read / Query Time -----
    ax = axes[1]
    times = [b["time_s"] for b in benchmarks]
    bars = ax.bar(range(len(labels)), times, color=bench_colors,
                  edgecolor="black", linewidth=0.5)
    for bar, hatch in zip(bars, hatch_patterns):
        bar.set_hatch(hatch)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_title("Read / Query Time")
    ax.set_ylabel("Time (s)")
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)

    # ----- (c) Peak ΔRSS -----
    ax = axes[2]
    rss = [b["peak_rss_mb"] for b in benchmarks]
    bars = ax.bar(range(len(labels)), rss, color=bench_colors,
                  edgecolor="black", linewidth=0.5)
    for bar, hatch in zip(bars, hatch_patterns):
        bar.set_hatch(hatch)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_title("Peak Additional RAM (ΔRSS)")
    ax.set_ylabel("Memory (MB)")
    ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)

    # Legend for engine type
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="white", edgecolor="black", label="Pandas (eager)"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="Polars (lazy)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    fig.savefig(RESULT_IMG, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved → {RESULT_IMG}")


def main() -> None:
    print("\n" + "=" * 60)
    print("  RADIS Lazy-Loading Benchmark")
    print("  Comparing HDF5 / Feather / Parquet × Pandas / Polars")
    print("=" * 60)

    # Step 1 – fetch real CO2 HITRAN data
    df = fetch_real_hitran_data()

    # Step 2 – persist in each format
    sizes = save_all_formats(df)

    # Free the original DataFrame before benchmarking
    del df
    force_gc()

    # Step 3 – run benchmarks
    benchmarks = run_benchmarks()

    # Step 4 – report
    print_results(sizes, benchmarks)
    plot_results(sizes, benchmarks)

    print(f"\n{'='*60}")
    print("  Done. See benchmark_results.png for the visual summary.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
