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

Why Vaex must be replaced:
    - **Unmaintained**: Last significant update was in 2022
    - **Python 3.12+ incompatible**: Uses deprecated `imp` module (removed in 3.12)
    - **Installation failures**: Build errors on modern systems
    - Vaex benchmarks require Python ≤3.11 (run via Docker if needed)

Metrics captured:
    1. File size on disk (MB)
    2. Wall-clock read / query time (seconds)
    3. Peak process RSS (Resident Set Size) during the operation (MB)

=============================================================================
"""

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import psutil

DATA_DIR   = Path("data")               # directory for generated files
RESULT_IMG = "benchmark_results.png"

# HITRAN fetch parameters  (CO2; broad enough range to get a meaningful dataset)
MOLECULE       = "CO2"
WAVENUM_MIN    = 500       # cm⁻¹  – lower bound
WAVENUM_MAX    = 15000     # cm⁻¹  – upper bound  (covers IR + near-IR bands)

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
    """
    Force garbage collection and allow OS to reclaim memory.
    
    Multiple gc.collect() calls ensure cyclic references are cleaned up.
    The short sleep gives the OS time to actually reclaim the memory pages,
    which helps prevent negative delta RSS values in subsequent benchmarks.
    """
    gc.collect()
    gc.collect()
    gc.collect()
    # Brief pause to allow OS to reclaim memory pages
    time.sleep(0.1)


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


# SUBPROCESS BENCHMARK WORKER CODE
# This code is executed in isolated subprocesses to get accurate memory
# measurements. Each benchmark runs in a fresh Python process, eliminating
# memory cross-contamination between benchmarks.

BENCHMARK_WORKER_CODE = '''
import gc
import json
import os
import sys
import time

import psutil


def get_rss_mb() -> float:
    """Return current process RSS in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def force_gc() -> None:
    """Force garbage collection."""
    gc.collect()
    gc.collect()
    gc.collect()


def run_benchmark(engine: str, file_format: str, path: str, query_col: str = "int") -> dict:
    """
    Run a single benchmark in this isolated process.
    
    Parameters
    ----------
    engine : str
        One of: "pandas", "polars", "dask", "duckdb", "vaex"
    file_format : str
        One of: "hdf5", "feather", "parquet"
    path : str
        Path to the data file
    query_col : str
        Column to compute max() on
    
    Returns
    -------
    dict with keys: label, time_s, peak_rss_mb, result
    """
    # Clean slate before measuring baseline
    force_gc()
    baseline_rss = get_rss_mb()
    
    start = time.perf_counter()
    
    if engine == "pandas":
        import pandas as pd
        if file_format == "hdf5":
            df = pd.read_hdf(path, "hitran")
        elif file_format == "feather":
            df = pd.read_feather(path)
        elif file_format == "parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported format for Pandas: {file_format}")
        result = float(df[query_col].max())
        del df
    
    elif engine == "polars":
        import polars as pl
        if file_format == "feather":
            lf = pl.scan_ipc(path)
        elif file_format == "parquet":
            lf = pl.scan_parquet(path)
        else:
            raise ValueError(f"Unsupported format for Polars lazy: {file_format}")
        result = float(lf.select(pl.col(query_col).max()).collect().item())
        del lf
    
    elif engine == "dask":
        import dask.dataframe as dd
        import pandas as pd
        if file_format == "parquet":
            ddf = dd.read_parquet(path)
        elif file_format == "feather":
            # Dask doesn't have native feather support
            ddf = dd.from_pandas(pd.read_feather(path), npartitions=4)
        else:
            raise ValueError(f"Unsupported format for Dask: {file_format}")
        result = float(ddf[query_col].max().compute())
        del ddf
    
    elif engine == "duckdb":
        import duckdb
        con = duckdb.connect(":memory:")
        if file_format == "parquet":
            query = f"SELECT MAX({query_col}) FROM read_parquet(\\'{path}\\')"
        elif file_format == "feather":
            query = f"SELECT MAX({query_col}) FROM \\'{path}\\'"
        else:
            raise ValueError(f"Unsupported format for DuckDB: {file_format}")
        result = float(con.execute(query).fetchone()[0])
        con.close()
    
    # NOTE: Vaex support removed - incompatible with Python 3.12+
    # (uses deprecated `imp` module). This is why we need a replacement.
    
    else:
        raise ValueError(f"Unknown engine: {engine}")
    
    elapsed = time.perf_counter() - start
    peak_rss = get_rss_mb()
    
    force_gc()
    
    return {
        "time_s": round(elapsed, 4),
        "peak_rss_mb": round(peak_rss - baseline_rss, 2),
        "result": result,
    }


if __name__ == "__main__":
    # Args: engine, format, path, query_col
    engine = sys.argv[1]
    file_format = sys.argv[2]
    path = sys.argv[3]
    query_col = sys.argv[4] if len(sys.argv) > 4 else "int"
    
    result = run_benchmark(engine, file_format, path, query_col)
    print(json.dumps(result))
'''


def run_benchmark_subprocess(
    label: str,
    engine: str,
    file_format: str,
    path: str,
    query_col: str = "int",
) -> dict:
    """
    Run a benchmark in an isolated subprocess for accurate memory measurement.
    
    Each benchmark runs in a completely fresh Python process, eliminating
    memory cross-contamination that causes negative ΔRSS values.
    
    Parameters
    ----------
    label : str
        Human-readable name for this benchmark
    engine : str
        One of: "pandas", "polars", "dask", "duckdb"
    file_format : str
        One of: "hdf5", "feather", "parquet"
    path : str
        Path to the data file
    query_col : str
        Column to compute max() on
    
    Returns
    -------
    dict with keys: label, time_s, peak_rss_mb, result
    """
    # Write worker code to a temporary file
    worker_path = Path(__file__).parent / "_benchmark_worker.py"
    worker_path.write_text(BENCHMARK_WORKER_CODE)
    
    try:
        result = subprocess.run(
            [sys.executable, str(worker_path), engine, file_format, path, query_col],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"    ERROR: {result.stderr}", file=sys.stderr)
            return {
                "label": label,
                "time_s": -1,
                "peak_rss_mb": -1,
                "result": None,
                "error": result.stderr,
            }
        
        data = json.loads(result.stdout.strip())
        data["label"] = label
        return data
        
    finally:
        # Cleanup temp worker file
        if worker_path.exists():
            worker_path.unlink()


def run_benchmarks() -> list[dict]:
    """
    Execute every benchmark combination in isolated subprocesses.
    
    Each benchmark runs in a fresh Python process to ensure accurate
    memory measurements without cross-contamination between benchmarks.
    """
    results: list[dict] = []

    print(f"\n{'='*60}")
    print("  Running benchmarks (subprocess isolation for accurate memory)")
    print(f"{'='*60}")

    # ---- Eager (Pandas) ----
    print("\n  [Pandas Eager] Reading HDF5 …")
    results.append(run_benchmark_subprocess(
        "Pandas + HDF5", "pandas", "hdf5", str(HDF5_PATH),
    ))

    print("  [Pandas Eager] Reading Feather …")
    results.append(run_benchmark_subprocess(
        "Pandas + Feather", "pandas", "feather", str(FEATHER_PATH),
    ))

    print("  [Pandas Eager] Reading Parquet …")
    results.append(run_benchmark_subprocess(
        "Pandas + Parquet", "pandas", "parquet", str(PARQUET_PATH),
    ))

    # ---- Lazy (Polars) ----
    print("\n  [Polars Lazy] Scanning Feather (IPC) …")
    results.append(run_benchmark_subprocess(
        "Polars Lazy + Feather", "polars", "feather", str(FEATHER_PATH),
    ))

    print("  [Polars Lazy] Scanning Parquet …")
    results.append(run_benchmark_subprocess(
        "Polars Lazy + Parquet", "polars", "parquet", str(PARQUET_PATH),
    ))

    # ---- Lazy (Dask) ----
    print("\n  [Dask Lazy] Reading Parquet …")
    results.append(run_benchmark_subprocess(
        "Dask Lazy + Parquet", "dask", "parquet", str(PARQUET_PATH),
    ))

    # ---- Lazy (DuckDB) ----
    print("\n  [DuckDB Lazy] Reading Parquet …")
    results.append(run_benchmark_subprocess(
        "DuckDB + Parquet", "duckdb", "parquet", str(PARQUET_PATH),
    ))

    # ---- Vaex (the library we want to replace) ----
    # NOTE: Vaex is incompatible with Python 3.12+ (uses deprecated `imp` module)
    # This incompatibility is one of the key reasons we need to replace Vaex
    # To benchmark Vaex, use Python ≤3.11 via Docker

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
        (c) Peak delta-RSS (memory usage)
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

    # Add hatching to distinguish engines
    def _get_hatch(label: str) -> str:
        if "Pandas" in label:
            return ""          # No hatch for Pandas (eager)
        elif "Polars" in label:
            return "///"       # Diagonal lines for Polars
        elif "Dask" in label:
            return "xxx"       # Cross-hatch for Dask
        elif "DuckDB" in label:
            return "..."       # Dots for DuckDB
        return ""

    hatch_patterns = [_get_hatch(l) for l in labels]

    # ----- (b) Read / Query Time -----
    ax = axes[1]
    times = [b["time_s"] for b in benchmarks]
    bars = ax.bar(range(len(labels)), times, color=bench_colors,
                  edgecolor="black", linewidth=0.5)
    for bar, hatch in zip(bars, hatch_patterns):
        bar.set_hatch(hatch)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_title("Read / Query Time")
    ax.set_ylabel("Time (s)")
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=3)

    # ----- (c) Peak ΔRSS -----
    ax = axes[2]
    rss = [b["peak_rss_mb"] for b in benchmarks]
    bars = ax.bar(range(len(labels)), rss, color=bench_colors,
                  edgecolor="black", linewidth=0.5)
    for bar, hatch in zip(bars, hatch_patterns):
        bar.set_hatch(hatch)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_title("Peak Additional RAM ΔRSS")
    ax.set_ylabel("Memory (MB)")
    ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=3)

    # Legend for engine type
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="white", edgecolor="black", label="Pandas (eager)"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="Polars (lazy)"),
        Patch(facecolor="white", edgecolor="black", hatch="xxx", label="Dask (lazy)"),
        Patch(facecolor="white", edgecolor="black", hatch="...", label="DuckDB (lazy)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    fig.savefig(RESULT_IMG, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved → {RESULT_IMG}")


def main() -> None:
    print("\n" + "=" * 70)
    print("  RADIS Lazy-Loading Benchmark")
    print("  Comparing: Pandas / Polars / Dask / DuckDB")
    print("  (Vaex excluded - incompatible with Python 3.12+)")
    print("=" * 70)

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
