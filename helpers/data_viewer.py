# eda_car_hacking_polars_splits.py
# Enhanced EDA for CAN dataset using Polars
# Produces LLM-friendly JSON + CSV outputs with schema-safe concat

import os
import json
from pathlib import Path
import polars as pl

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
OUTPUT_DIR = ROOT / "eda_out"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JSON_OUT = OUTPUT_DIR / "eda_summary.json"

# ---------------- Dataset mapping ----------------
SPLIT_MAP = [
    ("train", DATA_ROOT / "0_Preliminary" / "0_Training"),
    ("test",  DATA_ROOT / "0_Preliminary" / "1_Submission"),
    ("val",   DATA_ROOT / "1_Final"),
]

EXPECTED_COLS = ["Timestamp", "Arbitration_ID", "DLC", "Data", "Class", "SubClass"]

# ---------------- Helpers ----------------
def scan_split(split: str, base_dir: Path) -> pl.LazyFrame:
    """
    Scan all CSVs under base_dir, enforce consistent schema.
    """
    if not base_dir.exists():
        return pl.LazyFrame(schema={c: pl.Utf8 for c in EXPECTED_COLS}).with_columns(
            pl.lit(split).alias("split"),
            pl.lit("").alias("source_file"),
            pl.lit("").alias("source_dir"),
        )

    files = sorted(base_dir.rglob("*.csv"))
    if not files:
        return pl.LazyFrame(schema={c: pl.Utf8 for c in EXPECTED_COLS}).with_columns(
            pl.lit(split).alias("split"),
            pl.lit("").alias("source_file"),
            pl.lit("").alias("source_dir"),
        )

    lfs = []
    for f in files:
        lf = pl.scan_csv(
            f.as_posix(),
            has_header=True,
            ignore_errors=True,
            infer_schema_length=5000,
        )
        # Ensure all expected cols exist
        for c in EXPECTED_COLS:
            if c not in lf.columns:
                lf = lf.with_columns(pl.lit(None).cast(pl.Utf8).alias(c))

        lf = lf.with_columns(
            pl.lit(split).alias("split"),
            pl.lit(f.name).alias("source_file"),
            pl.lit(str(f.parent)).alias("source_dir"),
        )
        lfs.append(lf)

    return pl.concat(lfs, how="vertical_relaxed")

def normalize(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = lf.with_columns(
        [
            pl.col("Timestamp").cast(pl.Float64, strict=False),
            pl.col("Arbitration_ID").cast(pl.Utf8, strict=False),
            pl.col("DLC").cast(pl.Int64, strict=False),
            pl.col("Data").cast(pl.Utf8, strict=False),
            pl.col("Class").cast(pl.Utf8, strict=False),
            pl.col("SubClass").cast(pl.Utf8, strict=False),
        ]
    ).with_columns(
        [
            pl.when(pl.col("Data").is_not_null())
              .then(pl.col("Data").str.replace_all(r"[^0-9A-Fa-f]", "").str.to_uppercase())
              .otherwise(pl.lit(None))
              .alias("Data"),
        ]
    ).with_columns(
        [
            pl.when(pl.col("Data").is_not_null())
              .then((pl.col("Data").str.len_chars() // 2).cast(pl.Int64))
              .otherwise(pl.lit(None, dtype=pl.Int64))
              .alias("data_len_bytes")
        ]
    )

    lf = (
        lf.group_by(["split", "source_dir", "source_file"])
          .agg([pl.all(), pl.col("Timestamp").min().alias("_ts0")])
          .explode(pl.exclude("_ts0"))
          .with_columns((pl.col("Timestamp") - pl.col("_ts0")).alias("ts_rel_s"))
          .drop("_ts0")
    )
    return lf

def write_csv(df: pl.DataFrame, name: str):
    out = OUTPUT_DIR / name
    df.write_csv(out.as_posix())

# ---------------- EDA aggregations ----------------
def compute_basic(lf: pl.LazyFrame) -> dict:
    total_rows = lf.select(pl.len().alias("rows")).collect().item(0, 0)
    per_split = (
        lf.group_by("split").agg(pl.len().alias("rows"))
          .sort("rows", descending=True).collect().to_dict(as_series=False)
    )
    return {
        "total_rows": int(total_rows),
        "per_split": [{"split": s, "rows": int(r)} for s, r in zip(per_split["split"], per_split["rows"])],
    }

def compute_schema_sample(lf: pl.LazyFrame) -> dict:
    schema = {k: str(v) for k, v in lf.schema.items()}
    head = lf.head(20).collect().to_dicts()
    write_csv(lf.head(200).collect(), "sample_head.csv")
    return {"schema": schema, "sample_head": head}

def compute_missingness(lf: pl.LazyFrame) -> dict:
    miss = lf.select(
        [
            pl.len().alias("rows"),
            *[pl.sum(pl.col(c).is_null().cast(pl.Int64)).alias(f"missing_{c}") for c in EXPECTED_COLS]
        ]
    ).collect().to_dicts()[0]
    return {"overall": miss}

def compute_class_dist(lf: pl.LazyFrame) -> dict:
    cls = lf.group_by("Class").agg(pl.len().alias("count")).sort("count", descending=True).collect()
    sub = lf.group_by(["Class", "SubClass"]).agg(pl.len().alias("count")).sort(["Class", "count"], descending=[True, True]).collect()
    return {
        "overall": cls.to_dict(as_series=False),
        "subclass": sub.to_dict(as_series=False),
    }

def compute_id_stats(lf: pl.LazyFrame, top_k: int = 50) -> dict:
    top_ids_overall = (
        lf.group_by("Arbitration_ID").agg(pl.len().alias("count"))
          .sort("count", descending=True).head(top_k).collect()
    )
    return {"top_overall": top_ids_overall.to_dict(as_series=False)}

def compute_payload_dlc(lf: pl.LazyFrame) -> dict:
    dlc_dist = lf.group_by("DLC").agg(pl.len().alias("count")).sort("DLC").collect()
    data_len_dist = lf.group_by("data_len_bytes").agg(pl.len().alias("count")).sort("data_len_bytes").collect()
    return {
        "dlc_distribution": dlc_dist.to_dict(as_series=False),
        "data_len_distribution": data_len_dist.to_dict(as_series=False),
    }

def compute_time_activity(lf: pl.LazyFrame) -> dict:
    mps_split = (
        lf.with_columns(pl.col("ts_rel_s").floor().alias("t_s"))
          .group_by(["split", "t_s"]).agg(pl.len().alias("msgs"))
          .sort(["split", "t_s"]).collect()
    )
    return {"messages_per_second_by_split": mps_split.head(2000).to_dict(as_series=False)}

# ---------------- Main ----------------
def main():
    lfs = [scan_split(split, base) for split, base in SPLIT_MAP]
    lf = pl.concat(lfs, how="vertical_relaxed")
    lf = normalize(lf)

    summary = {
        "dataset": {
            "name": "Car Hacking: Attack & Defense Challenge 2020",
            "source_url": "https://ieee-dataport.org/open-access/car-hacking-attack-defense-challenge-2020-dataset",
            "splits": [{"split": s, "path": str(p)} for s, p in SPLIT_MAP],
            "expected_columns": EXPECTED_COLS,
        },
        "overview": compute_basic(lf),
        "schema_and_sample": compute_schema_sample(lf),
        "missingness": compute_missingness(lf),
        "class_distribution": compute_class_dist(lf),
        "arbitration_id_stats": compute_id_stats(lf, top_k=100),
        "payload_and_dlc": compute_payload_dlc(lf),
        "time_activity": compute_time_activity(lf),
    }

    JSON_OUT.write_text(json.dumps(summary, indent=2))
    print(f"Wrote JSON: {JSON_OUT}")
    print(f"CSV artifacts: {OUTPUT_DIR}")

if __name__ == "__main__":
    pl.Config.set_tbl_rows(50)
    pl.Config.set_fmt_str_lengths(200)
    main()
