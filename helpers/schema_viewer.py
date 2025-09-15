# schema_viewer.py
# Quick diagnostic: list schema + sample rows for each CSV

from pathlib import Path
import polars as pl
import json

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
OUTPUT_DIR = ROOT /"out" /"schema_debug"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_MAP = [
    ("train", DATA_ROOT / "0_Preliminary" / "0_Training"),
    ("test",  DATA_ROOT / "0_Preliminary" / "1_Submission"),
    ("val",   DATA_ROOT / "1_Final"),
]

def scan_and_describe(file: Path, split: str) -> dict:
    try:
        df = pl.read_csv(file.as_posix(), n_rows=50, infer_schema_length=1000)
    except Exception as e:
        return {"file": str(file), "split": split, "error": str(e)}

    schema = {col: str(df[col].dtype) for col in df.columns}
    head = df.head(5).to_dicts()
    return {
        "file": str(file),
        "split": split,
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "schema": schema,
        "sample": head,
    }

def print_head(lf: pl.LazyFrame, n: int = 10) -> None:
	print(lf.head(n).collect())

def main():
    report = []
    for split, base_dir in SPLIT_MAP:
        if not base_dir.exists():
            continue
        for f in sorted(base_dir.rglob("*.csv")):
            print(f"Scanning {f}")
            info = scan_and_describe(f, split)
            report.append(info)

    out_json = OUTPUT_DIR / "schema_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"Wrote schema report: {out_json}")

if __name__ == "__main__":
    main()
