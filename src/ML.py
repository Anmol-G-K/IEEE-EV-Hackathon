# ids_can_ids.py
"""
Standalone IDS script: Random Forest & XGBoost on CAN bus CSV dataset.

Usage:
    python ids_can_ids.py             # Uses default dataset paths from EDA script
    python ids_can_ids.py file.csv    # Uses single file provided
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ------------------------- Paths from EDA -------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"

SPLIT_MAP = [
    ("train", DATA_ROOT / "0_Preliminary" / "0_Training"),
    ("test",  DATA_ROOT / "0_Preliminary" / "1_Submission"),
    ("val",   DATA_ROOT / "1_Final"),
]

# ------------------------- Payload utilities -------------------------
def parse_payload_hex_to_bytes(hexstr: str):
    if pd.isna(hexstr):
        return [0]*8
    s = str(hexstr).replace("0x","").replace(" ","").strip()
    if len(s) % 2 != 0:
        s = s[:-1]
    try:
        bytes_list = [int(s[i:i+2],16) for i in range(0, min(len(s),16), 2)]
        if len(bytes_list) < 8:
            bytes_list += [0]*(8-len(bytes_list))
        return bytes_list[:8]
    except Exception:
        parts = [p.strip() for p in str(hexstr).replace("[","").replace("]","").split(",")]
        vals = []
        for p in parts:
            try:
                vals.append(int(p,0))
            except:
                try:
                    vals.append(int(p))
                except:
                    vals.append(0)
        vals += [0]*(8-len(vals))
        return vals[:8]

def payload_entropy(payload):
    if len(payload) == 0:
        return 0.0
    vals, counts = np.unique(payload, return_counts=True)
    probs = counts / counts.sum()
    ent = -np.sum(probs * np.log2(probs + 1e-12))
    return float(ent)

# ------------------------- Feature extraction -------------------------
def extract_frame_features(df: pd.DataFrame):
    df = df.copy()
    # timestamp handling
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["Timestamp"] = df["Timestamp"].fillna(method="ffill").fillna(pd.Timestamp.now())
        times_ms = (df["Timestamp"] - df["Timestamp"].min()).dt.total_seconds() * 1000.0
    else:
        df["__ts"] = pd.date_range(start=pd.Timestamp.now(), periods=len(df), freq="ms")
        times_ms = (df["__ts"] - df["__ts"].min()).dt.total_seconds() * 1000.0

    print("[DEBUG] Extracting payload features...")
    payloads = []
    for _, row in df.iterrows():
        if any([f"data{i}" in df.columns for i in range(8)]):
            b = [int(row.get(f"data{i}",0)) for i in range(8)]
            payloads.append(b)
        elif "Data" in df.columns:
            payloads.append(parse_payload_hex_to_bytes(row["Data"]))
        elif "payload" in df.columns:
            payloads.append(parse_payload_hex_to_bytes(row["payload"]))
        else:
            payloads.append([0]*8)

    payload_df = pd.DataFrame(payloads, columns=[f"byte_{i}" for i in range(8)], index=df.index)
    features = pd.DataFrame(index=df.index)

    # CAN ID
    if "CAN ID" in df.columns:
        def canid_to_int(x):
            try:
                xstr = str(x)
                if xstr.startswith("0x") or xstr.startswith("0X"):
                    return int(xstr,16)
                return int(x)
            except:
                return 0
        features["can_id_int"] = df["CAN ID"].apply(canid_to_int)
    elif "id" in df.columns:
        features["can_id_int"] = df["id"].astype(int)
    else:
        features["can_id_int"] = 0

    features["dlc"] = pd.to_numeric(df.get("DLC", 8), errors="coerce").fillna(8).astype(int)

    for i in range(8):
        features[f"byte_{i}"] = payload_df[f"byte_{i}"].astype(int)

    features["payload_sum"] = payload_df.sum(axis=1)
    features["payload_std"] = payload_df.std(axis=1).fillna(0)
    features["payload_entropy"] = payload_df.apply(lambda r: payload_entropy(r.values.astype(int)), axis=1)

    # timing features
    features["delta_t"] = times_ms.diff().fillna(0).abs()
    last_ts = {}
    delta_per_id = []
    for idx, row in df.iterrows():
        can = row.get("CAN ID", row.get("id", None))
        ts = times_ms.iloc[idx]
        prev = last_ts.get(can, None)
        delta_per_id.append(0.0 if prev is None else abs(ts - prev))
        last_ts[can] = ts
    features["delta_per_id"] = delta_per_id
    features["freq_est"] = 1.0 / (features["delta_per_id"] + 1e-6)

    print("[DEBUG] Feature extraction complete.")
    return features

# ------------------------- Preprocess -------------------------
def preprocess_data(df_feat_with_label: pd.DataFrame, label_col="Label"):
    X = df_feat_with_label.drop(columns=[label_col])
    y_raw = df_feat_with_label[label_col].astype(str)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler, label_encoder

# ------------------------- Models -------------------------
def train_random_forest(X_train, X_test, y_train, y_test, label_encoder=None):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    labels = list(label_encoder.classes_) if label_encoder else None
    report = classification_report(y_test, y_pred, target_names=labels)
    cm = confusion_matrix(y_test, y_pred)
    return clf, report, cm

def train_xgboost(X_train, X_test, y_train, y_test, label_encoder=None):
    clf = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    labels = list(label_encoder.classes_) if label_encoder else None
    report = classification_report(y_test, y_pred, target_names=labels)
    cm = confusion_matrix(y_test, y_pred)
    return clf, report, cm

# ------------------------- Main -------------------------
def process_file(file_path: Path):
    print(f"[INFO] Processing file: {file_path}")
    df_raw = pd.read_csv(file_path)
    print(f"[DEBUG] Loaded data: {df_raw.shape}")

    features_df = extract_frame_features(df_raw)

    # Use "attack_type" or last column as label
    label_col = "attack_type" if "attack_type" in df_raw.columns else df_raw.columns[-1]
    print(f"[DEBUG] Using label column: {label_col}")

    df_feat_with_label = features_df.copy()
    df_feat_with_label["Label"] = df_raw[label_col].astype(str).values[:len(features_df)]

    print(f"[INFO] Label distribution:\n{df_feat_with_label['Label'].value_counts()}")

    X_train, X_test, y_train, y_test, scaler, label_encoder = preprocess_data(df_feat_with_label)

    print("\n=== Random Forest ===")
    rf_model, rf_report, rf_cm = train_random_forest(X_train, X_test, y_train, y_test, label_encoder)
    print("Classification report:\n", rf_report)
    print("Confusion matrix:\n", rf_cm)

    print("\n=== XGBoost ===")
    xgb_model, xgb_report, xgb_cm = train_xgboost(X_train, X_test, y_train, y_test, label_encoder)
    print("Classification report:\n", xgb_report)
    print("Confusion matrix:\n", xgb_cm)


def get_all_csv_files_from_split_map(split_map):
    all_files = []
    for split, base_path in split_map:
        if not base_path.exists():
            print(f"[WARNING] Split path not found: {base_path}")
            continue
        files = sorted(base_path.rglob("*.csv"))
        print(f"[INFO] Found {len(files)} CSVs in {split}: {base_path}")
        for f in files:
            all_files.append((split, f))
    return all_files


def main():
    if len(sys.argv) == 2:
        # Single file mode
        file_arg = sys.argv[1]
        file_path = Path(file_arg).resolve()
        if not file_path.exists():
            print(f"[ERROR] File not found: {file_path}")
            sys.exit(1)
        process_file(file_path)
    else:
        # Batch mode: walk all known dataset splits
        all_csvs = get_all_csv_files_from_split_map(SPLIT_MAP)

        if not all_csvs:
            print("[ERROR] No CSV files found in dataset splits.")
            sys.exit(1)

        for split, csv_path in all_csvs:
            print(f"\n=== Processing [{split}] - {csv_path.name} ===")
            try:
                process_file(csv_path)
            except Exception as e:
                print(f"[ERROR] Failed to process {csv_path.name}: {e}")


if __name__ == "__main__":
    main()
