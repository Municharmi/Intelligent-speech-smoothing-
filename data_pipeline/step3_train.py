"""
data_pipeline/step3_train.py
Trains the severity classifier on combined dataset
(student + UASpeech + YouTube).

Run: python data_pipeline/step3_train.py \
         --data data/training_data.csv \
         --output models/severity_model.pkl
"""
import argparse, logging, os, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

# Feature columns available in the dataset
ACOUSTIC_FEATURES = [
    "pitch_mean_hz","pitch_std_hz","pitch_range_hz","voiced_ratio",
    "speech_rate_syl_per_sec","pause_ratio","pause_count","mean_pause_duration_s",
    "spectral_centroid_mean","spectral_centroid_std","spectral_flatness_mean",
    "zcr_mean","zcr_std","rms_mean","rms_std","energy_dynamic_range",
    "jitter_percent","shimmer_percent",
]
LINGUISTIC_FEATURES = [
    "utterance_length_words","type_token_ratio","mean_word_length",
    "verb_ratio","function_word_ratio","oov_rate",
]
MFCC_FEATURES = [f"mfcc_mean_{i}" for i in range(13)]
ALL_FEATURES  = ACOUSTIC_FEATURES + LINGUISTIC_FEATURES + MFCC_FEATURES

LABEL_MAP     = {"mild":2,"moderate":1,"severe":0}
LABEL_INVERSE = {v:k for k,v in LABEL_MAP.items()}


def load_features_from_audio(csv_path: str) -> pd.DataFrame:
    """
    Extract acoustic features from audio files listed in the CSV.
    Uses librosa + our FeatureExtractor.
    """
    logger.info("Extracting features from audio files...")

    from config import Config
    from core.audio_processor import AudioProcessor, AudioData
    from core.feature_extractor import FeatureExtractor

    df  = pd.read_csv(csv_path)
    df  = df[df["severity"].isin(["mild","moderate","severe"])].copy()
    cfg = Config()
    extractor = FeatureExtractor(cfg)
    processor = AudioProcessor(cfg)

    feature_rows = []
    total = len(df)

    for i, row in df.iterrows():
        fname      = row.get("filename","")
        transcript = str(row.get("transcript",""))
        severity   = row.get("severity","")

        print(f"\r  [{len(feature_rows)+1}/{total}] {os.path.basename(fname)[:40]}", end="", flush=True)

        if not os.path.exists(fname):
            # No audio — use linguistic features only
            feat = {"severity": severity, "source": row.get("source","")}
            for c in ALL_FEATURES: feat[c] = 0.0
            if transcript:
                words = transcript.lower().split()
                n = len(words)
                if n > 0:
                    FUNC = {"the","a","an","is","are","was","want","need","i","you","please"}
                    VERB = {"is","are","was","want","go","eat","drink","feel","need"}
                    feat["utterance_length_words"] = n
                    feat["type_token_ratio"]       = len(set(words))/n
                    feat["mean_word_length"]       = sum(len(w) for w in words)/n
                    feat["function_word_ratio"]    = sum(1 for w in words if w in FUNC)/n
                    feat["verb_ratio"]             = sum(1 for w in words if w in VERB)/n
            feature_rows.append(feat)
            continue

        try:
            audio_data = processor.load_and_preprocess(fname)
            features   = extractor.extract(audio_data, transcript=transcript)
            feat       = features.to_dict()

            # Flatten MFCC list
            mfcc = feat.pop("mfcc_mean", [])
            for idx, v in enumerate(mfcc[:13]):
                feat[f"mfcc_mean_{idx}"] = v
            mfcc_std = feat.pop("mfcc_std", [])
            feat.pop("mfcc_delta_mean", None)

            feat["severity"] = severity
            feat["source"]   = row.get("source","")
            feature_rows.append(feat)

        except Exception as e:
            logger.warning("Failed %s: %s", fname, e)
            feat = {"severity": severity, "source": row.get("source","")}
            for c in ALL_FEATURES: feat[c] = 0.0
            feature_rows.append(feat)

    print()
    return pd.DataFrame(feature_rows)


def train(csv_path: str, output_path: str, text_only: bool = False):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score, LeaveOneOut
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    print("="*55)
    print("  Training Severity Classifier")
    print("="*55)

    if text_only:
        logger.info("Text-only mode — using linguistic features only")
        df = pd.read_csv(csv_path)
        df = df[df["severity"].isin(["mild","moderate","severe"])].copy()
        # Build simple linguistic features
        rows = []
        for _, row in df.iterrows():
            words = str(row.get("transcript","")).lower().split()
            n = max(len(words), 1)
            FUNC = {"the","a","an","is","are","was","want","need","i","you","please"}
            VERB = {"is","are","was","want","go","eat","drink","feel","need"}
            rows.append({
                "utterance_length_words": n,
                "type_token_ratio":       len(set(words))/n,
                "mean_word_length":       sum(len(w) for w in words)/n,
                "function_word_ratio":    sum(1 for w in words if w in FUNC)/n,
                "verb_ratio":             sum(1 for w in words if w in VERB)/n,
                "oov_rate":               0.0,
                "severity":               row["severity"],
                "source":                 row.get("source",""),
            })
        feat_df = pd.DataFrame(rows)
        feature_cols = LINGUISTIC_FEATURES[:6]
    else:
        feat_df = load_features_from_audio(csv_path)
        feature_cols = [c for c in ALL_FEATURES if c in feat_df.columns]

    feat_df = feat_df.dropna(subset=["severity"])
    X = feat_df[feature_cols].fillna(0).values.astype(np.float32)
    y = feat_df["severity"].map(LABEL_MAP).values

    n       = len(X)
    counts  = {LABEL_INVERSE[k]:int(v) for k,v in zip(*np.unique(y,return_counts=True))}
    print(f"\n  Samples   : {n}")
    print(f"  Features  : {len(feature_cols)}")
    print(f"  Classes   : {counts}")

    if n < 6:
        print("  ERROR: Need at least 6 labelled samples to train. Label more recordings.")
        sys.exit(1)

    # SMOTE for class balance
    min_count = min(np.bincount(y))
    if min_count < 3 and n >= 9:
        try:
            from imblearn.over_sampling import SMOTE
            sm  = SMOTE(k_neighbors=min(min_count-1,2), random_state=42)
            X,y = sm.fit_resample(X,y)
            logger.info("SMOTE applied: %d → %d samples", n, len(X))
        except ImportError:
            logger.warning("imbalanced-learn not installed — skipping SMOTE")

    # Scale
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Choose CV strategy
    cv      = LeaveOneOut() if len(X) <= 20 else StratifiedKFold(n_splits=min(5,min(np.bincount(y))), shuffle=True, random_state=42)
    cv_name = "LOOCV" if len(X) <= 20 else "Stratified K-Fold"

    # Train Random Forest
    print(f"\n  Training Random Forest ({cv_name})...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=1,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring="accuracy")
    rf.fit(X_scaled, y)
    print(f"  RF  accuracy: {rf_scores.mean():.3f} ± {rf_scores.std():.3f}")

    # Train Gradient Boosting
    print("  Training Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42)
    gb_scores = cross_val_score(gb, X_scaled, y, cv=cv, scoring="accuracy")
    gb.fit(X_scaled, y)
    print(f"  GB  accuracy: {gb_scores.mean():.3f} ± {gb_scores.std():.3f}")

    # Pick best
    best       = rf if rf_scores.mean() >= gb_scores.mean() else gb
    best_name  = "Random Forest" if rf_scores.mean() >= gb_scores.mean() else "Gradient Boosting"
    best_score = max(rf_scores.mean(), gb_scores.mean())
    print(f"\n  Best model: {best_name} ({best_score:.3f})")

    # Calibrate
    n_per_class = min(np.bincount(y))
    if n_per_class >= 3:
        cal = CalibratedClassifierCV(best, cv=min(3,n_per_class), method="sigmoid")
        cal.fit(X_scaled, y)
        final = cal
    else:
        final = best

    # Evaluation
    print("\n  Classification Report (training set):")
    y_pred = best.predict(X_scaled)
    print(classification_report(y, y_pred, target_names=["severe","moderate","mild"], zero_division=0))

    # Feature importance
    if hasattr(best, "feature_importances_"):
        print("  Top 10 features:")
        imp = sorted(zip(feature_cols, best.feature_importances_), key=lambda x:-x[1])
        for feat, score in imp[:10]:
            bar = "█" * int(score*40)
            print(f"    {feat:<35} {score:.4f}  {bar}")

    # Save
    bundle = {
        "model":         final,
        "scaler":        scaler,
        "feature_cols":  feature_cols,
        "label_map":     LABEL_MAP,
        "label_inverse": LABEL_INVERSE,
        "model_name":    best_name,
        "cv_score":      float(best_score),
        "n_train":       int(len(X)),
        "text_only":     text_only,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n  Model saved → {output_path}")
    print(f"  CV accuracy : {best_score:.3f}")
    print(f"  Samples     : {len(X)}")
    print("\n  Restart Flask to load the new model: python app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      default="data/training_data.csv")
    parser.add_argument("--output",    default="models/severity_model.pkl")
    parser.add_argument("--text_only", action="store_true")
    args = parser.parse_args()
    train(args.data, args.output, args.text_only)
