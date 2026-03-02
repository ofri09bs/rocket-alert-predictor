import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import joblib

# Load and clean raw alert dataset
def load_and_clean_data():
    columns = ["Time", "Location", "IsDrill", "id", "Threat", "Extra"]
    data = pd.read_csv("alarms_updated.csv", names=columns, skiprows=1, index_col=False)
    data["Time"] = pd.to_datetime(data["Time"], errors="coerce")
    data.dropna(subset=["Time", "Location"], inplace=True)
    return data

def train_undersampled_master_model():
    data = load_and_clean_data()
    
    # 1. Encode Locations
    encoder = LabelEncoder()
    data["Location_Encoded"] = encoder.fit_transform(data["Location"])

    # 2. Geographic Profiling
    loc_counts = data["Location_Encoded"].value_counts().to_dict()
    total_alerts = len(data)
    loc_risk_ratio = {k: v / total_alerts for k, v in loc_counts.items()}
    
    data["total_counts"] = data["Location_Encoded"].map(loc_counts)
    data["loc_risk_ratio"] = data["Location_Encoded"].map(loc_risk_ratio)
    data["is_rare"] = data["Location_Encoded"].apply(lambda x: 1 if loc_counts.get(x, 0) < 15 else 0)

    # 3. Time Series & Tactical Features
    data = data.sort_values(["Location_Encoded", "Time"])
    
    data["min_since_last"] = data.groupby("Location_Encoded")["Time"].diff().dt.total_seconds() / 60.0
    data["min_since_last"] = data["min_since_last"].fillna(-1)
    
    data["prev_gap"] = data.groupby("Location_Encoded")["min_since_last"].shift(1)
    data["acceleration"] = data["min_since_last"] - data["prev_gap"]
    data["acceleration"] = data["acceleration"].fillna(0)

    # National Context (Rolling 1h Load)
    data = data.sort_values("Time")
    data.set_index("Time", inplace=True)
    data["country_load_1h"] = data["Location_Encoded"].rolling("1h").count()
    data.reset_index(inplace=True)
    
    data["localized_impact"] = data["country_load_1h"] * data["loc_risk_ratio"]

    # Local 24h Load
    data = data.sort_values(["Location_Encoded", "Time"])
    data.set_index("Time", inplace=True)
    data["local_alerts_24h"] = data.groupby("Location_Encoded")["Location_Encoded"].rolling("24h").count().values
    data.reset_index(inplace=True)

    # Cyclical Time Features
    data["Hour_sin"] = np.sin(2 * np.pi * data["Time"].dt.hour / 24.0)
    data["Hour_cos"] = np.cos(2 * np.pi * data["Time"].dt.hour / 24.0)

    # 4. Target Formulation
    data["next_alert"] = data.groupby("Location_Encoded")["Time"].shift(-1)
    data["Y_min_to_next"] = (data["next_alert"] - data["Time"]).dt.total_seconds() / 60.0
    
    data = data.dropna(subset=["Y_min_to_next"])
    
    # Filter dataset to a 24-hour tactical window
    train_set = data[data["Y_min_to_next"] <= 1440].copy()
    
    # Classification Target: Event within 6 hours
    train_set["Y_binary_6h"] = (train_set["Y_min_to_next"] <= 360).astype(int)

    features = [
        "Location_Encoded", "Hour_sin", "Hour_cos", "min_since_last", 
        "acceleration", "country_load_1h", "localized_impact", 
        "local_alerts_24h", "is_rare", "total_counts"
    ]

    X = train_set[features]
    y_clf = train_set["Y_binary_6h"]
    y_reg = train_set["Y_min_to_next"]

    # 5. Split for Evaluation (Chronological split to prevent data leakage)
    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=42, shuffle=False 
    )

    # --- 6. SMART UNDERSAMPLING (Applying only to Training Set) ---
    print("Performing Undersampling on the training data to prevent overfitting...")
    train_clf_df = pd.concat([X_train, y_clf_train], axis=1)
    
    # Identify majority (alerts within 6h) and minority (quiet periods)
    majority = train_clf_df[train_clf_df["Y_binary_6h"] == 1]
    minority = train_clf_df[train_clf_df["Y_binary_6h"] == 0]
    
    # Downsample the majority class to match the minority class exactly
    majority_downsampled = resample(majority, 
                                    replace=False, 
                                    n_samples=len(minority), 
                                    random_state=42)
    
    # Recombine and shuffle
    balanced_train = pd.concat([majority_downsampled, minority]).sample(frac=1, random_state=42)
    
    X_clf_train_balanced = balanced_train[features]
    y_clf_train_balanced = balanced_train["Y_binary_6h"]

    print(f"Classifier Train Set reduced to: {len(X_clf_train_balanced)} perfectly balanced samples.")

    # 7. Train Models
    print("Training Undersampled Classifier...")
    clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    clf.fit(X_clf_train_balanced, y_clf_train_balanced)

    print("Training Full Regressor...")
    reg = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_reg_train)

    # 8. EVALUATION BLOCK
    print("\n" + "="*40)
    print("--- UNDERSAMPLED MODEL EVALUATION RESULTS ---")
    
    # Evaluate on the natural, un-tampered test set
    clf_preds = clf.predict(X_test)
    print("\nClassifier Performance (Natural Test Set):")
    print(classification_report(y_clf_test, clf_preds))
    
    reg_preds = reg.predict(X_test)
    mae = mean_absolute_error(y_reg_test, reg_preds)
    print(f"\nRegressor Mean Absolute Error (MAE): {mae:.2f} minutes")
    print("="*40 + "\n")

    # 9. Final Train on Full Undersampled Data for Export
    print("Preparing final export models...")
    full_df = pd.concat([X, y_clf], axis=1)
    majority_full = full_df[full_df["Y_binary_6h"] == 1]
    minority_full = full_df[full_df["Y_binary_6h"] == 0]
    
    majority_downsampled_full = resample(majority_full, replace=False, n_samples=len(minority_full), random_state=42)
    balanced_full = pd.concat([majority_downsampled_full, minority_full]).sample(frac=1, random_state=42)
    
    clf.fit(balanced_full[features], balanced_full["Y_binary_6h"])
    reg.fit(X, y_reg) # Regressor learns from everything

    # Export Artifacts
    last_alert_memory = data.groupby("Location_Encoded")["Time"].max().to_dict()
    last_gap_memory = data.groupby("Location_Encoded")["min_since_last"].last().to_dict()

    joblib.dump(clf, "alert_classifier.joblib", compress=9)
    joblib.dump(reg, "alert_regressor.joblib", compress=9)
    joblib.dump(encoder, "location_encoder.joblib")
    joblib.dump({
        "time": last_alert_memory,
        "gap": last_gap_memory,
        "counts": loc_counts,
        "ratios": loc_risk_ratio
    }, "system_memory.joblib")
    joblib.dump(features, "feature_names.joblib")

    print("Success: Final Anti-Yes-Man Engine Saved.")

if __name__ == "__main__":
    train_undersampled_master_model()