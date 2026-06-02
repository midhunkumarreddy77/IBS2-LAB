import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings

# Ignore some LightGBM warnings for cleaner output
warnings.filterwarnings("ignore")

def main():
    print("Loading datasets...")
    try:
        X_train = pd.read_csv('X_train.csv')
        y_train = pd.read_csv('y_train.csv').values.ravel()
        X_test = pd.read_csv('X_test.csv')
        y_test = pd.read_csv('y_test.csv').values.ravel()
    except Exception as e:
        print(f"Error loading datasets: {e}")
        sys.exit(1)

    print("Scaling features...")
    # SVM and many other models perform better with scaled features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1)
    }

    results = []

    print("\nTraining and Evaluating Models:")
    print("-" * 40)
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"{name} Results:")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1 Score : {f1:.4f}\n")
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "F1 Score": f1
        })

    # Display Summary
    print("-" * 40)
    print("Summary of Results:")
    summary_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
    print(summary_df.to_string(index=False))
    
    # Save results to CSV
    summary_df.to_csv("model_evaluation_results.csv", index=False)
    print("\nResults saved to 'model_evaluation_results.csv'")

if __name__ == "__main__":
    main()
