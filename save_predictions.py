import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import sys

def main():
    print("Loading data...")
    try:
        X_train = pd.read_csv('X_train.csv')
        y_train = pd.read_csv('y_train.csv').values.ravel()
        X_test = pd.read_csv('X_test.csv')
        y_test = pd.read_csv('y_test.csv').values.ravel()
    except Exception as e:
        print(f"Error loading datasets: {e}")
        sys.exit(1)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train best model
    print("Training XGBoost...")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predict
    print("Making predictions on the test set...")
    predictions = model.predict(X_test_scaled)

    # Create a DataFrame to compare
    results_df = pd.DataFrame({
        'Actual_Activity': y_test,
        'Predicted_Activity': predictions
    })

    print("Saving predictions to 'xgboost_predictions.csv'...")
    results_df.to_csv('xgboost_predictions.csv', index=False)
    print("Done! You can now view the predictions.")

if __name__ == "__main__":
    main()
