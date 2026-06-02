import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import sys

def main():
    print("Loading data to train final model...")
    try:
        X_train = pd.read_csv('X_train.csv')
        y_train = pd.read_csv('y_train.csv').values.ravel()
    except Exception as e:
        print(f"Error loading datasets: {e}")
        sys.exit(1)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train best model
    print("Training XGBoost...")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train_scaled, y_train)

    print("Exporting model and scaler...")
    with open('xgboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print("Saved 'xgboost_model.pkl' and 'scaler.pkl' successfully.")

if __name__ == "__main__":
    main()
