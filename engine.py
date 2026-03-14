import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_and_save():
    print("Step 1: Generating Synthetic Session Data...")
    np.random.seed(42)
    n = 10000
    
    data = {
        'days_to_departure': np.random.randint(1, 180, n),
        'stay_duration': np.random.randint(1, 21, n),
        'session_duration_sec': np.random.exponential(400, n),
        'price_dev_avg': np.random.normal(0, 0.15, n),
        'last_10m_price_change': np.random.normal(0, 0.05, n),
        'search_freq_24h': np.random.poisson(4, n)
    }
    
    df = pd.DataFrame(data)
    # Simple logic: Booking is likely if it's last minute, price is low, and session is long
    logit = -4.0 - 0.03*df['days_to_departure'] - 6.0*df['price_dev_avg'] + 0.004*df['session_duration_sec']
    prob = 1 / (1 + np.exp(-logit))
    df['is_booked'] = np.random.binomial(1, prob)

    X = df.drop('is_booked', axis=1)
    y = df['is_booked']

    print("Step 2: Training LightGBM...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = lgb.LGBMClassifier(n_estimators=100, scale_pos_weight=10, verbosity=-1)
    model.fit(X_scaled, y)

    print("Step 3: Saving Artifacts...")
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Success: model.joblib and scaler.joblib created.")

if __name__ == "__main__":
    train_and_save()