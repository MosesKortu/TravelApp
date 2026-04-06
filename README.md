# 🛫 SkyFlow AI - Predictive Booking Intelligence

**Real-time predictive analytics for travel platforms**  
*Turning window shoppers into confirmed travelers through intelligent intent detection.*

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4A90E2?style=for-the-badge&logo=lightgbm&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stars](https://img.shields.io/github/stars/MosesBargueKortuJr/skyflow-ai.svg)](https://github.com/MosesBargueKortuJr/skyflow-ai/stargazers)

---

## ✨ Project Overview

In the fast-paced world of travel e-commerce, **timing is everything**. SkyFlow AI is a real-time predictive analytics engine that analyzes high-velocity session behavior to identify booking intent and trigger smart, prescriptive interventions.

By combining features like price momentum, session duration, search frequency, and urgency signals, SkyFlow helps travel operators optimize marketing resource allocation and convert highly engaged "window shoppers" into confirmed bookings — for example, by dynamically offering **Price Freezes** at the perfect moment.

**Key Capabilities:**
- Binary classification of booking probability
- Real-time probability scoring for dynamic business logic
- Automated model training & artifact management
- Beautiful, interactive Streamlit dashboard

---

## 📊 Dataset & Features

The model is trained on **10,000 synthetic high-velocity session records** that closely mimic real traveler behavior on travel platforms.

### Features

| Feature              | Code     | Unit      | Range          | Description |
|----------------------|----------|-----------|----------------|-------------|
| Days to Departure    | `D2D`    | Days      | 1 – 180        | Urgency factor — how soon the trip is |
| Stay Duration        | `Stay`   | Days      | 1 – 21         | Intended length of the vacation/trip |
| Session Duration     | `Duration` | Seconds | 10 – 3600     | Total active browsing time in the session |
| Price Deviation      | `p_dev`  | %         | -20.0 – 20.0   | Current price vs 30-day historical average |
| 10m Price Momentum   | `m_change` | %       | -10.0 – 10.0   | Velocity of price movement in last 10 minutes |
| Search Frequency     | `Freq`   | Count     | 1 – 20         | Number of related searches in last 24 hours |

**Target:** `is_booked` (Binary) — 1 = Booking completed, 0 = Cart abandonment or continued browsing.

---

## 🛠️ Modeling Approach

### 1. Feature Engineering & Preprocessing
- Realistic synthetic data generation using **NumPy** (exponential distributions for session time, Poisson for search frequency, etc.)
- Class imbalance handling via `scale_pos_weight=10` in LightGBM
- Feature scaling with **StandardScaler**

### 2. Model Selection
- **Algorithm:** `LGBMClassifier` (LightGBM) — highly optimized gradient boosting
- Automated training pipeline (`engine.py`) that:
  - Detects missing model artifacts on startup
  - Trains the classifier
  - Serializes artifacts with **joblib** for instant dashboard use

### 3. Business Logic Integration
The model outputs a **booking probability score** that powers intelligent actions:

- **< 40% probability + high engagement** → Trigger "Price Protection" or recovery offers
- **> 75% probability** → Standard checkout flow (protect margins, avoid unnecessary discounts)

---

## 🏆 Results

LightGBM excels at capturing non-linear relationships in high-velocity tabular data, especially interactions between **urgency (D2D)** and **price friction (momentum & deviation)**.

> *Note: This iteration uses synthetic data. Metrics will scale and improve further with real clickstream data.*

---

## 🚀 Next Steps & Future Enhancements

- Real-time ingestion from live clickstream databases (Snowflake, BigQuery, etc.)
- Advanced hyperparameter tuning with **Optuna**
- Sequential modeling using **LSTM** or **Transformer** architectures on full click-path data
- A/B testing framework for intervention strategies
- Production deployment (Docker + FastAPI backend)

---

## 💻 Quick Start — Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/MosesBargueKortuJr/skyflow-ai.git
cd skyflow-ai
```
### 2. Install dependencies
``` bash
pip install -r requirements.txt
```
### 3. Launch the Real-Time Dashboard
``` bash
streamlit run app.py
```
### 📁 Project Structure
``` bash 
skyflow-ai/
├── app.py                    # 🎨 Main Streamlit dashboard (real-time predictions)
├── engine.py                 # ⚙️ Automated model training & artifact management
├── data_generator.py         # 🔬 Synthetic dataset generator (10k sessions)
├── models/                   # 💾 Saved LightGBM model artifacts (.joblib)
│   ├── lightgbm_model.joblib
│   └── scaler.joblib
├── requirements.txt          # 📦 All Python dependencies
├── README.md
└── ...
```
### Contributing
Contributions, bug reports, and feature requests are highly welcome!

Open a new Issue
Submit a Pull Request
