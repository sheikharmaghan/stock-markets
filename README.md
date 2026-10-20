# 📈 Pakistan Stock Market (PSX) Closing Price Prediction

Predicts the **closing price** of Pakistan Stock Exchange (PSX) stocks using a **Decision Tree Regressor**. A **Streamlit** web app lets you upload a CSV and get predictions.

**Live app:** _paste your Streamlit link here after deployment_

---

## 📊 Model Results (20% test data, 168,066 rows)

| Metric | Value |
|--------|-------|
| MAE | 2.82 |
| RMSE | 88.64 |
| R² Score | 0.990 (99.0%) |

Full numbers are in `psx_model_evaluation.csv`.

---

## 🧠 How It Works

- **Dataset:** Pakistan Stock Market Data 2017–2025 (Kaggle, `fayaznoor10`).
- **Input features:** `SYMBOL`, `LDCP`, `OPEN`, `HIGH`, `LOW`, `VOLUME`, `DATE`
- **Target:** `CLOSE` (closing price)
- **Preprocessing:** `DATE` is split into year, month and day. Leakage columns such as `CHANGE` are removed. Numeric columns use median imputation. `SYMBOL` is one-hot encoded.
- **Model:** `DecisionTreeRegressor(max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42)` inside a scikit-learn `Pipeline`.
- **Split:** 80% training, 20% testing.

---

## 📂 Repository Files

```
├── app.py                        # Streamlit app
├── requirements.txt              # Dependencies (exact name required)
├── psx_decision_tree_model.pkl   # Trained model
├── psx_model_evaluation.csv      # Test metrics
├── psx_test_predictions.csv      # Actual vs predicted on test data
├── __notebook_source__.ipynb     # Kaggle notebook (EDA + training)
└── README.md
```

---

## 📦 requirements.txt

The file must be named exactly `requirements.txt` and contain:

```
streamlit
joblib
scikit-learn==1.6.1
pandas
numpy
```

> ⚠️ The model was trained with **scikit-learn 1.6.1**. A different version gives errors like `_RemainderColsList` not found, so do not remove this pin.

---

## 🚀 Deploy on Streamlit Community Cloud

1. Make sure these 3 files are in the **root** of the GitHub repo: `app.py`, `psx_decision_tree_model.pkl`, `requirements.txt`.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** and choose **Deploy a public app from GitHub**.
4. Fill in:
   - **Repository:** `your-username/stockmarkets`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Open **Advanced settings** and set **Python version** to `3.11`.
6. Click **Deploy** and wait a few minutes while the packages install.
7. Copy the app link and paste it at the top of this README.

**If the app shows an error:** open **Manage app** (bottom right) and read the logs. After fixing a file on GitHub, use **Manage app → Reboot app**.

---

## 💻 Run Locally

```bash
git clone https://github.com/your-username/stockmarkets.git
cd stockmarkets
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧪 Using the App

1. Upload a CSV with the columns `DATE, SYMBOL, LDCP, OPEN, HIGH, LOW, VOLUME` (a `CLOSE` column is optional).
2. The predicted closing prices appear in a table.
3. If `CLOSE` is present, the app also shows MAE, RMSE, R² and an Actual vs Predicted chart.
4. Download the results as a CSV.

---

## 🛠️ Tech Stack

Python · Pandas · NumPy · Scikit-learn · Joblib · Matplotlib · Streamlit

---

## ⚠️ Disclaimer

For educational purposes only. Predictions are not financial advice.
