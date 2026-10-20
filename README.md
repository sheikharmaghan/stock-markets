# stock-markets  
# 📈 Pakistan Stock Market (PSX) Closing Price Prediction

A machine learning project that predicts the **closing price** of Pakistan Stock Exchange (PSX) stocks using a **Decision Tree Regressor**, with a simple **Streamlit** web app for making predictions from a CSV file.

**Live app:** _add your Streamlit app link here_

---

## 📌 Project Overview

- **Goal:** Predict the closing price of PSX stocks from historical market data.
- **Dataset:** Pakistan Stock Market Data 2017–2025 (`compiled_psx_historical_2017_2025.csv`) from Kaggle, by `fayaznoor10`.
- **Model:** Scikit-learn `Pipeline` (preprocessing + Decision Tree Regressor).
- **Training platform:** Kaggle Notebook.
- **App:** Streamlit (deployed on Streamlit Community Cloud).

---

## 🧠 How the Model Works

**1. Data cleaning**
- Column names are stripped of extra spaces.
- Infinite values are converted to missing values.
- Rows with a missing closing price are removed, and duplicate rows are dropped.

**2. Feature preparation**
- Date/time columns are split into `year`, `month` and `day` features.
- Columns that directly reveal the target (for example `change`, `daily_return`) are removed to avoid **target leakage**.
- Numeric-looking text columns are converted to numbers.

**3. Preprocessing (inside the Pipeline)**
- Numeric columns: median imputation.
- Categorical columns: most-frequent imputation and One-Hot Encoding.

**4. Model**

```python
DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
```

**5. Train/test split:** 80% training, 20% testing (`random_state=42`).

**6. Evaluation metrics:** MAE, MSE, RMSE and R² score. Results are saved in `psx_model_evaluation.csv`.

---

## 📂 Repository Structure

```
├── app.py                        # Streamlit web app
├── requirements.txt              # Python dependencies
├── psx_decision_tree_model.pkl   # Trained model (scikit-learn Pipeline)
├── psx_model_evaluation.csv      # MAE, MSE, RMSE, R² on the test set
├── psx_test_predictions.csv      # Actual vs predicted values on the test set
├── __notebook_source__.ipynb     # Kaggle notebook (EDA, training, evaluation)
└── README.md
```

---

## 🚀 Run the App

**Online:** open the Streamlit link above, upload a CSV, and view the predictions.

**Locally:**

```bash
git clone https://github.com/<your-username>/stockmarkets.git
cd stockmarkets
pip install -r requirements.txt
streamlit run app.py
```

### Using the app

1. Upload a CSV file that has the **same columns** as the training dataset.
2. The app applies the same cleaning steps used in training.
3. The predicted closing prices are shown in a table.
4. If your CSV also contains the real closing price, the app shows MAE, RMSE, R² and an Actual vs Predicted chart.
5. Download the predictions as a CSV file.

---

## 📦 Requirements

```
streamlit
joblib==1.6.0
scikit-learn==1.6.1
pandas==2.3.3
numpy==2.1.3
```

> ⚠️ **Important:** The model was trained with **scikit-learn 1.6.1**. Loading the `.pkl` file with a different scikit-learn version can cause errors such as `_RemainderColsList` not found. Keep the version pinned.

---

## 🛠️ Tech Stack

Python · Pandas · NumPy · Scikit-learn · Joblib · Matplotlib · Streamlit

---

## 👤 Author

**Muhammad Naqi**
BS Robotics, Section 2A
Course: Programming Fundamentals

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. Predictions are not financial advice and should not be used for real trading decisions.
