# Heart Disease Prediction

A machine learning project that predicts the risk of heart disease from patient clinical features, with an interactive **Streamlit** web app.

## Project Structure

```
Heart-Disease-Prediction-/
├── Dataset/
│   └── heart.csv                 # Patient dataset
├── Notebook/
│   └── HeartDiseasePrediction.ipynb  # EDA, preprocessing & model training
├── models/
│   ├── LogReg_heart.pkl          # Trained Logistic Regression model
│   ├── scaler.pkl                # StandardScaler used for input scaling
│   └── columns.pkl               # Expected feature columns for inference
├── Streamllt_App/
│   └── app.py                    # Streamlit app for real-time prediction
└── requirements.txt
```

## Features

- **Age, Sex, Chest Pain Type (ATA/NAP/TA/ASY)**
- **Resting Blood Pressure, Cholesterol, Fasting Blood Sugar**
- **Resting ECG, Max Heart Rate, Exercise-Induced Angina**
- **Oldpeak (ST Depression), ST Slope**

## Tech Stack

- Python 3.14
- pandas, NumPy, Matplotlib, Seaborn
- scikit-learn (Logistic Regression + StandardScaler)
- Streamlit, Joblib

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Vivekjoshi1973/Heart-Disease-Prediction-.git
   cd Heart-Disease-Prediction-
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**

   ```bash
   cd Streamllt_App
   streamlit run app.py
   ```

   The app opens at `http://localhost:8501`.

## Notebook

Open `Notebook/HeartDiseasePrediction.ipynb` in Jupyter to explore the full pipeline:

1. Data loading and inspection
2. EDA and data cleaning (zero-value imputation for Cholesterol/RestingBP)
3. Encoding and scaling
4. Model training and evaluation

> **Note:** The notebook contains `from google.colab import files` — this only works on Google Colab. When running locally, remove those two lines and load the data with:
>
> ```python
> df = pd.read_csv('../Dataset/heart.csv')
> ```

## Model

Logistic Regression trained on the heart disease dataset, persisted with Joblib along with the fitted `StandardScaler` and the exact column order required at inference time.
