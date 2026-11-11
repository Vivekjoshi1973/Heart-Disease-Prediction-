# Heart Disease Prediction

A machine learning project that predicts the risk of heart disease from patient clinical features, with an interactive **Streamlit** web app.

## 🚀 Live Demo

**https://heart-disease-prediction-vj.streamlit.app/**

## Project Structure

```
Heart-Disease-Prediction-/
├── Dataset/
│   └── heart.csv                 # Patient dataset (918 rows, 12 columns)
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

## Dataset

| Property | Value |
| --- | --- |
| Source file | `Dataset/heart.csv` |
| Rows / Columns | 918 × 12 |
| Duplicates / Missing | 0 / 0 |
| Target | `HeartDisease` (1 = disease, 0 = no disease) |
| Class balance | 508 positive (55.3%) : 410 negative (44.7%) |

**Features (11 inputs):**

- **Age, Sex, Chest Pain Type (ATA/NAP/TA/ASY)**
- **Resting Blood Pressure, Cholesterol, Fasting Blood Sugar**
- **Resting ECG, Max Heart Rate, Exercise-Induced Angina**
- **Oldpeak (ST Depression), ST Slope**

## Pipeline

1. **Data loading and inspection** — shape, info, describe, duplicates, null check.
2. **EDA & cleaning** — histograms, count plots, box/violin plots, correlation heatmap.
   - Zero values in `Cholesterol` and `RestingBP` (physiologically impossible) are replaced with the mean of the non-zero values.
3. **Encoding** — `pd.get_dummies(drop_first=True)` converts categorical features into one-hot columns, then cast to `int`.
4. **Scaling** — `StandardScaler` on `Age, RestingBP, Cholesterol, MaxHR, Oldpeak`.
5. **Split** — 80/20 `train_test_split(random_state=42)` → 734 train / 184 test rows.
6. **Model training & evaluation** — five classifiers compared on Accuracy and F1.
7. **Export** — best model (`LogReg_heart.pkl`), `scaler.pkl`, `columns.pkl` saved with Joblib.

## Model Comparison

Evaluation on the 184-sample held-out test set (20%, `random_state=42`):

| Rank | Model | Accuracy | F1 Score |
| --- | --- | --- | --- |
| 🥇 1 | **Logistic Regression** | **0.8696 (86.96%)** | **0.8846** |
| 🥈 2 | KNN | 0.8641 (86.41%) | 0.8815 |
| 🥉 3 | SVM (RBF) | 0.8478 (84.78%) | 0.8667 |
| 4 | Naive Bayes | 0.8478 (84.78%) | 0.8614 |
| 5 | Decision Tree | 0.7663 (76.63%) | 0.7902 |

> **Highest accuracy: Logistic Regression (86.96%).**

## Classification Reports — Every Model

### 1. Logistic Regression — Accuracy 86.96%, F1 0.8846 ✅ (chosen)

```
              precision    recall  f1-score   support

           0       0.82      0.88      0.85        77
           1       0.91      0.86      0.88       107

    accuracy                           0.87       184
   macro avg       0.87      0.87      0.87       184
weighted avg       0.87      0.87      0.87       184
```

### 2. K-Nearest Neighbors — Accuracy 86.41%, F1 0.8815

```
              precision    recall  f1-score   support

           0       0.82      0.86      0.84        77
           1       0.89      0.87      0.88       107

    accuracy                           0.86       184
   macro avg       0.86      0.86      0.86       184
weighted avg       0.87      0.86      0.86       184
```

### 3. Support Vector Machine — Accuracy 84.78%, F1 0.8667

```
              precision    recall  f1-score   support

           0       0.80      0.84      0.82        77
           1       0.88      0.85      0.87       107

    accuracy                           0.85       184
   macro avg       0.84      0.85      0.84       184
weighted avg       0.85      0.85      0.85       184
```

### 4. Naive Bayes — Accuracy 84.78%, F1 0.8614

```
              precision    recall  f1-score   support

           0       0.78      0.90      0.83        77
           1       0.92      0.81      0.86       107

    accuracy                           0.85       184
   macro avg       0.85      0.85      0.85       184
weighted avg       0.86      0.85      0.85       184
```

### 5. Decision Tree — Accuracy 76.63%, F1 0.7902

```
              precision    recall  f1-score   support

           0       0.71      0.84      0.77        77
           1       0.87      0.76      0.81       107

    accuracy                           0.79       184
   macro avg       0.80      0.80      0.79       184
weighted avg       0.81      0.79      0.79       184
```

> The Decision Tree is unseeded (`DecisionTreeClassifier()` with no `random_state`), so its exact numbers fluctuate between runs (roughly 76–80% accuracy). Its report above is from a reproduction run; the comparison table uses the value recorded in the notebook.

## Why We Chose Logistic Regression

1. **Highest accuracy** — 86.96% on the held-out test set, ahead of KNN (86.41%), SVM and Naive Bayes (84.78%), and Decision Tree (76.63%).
2. **Highest F1 score (0.8846)** — the best balance of precision and recall, which matters because both false negatives (missed disease) and false positives (unnecessary alarm) are costly in screening.
3. **Best precision on the positive class (0.91)** — when the model flags a patient as high-risk, it is correct ~91% of the time, the strongest of all five models.
4. **Most balanced per-class performance** — macro averages are 0.87/0.87/0.87 for both classes, so it does not trade away one class for the other (Naive Bayes, for example, over-predicts the negative class: recall 0.90 vs 0.81).
5. **Stable and interpretable** — linear coefficients can be read as per-feature risk contributions, unlike KNN/SVM black boxes, and unlike the Decision Tree it does not overfit a 918-row dataset.
6. **Cheap at inference** — a single dot product plus the scaler, ideal for the Streamlit app's real-time predictions.

Logistic Regression is therefore the deployed model: `LogReg_heart.pkl` + `scaler.pkl` + `columns.pkl`.

## Tech Stack

- Python 3.14
- pandas, NumPy, Matplotlib, Seaborn
- scikit-learn (Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM + StandardScaler)
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
4. Model training and evaluation (5 classifiers, accuracy + F1)

> **Note:** The notebook contains `from google.colab import files` — this only works on Google Colab. When running locally, remove those two lines and load the data with:
>
> ```python
> df = pd.read_csv('../Dataset/heart.csv')
> ```

## Model

Logistic Regression trained on the heart disease dataset, persisted with Joblib along with the fitted `StandardScaler` and the exact column order required at inference time. The Streamlit app loads all three artifacts, one-hot encodes the raw user input, aligns it to `columns.pkl`, scales it, and prints a High/Low risk verdict.
