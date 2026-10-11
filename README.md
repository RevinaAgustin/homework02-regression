# Machine Learning Zoomcamp 2026 — Homework 2: Linear Regression

This repository contains the solution and implementation for **Homework 2 (Regression)** from the [Machine Learning Zoomcamp 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp) by DataTalksClub.

The primary objective is to implement Linear Regression from scratch (without scikit-learn) using NumPy and Pandas to predict vehicle fuel efficiency (`fuel_efficiency_mpg`).

---

## 📊 Dataset Overview

* **Dataset:** 2026 Car Fuel Efficiency Dataset
* **Source:** [DataTalksClub 2026 Cohort Repository](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv)
* **Target Variable:** `fuel_efficiency_mpg`
* **Features Selected:**
  * `engine_displacement`
  * `horsepower`
  * `vehicle_weight`
  * `model_year`

Unlike car prices, the target variable `fuel_efficiency_mpg` exhibits an approximately symmetric (normal) distribution without an extreme long tail, so no logarithmic transformation (`log1p`) is required.

---

## 🛠️ Methodology & Implementation Details

All models are constructed from first principles using NumPy:

1. **Validation Framework:**
   * Split ratio: 60% Train, 20% Validation, 20% Test.
   * Reproducible shuffling via `np.random.seed`.

2. **Linear Regression via Normal Equation:**
   $$\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

3. **Regularized Linear Regression (Ridge / L2):**
   $$\mathbf{w} = (\mathbf{X}^T \mathbf{X} + r\mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$
   where $r$ controls regularization strength to avoid multicollinearity.

4. **Evaluation Metric:**
   * **Root Mean Squared Error (RMSE):**
     $$\text{RMSE} = \sqrt{\frac{1}{m} \sum_{i=1}^m (y_i - \hat{y}_i)^2}$$

---

## 📝 Homework Questions & Results Summary

| Question | Topic | Findings / Metric | Final Answer |
| :--- | :--- | :--- | :--- |
| **Q1** | Missing Value Identification | `horsepower` has 877 missing entries (`NaN`). | **`horsepower`** |
| **Q2** | Median Calculation | Median value for `horsepower`. | **`254`** |
| **Q3** | Missing Value Imputation | Imputing with 0 (RMSE: `2.205`) vs Mean (RMSE: `2.202`). | **`With mean`** |
| **Q4** | Regularization Tuning ($r$) | Tested $r \in [0, 0.01, 0.1, 1, 5, 10, 100]$. Smallest RMSE (`2.2053`) achieved at $r = 0$. | **`0`** |
| **Q5** | Model Stability Evaluation | Standard deviation of RMSE across 10 random seeds (`0–9`) is `0.029`. | **`0.029`** |
| **Q6** | Final Test Set Evaluation | Combined Train + Val (Seed 9, $r = 0.001$). Test RMSE is `2.236`. | **`2.236`** |

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RevinaAgustin/homework02-regression.git
   cd homework02-regression
   ```

2. **Activate your environment & install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt # or: pip install pandas numpy jupyter
   ```

3. **Launch the Notebook:**
   ```bash
   jupyter notebook main.ipynb
   # or open main.ipynb in VS Code
   ```

---

## 📦 Tech Stack

* **Language:** Python 3.13
* **Libraries:** NumPy, Pandas, Matplotlib / Seaborn
* **Environment:** Jupyter Notebook / Visual Studio Code
