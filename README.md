# gradient-boosting-from-scratch

End-to-end regression pipeline predicting telecom customer monthly charges using a custom Gradient Boosting implementation from scratch, benchmarked against XGBoost and CatBoost.

---

## 📊 Dataset Overview
This project utilizes the well-known **Telco Customer Churn** dataset (sourced via IBM), adapted here into a **regression task**. 
* **Target Variable:** `MonthlyCharges` (predicting how much a customer pays on a monthly basis).
* **Features:** Includes customer demographics, account tenure, contract types, payment methods, internet services, and total expenses.
* **Preprocessing:** Cleaned missing values and whitespace in structural financial columns (`TotalCharges`), encoded binary features via custom mapping, and handled multi-class categorical features using One-Hot Encoding (`pd.get_dummies`).

---

## 🛠️ Custom Gradient Boosting Implementation (From Scratch)
Implemented entirely in **Python** using first principles without relying on black-box boosting wrappers for the core logic:
* **Base Learners:** Utilizes `DecisionTreeRegressor` (from Scikit-Learn) as weak learners.
* **Iterative Residual Fitting:** Sequentially calculates pseudo-residuals (gradients of the loss function) and trains new trees to correct the errors of previous iterations.
* **Regularization & Control:** Built-in hyperparameters for learning rate (`learning_rate`), number of estimators (`n_estimators`), tree depth, and custom gamma regularization.
* **Validation & Overfitting Check:** Includes custom evaluation functions comparing training vs. testing RMSE dynamics to monitor model stability.

---

## 🚀 Model Benchmarking
To validate the effectiveness of the custom-built architecture, the performance was compared against industry-standard production libraries:
* **XGBoost (`XGBRegressor`)**
* **CatBoost (`CatBoostRegressor`)**

---

## ⚙️ How to Run
1. Clone the repository or open the script in Google Colab.
2. Install the required dependencies:
   ```bash
   pip install xgboost catboost scikit-learn pandas numpy
