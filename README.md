# Employee Turnover Prediction

## Problem Statement

TalentCore Pvt. Ltd., a multinational company, has been experiencing a rising number of employee resignations, leading to increased recruitment costs, project delays, and loss of skilled talent. The HR department wants an intelligent ML system that can predict whether an employee is likely to leave the company, based on job satisfaction, salary, age, work-life balance, training hours, bonuses, and other work-related factors.

## Objective

As the AI/ML Engineer on this project, the goals are to:

1. Build a baseline **Logistic Regression** model.
2. Improve it using **Regularization (L1 and L2)**.
3. Compare the performance of all models and recommend the best one.

## Dataset Description

The employee dataset contains **900 rows** and **15 columns**, representing various employee metrics reflecting realistic corporate scenarios.

| Feature | Description |
|---|---|
| Job_Satisfaction | Level of satisfaction with the job |
| Performance_Rating | Employee performance score |
| Years_At_Company | Number of years worked in the company |
| Work_Life_Balance | Balance between work and personal life |
| Distance_From_Home | Distance of home from workplace |
| Monthly_Income | Monthly salary |
| Education_Level | Education qualification level |
| Age | Age of employee |
| Num_Companies_Worked | Number of companies worked previously |
| Employee_Role | Encoded job role |
| Annual_Bonus | Bonus received annually |
| Training_Hours | Training hours attended |
| Department | Encoded department |
| Annual_Bonus_Squared | Engineered feature (bonus²) |
| Annual_Bonus_Training_Hours_Interaction | Interaction feature between annual bonus and training hours |
| **Employee_Turnover (Target)** | 1 = Employee Left, 0 = Stayed |

## Approach

- **Baseline model:** Logistic Regression (no regularization)
- **L1 Regularization:** Logistic Regression with Lasso penalty (`penalty='l1'`, `solver='liblinear'`) — performs feature selection by shrinking less important coefficients to zero
- **L2 Regularization:** Logistic Regression with Ridge penalty (`penalty='l2'`) — shrinks coefficients to reduce overfitting without eliminating features
- **Evaluation metrics:** Accuracy, Precision, Recall, F1-score (via `classification_report`)
- **Train/test split:** 80/20

## Repository Structure

```
employee-turnover-prediction/
├── README.md
├── .gitignore
├── employee_turnover_project.ipynb   # main notebook: EDA, modeling, evaluation
├── employee_turnover.csv                          # dataset
```

## Requirements

```
pandas
scikit-learn
numpy
matplotlib
seaborn
jupyter
```

## Usage

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place `employee_turnover.csv` in the proper place (or update the path in the notebook)
4. Run the notebook: `jupyter notebook employee_turnover_project.ipynb`
