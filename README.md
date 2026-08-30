# Employee Salary Prediction

## Project Overview

This project predicts an employee's salary using Machine Learning.

The model uses information such as:

- Age
- Gender
- Department
- Job Title
- Experience
- Education Level
- Location

I also created a Streamlit web application where users can enter employee details and get a predicted salary.


## Problem Statement

The goal of this project is to predict an employee's expected salary based on their personal and professional information.

This is a **Regression Machine Learning problem** because the output is a continuous numerical value (salary).


## Dataset

The dataset contains employee information and their salaries.

### Features

 | Feature | Description |
|---|---|
| Age | Employee's age |
| Gender | Employee's gender |
| Department | Employee's department |
| Job_Title | Employee's job title |
| Experience_Years | Years of experience |
| Education_Level | Employee's education level |
| Location | Employee's location |
| Salary | Employee's salary |

`Salary` is the **target variable**.
> Note: Employee ID and Name are identifiers and are not used as prediction features in the Streamlit application.


## Data Analysis

During the project, I performed:

- Data cleaning
- Missing value checking
- Duplicate checking
- Outlier analysis
- Descriptive statistics
- Categorical data analysis
- Salary analysis
- Correlation analysis

I also analyzed how factors such as education, job title, and experience are related to salary.



## Machine Learning

I used **Linear Regression** to predict employee salary.

The categorical features were converted into numerical values using **One-Hot Encoding**.

I used a Machine Learning **Pipeline** so that preprocessing and prediction happen together.



## Model Evaluation

The model was evaluated using:

- R² Score
- MAE
- MSE
- RMSE
- Train/Test comparison
- Cross-validation
- Residual analysis
- Leakage checks

### Model Results

- **R² Score:** 0.9918
- **MAE:** approximately ₹3,337.94
- **RMSE:** approximately ₹4,168.34
- **MSE:** approximately 17,375,021.62
- **Mean 5-Fold CV R²:** approximately 0.9913
- **CV Standard Deviation:** approximately 0.0002


### Model Validation & Trust

A high R² score alone does not mean that a model is perfect.

To check whether the high R² result was reliable, I also checked:

- Training vs testing performance
- 5-Fold Cross-validation
- MAE and RMSE
- Residual analysis
- Data leakage

The training and testing R² scores were very close, and the cross-validation scores were also consistent across the five folds.


## Streamlit Application

I created an interactive Streamlit dashboard.

The application contains three main sections:

### Dashboard

Shows:

- Total employees
- Average salary
- Minimum salary
- Maximum salary
- Salary by education
- Salary by job title
- Salary distribution
- Salary vs experience
- Salary by department

### Model Validation

Shows the model evaluation results and explains why the model's performance should be checked using multiple validation methods.

### Salary Prediction

Users can enter employee information and get a predicted salary.


## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Plotly
- Streamlit
- Jupyter Notebook



## How to Run

### 1. Install the required libraries

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.


## Project Structure

```text
Emp-Salary-Prediction/
│
├── app.py
├── Employee_Salary_Prediction_Analysis.ipynb
├── Employers_data.csv
├── employee_salary_model.pkl
├── requirements.txt
└── README.md
```
