# Loan Prediction Model

This project builds a machine learning model to predict loan approval status (approved or rejected) based on various features of individuals. The goal is to use multiple machine learning algorithms and evaluate their performance in terms of accuracy and other metrics.

## Project Overview

This project uses various classification algorithms such as Logistic Regression, Support Vector Classifier (SVC), K-Nearest Neighbors (KNN), and Random Forest to predict whether a loan is approved based on multiple features such as person age, loan amount, credit history, and others. The model performance is evaluated using metrics like accuracy, confusion matrix, and classification report.

## Datasets

- **Loan Data:** This dataset contains various features related to the applicant's personal and loan details such as `person_age`, `loan_amount`, `credit_history`, and the target label `loan_status` (approved or rejected).
The dataset file is stored in `dataset/loan_data.csv`.
##Visualization
We visualize the dataset using various types of plots:

Categorical Data Visualization: Count plots are used for categorical columns.
![Figure_1](https://github.com/user-attachments/assets/6ef77a0f-021d-4a66-b496-10e7dc3eea0c)
![Figure_2](https://github.com/user-attachments/assets/75fb2201-b624-4fbd-a1a3-a69cad35b74f)
![Figure_3](https://github.com/user-attachments/assets/b5820b5c-292c-43be-b380-59a03407ac54)
![Figure_4](https://github.com/user-attachments/assets/e4f0fd36-45e3-4e38-805b-e49cb7b91faf)


Numerical Data Visualization: Histograms are plotted for numerical columns:
![Figure_5](https://github.com/user-attachments/assets/31870c79-74e8-4959-acf7-10186570990b)


Target Variable Proportions: A pie chart is used to visualize the proportion of loan approvals vs. rejections.
![Figure_6](https://github.com/user-attachments/assets/55ae3028-df86-46c9-ac2c-7047aca1769a)

## Libraries Used

This project uses the following libraries:

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib`, `seaborn`, `plotly` - Data visualization
- `scikit-learn` - Machine learning models and evaluation metrics
- `xgboost` - Extreme Gradient Boosting (for additional models, if needed)



```bash
