import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("dataset\loan_data.csv")
df.info()
# convert to age to int format
df['person_age'] = df['person_age'].astype('int')
df['person_age'].dtypes
cat_cols = [var for var in df.columns if df[var].dtypes == 'object']
num_cols = [var for var in df.columns if df[var].dtypes != 'object']

print(f'Categorical columns: {cat_cols}')
print(f'Numerical columns: {num_cols}')
#visulaization
def plot_categorical_column(dataframe, column):

    plt.figure(figsize=(7, 7))
    ax = sns.countplot(x=dataframe[column])
    total_count = len(dataframe[column])
    threshold = 0.05 * total_count
    category_counts = dataframe[column].value_counts(normalize=True) * 100
    ax.axhline(threshold, color='red', linestyle='--', label=f'0.05% of total count ({threshold:.0f})')
    
    for p in ax.patches:
        height = p.get_height()
        percentage = (height / total_count) * 100
        ax.text(p.get_x() + p.get_width() / 2., height + 0.02 * total_count, f'{percentage:.2f}%', ha="center")
    
    plt.title(f'Label Cardinality for "{column}" Column')
    plt.ylabel('Count')
    plt.xlabel(column)
    plt.tight_layout()
    
    plt.legend()
    plt.show()

for col in cat_cols:
    plot_categorical_column(df, col)
#histogram
df[num_cols].hist(bins=30, figsize=(12,10))
plt.show()
#pie chart for rejected vs approval
label_prop = df['loan_status'].value_counts()

plt.pie(label_prop.values, labels=['Rejected (0)', 'Approved (1)'], autopct='%.2f')
plt.title('Target label proportions')
plt.show()
#feature engineering
threshold = 0.003
correlation_matrix = df.corr()
high_corr_features = correlation_matrix.index[abs(correlation_matrix["loan_status"]) > threshold].tolist()
high_corr_features.remove("loan_status")
print(high_corr_features)
X_selected = df[high_corr_features]
Y = df["loan_status"]
#logistic regression 
X_selected
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
X_train
X_test
Y_train
Y_test
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
#evaluation of logistic regressiom=n
accuracy=accuracy_score(Y_pred, Y_pred)
conf_matrix=confusion_matrix(Y_test, Y_pred)
class_report=classification_report(Y_test, Y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
#SVC
model2 = SVC()
model2.fit(X_train, Y_train)
Y_pred2 = model2.predict(X_test)
print(accuracy_score(Y_test, Y_pred2))
modelc = SVC(C= 10)
modelc.fit(X_train, Y_train)
Y_predc = modelc.predict(X_test)
print(accuracy_score(Y_test, Y_predc))
modelg = SVC(gamma= 0.0122)
modelg.fit(X_train, Y_train)
Y_predg = modelg.predict(X_test)
print(accuracy_score(Y_test, Y_predg))
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["Actual Negative", "Actual Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.show()
#KNN
from sklearn.neighbors import KNeighborsClassifier
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, Y_train)
y_pred_knn = knn.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred_knn)
print(f'Accuracy: {accuracy * 100:.2f}%')
classification_report(Y_test, y_pred_knn)