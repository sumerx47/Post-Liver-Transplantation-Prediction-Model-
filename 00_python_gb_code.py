import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
import joblib
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import sklearn.metrics as skmet
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


# Load the dataset
data = pd.read_csv(r'C:\Users\sumair\OneDrive\Desktop\post liver transplantation prediction\dataset\LiverT_dataset.csv')


# Define a dictionary mapping old column names to new column names
column_mapping = {
    'D Age': 'donor_age',
    'D Gender': 'donor_gender',
    'D BMI': 'donor_bmi',
    'D Cause of Death': 'donor_cause_of_death',
    'D Diabetes': 'donor_diabetes',
    'D Hypertension': 'donor_hypertension',
    'D Alcohol Abuse': 'donor_alcohol_abuse',
    'D Smoking': 'donor_smoking',
    'D Lympochyte': 'donor_lymphocyte',
    'D Hepatitis B': 'donor_hepatitis_b',
    'D Hepatitis C': 'donor_hepatitis_c',
    'R Etiology': 'recipient_etiology',
    'R MELD Score': 'recipient_meld_score',
    'R Age': 'recipient_age',
    'R Gender': 'recipient_gender',
    'R BMI': 'recipient_bmi',
    'R Diabetes': 'recipient_diabetes',
    'R Hypertension': 'recipient_hypertension',
    'R Alcohol Abuse': 'recipient_alcohol_abuse',
    'R Smoking': 'recipient_smoking',
    'R Lympochyte': 'recipient_lymphocyte',
    'R Hepatitis B': 'recipient_hepatitis_b',
    'R Hepatitis C': 'recipient_hepatitis_c',
    'R Albumin level': 'recipient_albumin_level',
    'R Alcoholic cirrhosis': 'recipient_alcoholic_cirrhosis',
    'R Primary biliary cirrhosis': 'recipient_primary_biliary_cirrhosis',
    'R Na': 'recipient_na',
    'R Mg': 'recipient_mg',
    'R WBC': 'recipient_wbc',
    'R Platelets': 'recipient_platelets',
    'R Cold Ischemia Time': 'recipient_cold_ischemia_time',
    'R Warm Ischemia Time': 'recipient_warm_ischemia_time',
    'R Blood Transfusion': 'recipient_blood_transfusion',
    'R Immunosuppressant Medication': 'recipient_immunosuppressant_medication',
    'R Rejection Episodes': 'recipient_rejection_episodes',
    'Complications':'complications'
}

# Rename the columns using the column_mapping dictionary
data = data.rename(columns=column_mapping)

###########################################################
'''
#Auto EDA library - Sweetviz
import sweetviz as sv

report = sv.analyze(data)
report.show_html()

## Get the current working directory
import os

cwd = os.getcwd()
print("Current Working Directory:", cwd)

'''
############################################################

# Represents the information related to the dataset
data.info()

# Represents the total number of value each of the column has.
data.count()

# Represents the number of elements present in each row and total number of columns
data.shape

# Represents the each column name
data.columns

# Represents the first 5 rows with all columns of the data frame
data.head(5)

# Represents the last 5 rows with all columns of the data frame
data.tail(5)

# Dropping Time column
df = data.drop(['Column1'], axis=1)


## Exploratory Data Analysis (EDA)
# First Business Moment
# Mean
df.mean()

# Median
df.median()

# Mode
df.mode()

# Second Business Moment
# Standard Deviation
df.std()

# Variance
df.var()

# Third Business Moment
# Skewness
df.skew()

# Fourth Business Moment
# Kurtosis
df.kurt()

df.dtypes

# Univariate Plot (Histogram) for each column
for column in df.columns:
    plt.figure()
    df[column].hist()
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column}')

# Density Plot for each column
numeric_columns = ['donor_age', 'donor_bmi', 'recipient_meld_score', 'recipient_age', 'recipient_bmi',
                   'recipient_albumin_level', 'recipient_na', 'recipient_mg', 'recipient_wbc', 'recipient_platelets',
                   'recipient_cold_ischemia_time', 'recipient_warm_ischemia_time', 'recipient_rejection_episodes']

for column in numeric_columns:
    plt.figure()
    sns.kdeplot(df[column], shade=True)
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.title(f'Density Plot of {column}')
plt.show()


# Boxplot for numeric columns
numeric_columns = ['donor_age', 'donor_bmi', 'recipient_meld_score', 'recipient_age', 'recipient_bmi',
                   'recipient_albumin_level', 'recipient_na', 'recipient_mg', 'recipient_wbc', 'recipient_platelets','recipient_alcoholic_cirrhosis', 'recipient_primary_biliary_cirrhosis',
                   'recipient_cold_ischemia_time', 'recipient_warm_ischemia_time', 'recipient_rejection_episodes']

for column in numeric_columns:
    plt.figure()
    sns.boxplot(data=df[column])
    plt.xlabel(column)
    plt.title(f'Boxplot of {column}')

plt.show()

# Boxplot for categorical columns
categorical_columns = ['donor_gender', 'donor_cause_of_death', 'donor_diabetes', 'donor_hypertension',
                       'donor_alcohol_abuse', 'donor_smoking', 'donor_lymphocyte', 'donor_hepatitis_b',
                       'donor_hepatitis_c', 'recipient_etiology', 'recipient_gender', 'recipient_diabetes',
                       'recipient_hypertension', 'recipient_alcohol_abuse', 'recipient_smoking',
                       'recipient_lymphocyte', 'recipient_hepatitis_b', 'recipient_hepatitis_c',
                       
                       'recipient_blood_transfusion', 'recipient_immunosuppressant_medication']

for column in categorical_columns:
    plt.figure()
    sns.boxplot(data=df, x=column, y='recipient_age', hue=column)
    plt.ylabel('recipient_age')
    plt.title(f'Boxplot of recipient_age by {column}')

plt.show()


# Count Plot for each column
for column in df.columns:
    plt.figure()
    sns.countplot(data[column])
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(f'Count Plot of {column}')


# Scatter Plot for numeric columns
numeric_columns = ['donor_age', 'donor_bmi', 'recipient_meld_score', 'recipient_age', 'recipient_bmi',
                   'recipient_albumin_level', 'recipient_na', 'recipient_mg', 'recipient_wbc', 'recipient_platelets',
                   'recipient_cold_ischemia_time', 'recipient_warm_ischemia_time', 'recipient_rejection_episodes']
for column in numeric_columns:
    plt.figure()
    plt.scatter(data[column], data['complications'])
    plt.xlabel(column)
    plt.ylabel('Complications')
    plt.title(f'Scatter Plot of {column} vs Complications')

    
# Bar Plot for categorical columns
categorical_columns = ['donor_gender', 'donor_cause_of_death', 'donor_diabetes', 'donor_hypertension',
                       'donor_alcohol_abuse', 'donor_smoking', 'donor_lymphocyte', 'donor_hepatitis_b',
                       'donor_hepatitis_c', 'recipient_etiology', 'recipient_gender', 'recipient_diabetes',
                       'recipient_hypertension', 'recipient_alcohol_abuse', 'recipient_smoking',
                       'recipient_lymphocyte', 'recipient_hepatitis_b', 'recipient_hepatitis_c',
                       'recipient_alcoholic_cirrhosis', 'recipient_primary_biliary_cirrhosis',
                       'recipient_blood_transfusion', 'recipient_immunosuppressant_medication']
for column in categorical_columns:
    plt.figure()
    sns.countplot(data[column], hue=data['complications'])
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(f'Count Plot of {column} with Complications')

# Heatmap of correlation matrix
correlation_matrix = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

# Pairwise scatter plot of numeric columns
sns.pairplot(data[numeric_columns], diag_kind='kde')
plt.show()



# Univariate Plot
# Pie chart to check the ratio of data 
data["Complications"].value_counts().plot.pie(autopct = '%.1f')

#### Bi-variate plot
# Creating heatmap
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

###  Multi variate Plot
# Pair plot between the variables  
sns.pairplot(df)

# Correlation matrix 
df.corr()


# Check for missing/Null values
df.isnull().sum()

# Duplicates
df.duplicated().sum()
df1 = df.drop_duplicates()
df1.duplicated().sum()


# Handle missing values
imputer = SimpleImputer(strategy='median')
numerical_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(exclude=np.number).columns

df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

    
#Handling outliers
from feature_engine.outliers import Winsorizer
# Apply winsorization to 'recipient_na' column
winsorizer = Winsorizer(capping_method='iqr', tail='both', fold=0.05, variables=['recipient_na'])
df = winsorizer.fit_transform(df)
# Apply winsorization to 'recipient_mg' column
winsorizer = Winsorizer(capping_method='iqr', tail='both', fold=0.05, variables=['recipient_mg'])
df = winsorizer.fit_transform(df)
    
    
X_sampled = df.drop('complications', axis=1)
Y_sampled = df['complications']

# Perform label encoding for categorical columns
label_encoder = LabelEncoder()
for column in X_sampled.select_dtypes(include='object').columns:
    X_sampled[column] = label_encoder.fit_transform(X_sampled[column])
Y_sampled = label_encoder.fit_transform(Y_sampled)


from imblearn.over_sampling import SMOTE
# Instantiate the SMOTE algorithm
smote = SMOTE()

# Apply SMOTE to the dataset
X, Y = smote.fit_resample(X_sampled, Y_sampled)

# Check the class distribution before SMOTE
print("Class distribution before SMOTE:")
print(pd.Series((Y_sampled.squeeze())).value_counts())

# Check the class distribution after SMOTE
print("Class distribution after SMOTE:")
print(pd.Series(Y.squeeze()).value_counts())

X.shape

Y.shape

# Split the data into train, validation, and test sets
X_trainval, X_test, y_trainval, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)

# Define a dictionary to store the model accuracies
model_accuracies = {}

# Train and evaluate Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
dt_pred = dt_classifier.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
model_accuracies['Decision Tree'] = dt_accuracy

# Model evaluation
confusion_mat = confusion_matrix(y_test, dt_pred)
classification_rep = classification_report(y_test, dt_pred)
print("Decision Tree")
print("Confusion Matrix:")
print(confusion_mat)
print("Classification Report:")
print(classification_rep)
print("Accuracy:", dt_accuracy)
print()

# Train and evaluate Logistic Regression
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)
lr_pred = lr_classifier.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
model_accuracies['Logistic Regression'] = lr_accuracy

# Model evaluation
confusion_mat = confusion_matrix(y_test, lr_pred)
classification_rep = classification_report(y_test, lr_pred)
print("Logistic Regression")
print("Confusion Matrix:")
print(confusion_mat)
print("Classification Report:")
print(classification_rep)
print("Accuracy:", lr_accuracy)
print()

# Train and evaluate Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
rf_pred = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
model_accuracies['Random Forest'] = rf_accuracy

# Model evaluation
confusion_mat = confusion_matrix(y_test, rf_pred)
classification_rep = classification_report(y_test, rf_pred)
print("Random Forest")
print("Confusion Matrix:")
print(confusion_mat)
print("Classification Report:")
print(classification_rep)
print("Accuracy:", rf_accuracy)
print()

# Train and evaluate Support Vector Classifier
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
svm_pred = svm_classifier.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
model_accuracies['Support Vector Machine'] = svm_accuracy

# Model evaluation
confusion_mat = confusion_matrix(y_test, svm_pred)
classification_rep = classification_report(y_test, svm_pred)
print("Support Vector Machine")
print("Confusion Matrix:")
print(confusion_mat)
print("Classification Report:")
print(classification_rep)
print("Accuracy:", svm_accuracy)
print()

from sklearn.ensemble import GradientBoostingClassifier
# Train and evaluate Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(X_train, y_train)
gb_pred = gb_classifier.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
model_accuracies['Gradient Boosting'] = gb_accuracy

# Model evaluation
confusion_mat = confusion_matrix(y_test, gb_pred)
classification_rep = classification_report(y_test, gb_pred)
print("Gradient Boosting")
print("Confusion Matrix:")
print(confusion_mat)
print("Classification Report:")
print(classification_rep)
print("Accuracy:", gb_accuracy)
print()


# Print the accuracies of the trained models
for model, accuracy in model_accuracies.items():
    print(f"{model} Accuracy: {accuracy}")
    
    
# Apply hyperparameter tuning for Gradient Boosting Classifier
param_grid = {'learning_rate': [0.1, 0.01], 'n_estimators': [100, 200]}
cv_model = RandomizedSearchCV(gb_classifier, param_distributions=param_grid, cv=5, n_iter=10)
cv_model.fit(X_train, y_train)

# Evaluate the best hyperparameters on the validation set
best_model = cv_model.best_estimator_
y_val_pred = best_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", val_accuracy)

# Evaluate the final model on the test set
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)

# Print the accuracies of the trained models
for model, accuracy in model_accuracies.items():
    print(f"{model} Accuracy: {accuracy}")



# Save the best model
import pickle
pickle_out = open("gb_classifier1.pkl", "wb")
pickle.dump(best_model, pickle_out)
pickle_out.close()



