import psycopg2
import pandas as pd

# Establish a connection to the PostgreSQL database
conn = psycopg2.connect(
    host="127.0.0.1",
    port="5432",
    database="postgres",
    user="postgres",
    password="your password"
)

# Create a cursor object to interact with the database
cursor = conn.cursor()

try:
    # Roll back any previous transaction
    conn.rollback()

    # Execute the SQL query to retrieve the data from table df7
    cursor.execute("SELECT * FROM public.df7")

    # Fetch all the rows from the result
    rows = cursor.fetchall()

    # Get the column names from the cursor description
    column_names = [desc[0] for desc in cursor.description]

    # Create a pandas DataFrame from the rows and column names
    df = pd.DataFrame(rows, columns=column_names)

    # Print the DataFrame
    print(df)

    # Commit the transaction
    conn.commit()

except Exception as e:
    # Roll back the transaction if an error occurs
    conn.rollback()
    print("Error:", e)

finally:
    # Close the cursor and the connection
    cursor.close()
    conn.close()

####################################
'''
#Auto EDA library - Sweetviz
import sweetviz as sv

report = sv.analyze(df)
report.show_html()

## Get the current working directory
import os

cwd = os.getcwd()
print("Current Working Directory:", cwd)

'''

#################    Machine learning techniques


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load the data into a DataFrame
data = df
print(data.columns)

# Drop rows with missing values
data.dropna(inplace=True)

# Split the data into features and target
X = data.drop('complications', axis=1)
y = data['complications']


# Perform label encoding for categorical columns
label_encoder = LabelEncoder()
for column in X.select_dtypes(include='object').columns:
    X[column] = label_encoder.fit_transform(X[column])

# Perform standardization of numerical columns
#scaler = StandardScaler()
#X[X.select_dtypes(include=['int64', 'float64']).columns] = scaler.fit_transform(X.select_dtypes(include=['int64', 'float64']))

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define a dictionary to store the model accuracies
model_accuracies = {}

# Train and evaluate Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
dt_pred = dt_classifier.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
model_accuracies['Decision Tree'] = dt_accuracy

# Train and evaluate Logistic Regression
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)
lr_pred = lr_classifier.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
model_accuracies['Logistic Regression'] = lr_accuracy

# Train and evaluate Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
rf_pred = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
model_accuracies['Random Forest'] = rf_accuracy

# Train and evaluate Support Vector Classifier
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
svm_pred = svm_classifier.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
model_accuracies['Support Vector Machine'] = svm_accuracy


#################    Blackbox techniques , Emsembled techniques

#pip install xgboost
#pip install lightgbm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

#Train and evaluate additional classifiers:
#Multilayer Perceptron (Neural Network) Classifier:
mlp_classifier = MLPClassifier()
mlp_classifier.fit(X_train, y_train)
mlp_pred = mlp_classifier.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_pred)
model_accuracies['Multilayer Perceptron'] = mlp_accuracy


#AdaBoost Classifier:
ada_classifier = AdaBoostClassifier()
ada_classifier.fit(X_train, y_train)
ada_pred = ada_classifier.predict(X_test)
ada_accuracy = accuracy_score(y_test, ada_pred)
model_accuracies['AdaBoost'] = ada_accuracy


#Gradient Boosting Classifier:
gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(X_train, y_train)
gb_pred = gb_classifier.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
model_accuracies['Gradient Boosting'] = gb_accuracy


#XGBoost Classifier:
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
# Rest of the code for XGBoost Classifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train, y_train)
xgb_pred = xgb_classifier.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
model_accuracies['XGBoost'] = xgb_accuracy

 
#LightGBM Classifier:   
lgbm_classifier = LGBMClassifier()
lgbm_classifier.fit(X_train, y_train)
lgbm_pred = lgbm_classifier.predict(X_test)
lgbm_accuracy = accuracy_score(y_test, lgbm_pred)
model_accuracies['LightGBM'] = lgbm_accuracy


#Apply ensemble techniques:
#Voting Classifier (Combining multiple models):
voting_classifier = VotingClassifier(
    estimators=[('dt', dt_classifier), ('lr', lr_classifier), ('rf', rf_classifier)],
    voting='hard'
)
voting_classifier.fit(X_train, y_train)
voting_pred = voting_classifier.predict(X_test)
voting_accuracy = accuracy_score(y_test, voting_pred)
model_accuracies['Voting Classifier'] = voting_accuracy


#Print the accuracies of the trained models:
for model, accuracy in model_accuracies.items():
    print(f"{model} Accuracy: {accuracy}")














