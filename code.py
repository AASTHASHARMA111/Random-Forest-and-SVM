import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the datasets
train_path = '/content/drive/MyDrive/ResoluteAI/Aastha Sharma Task 1/train.xlsx'
test_path = '/content/drive/MyDrive/ResoluteAI/Aastha Sharma Task 1/test.xlsx'

train_df = pd.read_excel(train_path)
test_df = pd.read_excel(test_path)

# Encode the target variable in the training set
label_encoder = LabelEncoder()
train_df['target'] = label_encoder.fit_transform(train_df['target'])

# Split the training data into features and target
X = train_df.drop('target', axis=1)
y = train_df['target']

# Ensure the test data has the same features as the training data
X_test = test_df[X.columns]

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
rf_model = RandomForestClassifier(random_state=42)
svm_model = SVC(random_state=42)

# Train the Random Forest model
rf_model.fit(X_train, y_train)
y_train_pred_rf = rf_model.predict(X_train)
rf_train_accuracy = accuracy_score(y_train, y_train_pred_rf)

# Train the SVM model
svm_model.fit(X_train, y_train)
y_train_pred_svm = svm_model.predict(X_train)
svm_train_accuracy = accuracy_score(y_train, y_train_pred_svm)

# Evaluate on the validation set
y_val_pred_rf = rf_model.predict(X_val)
rf_val_accuracy = accuracy_score(y_val, y_val_pred_rf)

y_val_pred_svm = svm_model.predict(X_val)
svm_val_accuracy = accuracy_score(y_val, y_val_pred_svm)