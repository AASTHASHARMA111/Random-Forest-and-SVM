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