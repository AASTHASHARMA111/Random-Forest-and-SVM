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