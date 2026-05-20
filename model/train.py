import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

import joblib

df = pd.read_csv("attention_training_data.csv")

#explore the data
print(df.head())
print(df.shape)
print(df.columns)
print(df['label'].value_counts())

#clean the data
df = df.drop('timestamp',axis = 1)
df = df.drop('frame_id', axis = 1)

#X contains all columns except label and y contains only label column
X = df.drop('label', axis = 1)
y = df['label']

print(X.shape)
print(y.shape)  
print(X.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_scaled)
print(X_scaled.mean(axis = 0))
print(X_scaled.std(axis = 0))



model = SVC(kernel= 'rbf', C = 10, gamma = 'scale', probability = True, class_weight = 'balanced')
model.fit(X_scaled, y_train)
print("model trained successfully")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy: {accuracy}")
print("classification report:")
print(classification_report(y_test, y_pred))
print("confusion matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm) 

class_names = ['Attentive', 'Left/Right', 'Down Still', 'Writting']

sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = class_names, yticklabels = class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

joblib.dump(model, 'attention_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("model and scaler saved successfully")