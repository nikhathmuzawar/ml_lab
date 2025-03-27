import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, accuracy_score,f1_score,confusion_matrix,precision_score
from sklearn.linear_model import LogisticRegression


diabetes=load_diabetes()

df=pd.DataFrame(data=np.c_[diabetes["data"], diabetes["target"]],

columns=diabetes['feature_names']+["target"])


X=diabetes.data

y=diabetes.target


y_binary=(y>140).astype(int) 

X_train, X_test, y_train, y_test=train_test_split(X, y_binary,test_size=0.3, random_state=12)

model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print("computing using sklearn:")
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Confusion_matrix:",confusion_matrix(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))
print("roc:")

TP = np.sum((y_test == 1) & (y_pred == 1))
TN = np.sum((y_test == 0) & (y_pred == 0))
FP = np.sum((y_test == 0) & (y_pred == 1))
FN = np.sum((y_test == 1) & (y_pred == 0))

# Compute metrics manually
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Print results
print("computing manually:")
print("Confusion Matrix:")
print(f"[[{TN} {FP}]\n [{FN} {TP}]]")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1_score:.4f}")

#Plotting ROC and auc

import matplotlib.pyplot as plt

fpr, tpr, thresholds=roc_curve(y_test, y_pred)

roc_auc=auc(fpr, tpr)

#Plot the ROC curve

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area=%a.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], 'k', label='No Skill')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel("False Positive Rate")

plt.ylabel('True Positive Rate')

plt.title("ROC Curve for Diabetes Classification")

plt.legend()

plt.show()