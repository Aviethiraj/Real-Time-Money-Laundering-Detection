import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
    auc,
    classification_report
)

# --------------------------------------------------
# 1. LOAD YOUR EXISTING MODEL FILE
# --------------------------------------------------
sys.path.append(r"E:\Download\Avinash")

# Import your trained model (adjust variable name if needed)
from saravanabhava import model  # <-- make sure your model variable is named 'model'

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------
data_path = r"E:\Download\SAML-D.xlsx"
df = pd.read_excel(data_path)

# --------------------------------------------------
# 3. PREPROCESS (ADJUST BASED ON YOUR DATA)
# --------------------------------------------------
# Replace 'target' with your actual column name
target_column = "target"  

X = df.drop(columns=[target_column])
y = df[target_column]

# If encoding/scaling was used in training, apply same steps here!

# --------------------------------------------------
# 4. PREDICTIONS
# --------------------------------------------------
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]  # probability for class 1

# --------------------------------------------------
# 5. CONFUSION MATRIX (HEATMAP)
# --------------------------------------------------
cm = confusion_matrix(y, y_pred)

plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# --------------------------------------------------
# 6. PRECISION-RECALL CURVE (IMPORTANT)
# --------------------------------------------------
precision, recall, _ = precision_recall_curve(y, y_proba)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.show()

# --------------------------------------------------
# 7. ROC CURVE
# --------------------------------------------------
fpr, tpr, _ = roc_curve(y, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# --------------------------------------------------
# 8. FEATURE IMPORTANCE (TOP 10)
# --------------------------------------------------
importances = model.feature_importances_
feature_names = X.columns

feat_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

feat_imp = feat_imp.sort_values(by="importance", ascending=False).head(10)

plt.figure()
plt.barh(feat_imp["feature"], feat_imp["importance"])
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance Score")
plt.show()

# --------------------------------------------------
# 9. CLASSIFICATION REPORT
# --------------------------------------------------
print("\nClassification Report:\n")
print(classification_report(y, y_pred))