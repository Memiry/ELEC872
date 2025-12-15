import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


TEST_DATA_CSV = 'my_data.csv'

df = pd.read_csv(TEST_DATA_CSV)
y_true = df['label'].values

y_pred = []

for index, row in df.iterrows():
    nose_y = row['y0']
    shoulder_y = (row['y11'] + row['y12']) / 2


    diff = shoulder_y - nose_y


    if diff < 0.15:
        y_pred.append(0)
    else:
        y_pred.append(1)


acc = accuracy_score(y_true, y_pred)
print(f"=== Experiment Result: Method 1 (Geometric Rule) ===")
print(f"Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Bad (Pred)', 'Good (Pred)'],
            yticklabels=['Bad (Actual)', 'Good (Actual)'])
plt.title('Method 1: Geometric Rule Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()