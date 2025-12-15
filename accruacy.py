import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


DATA_CSV = 'my_data.csv'


MODEL_PATH = 'hybrid_model.pkl'

df = pd.read_csv(DATA_CSV)


X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = joblib.load(MODEL_PATH)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"=== 实验结果 ===")
print(f"Accuracy (准确率): {acc*100:.2f}%")
print("\n详细报告:")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Bad (Predicted)', 'Good (Predicted)'],
            yticklabels=['Bad (Actual)', 'Good (Actual)'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()