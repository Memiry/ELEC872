import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

print("正在加载数据...")
dfs = []

if os.path.exists('my_data.csv'):
    df_my = pd.read_csv('my_data.csv')
    dfs.append(df_my)
    print(f"- 自录数据: {len(df_my)} 条")

if os.path.exists('mpi_converted_data.csv'):
    df_mpi = pd.read_csv('mpi_converted_data.csv')
    dfs.append(df_mpi)
    print(f"- MPI 数据: {len(df_mpi)} 条")

if not dfs:
    print("错误：没有找到任何 CSV 数据！请先运行前两步。")
    exit()

df_final = pd.concat(dfs, ignore_index=True)
print(f"=== 总数据量: {len(df_final)} ===")
print("数据分布 (0=Bad, 1=Good):")
print(df_final['label'].value_counts())

X = df_final.iloc[:, :-1]
y = df_final.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"模型准确率: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))

joblib.dump(model, 'hybrid_model.pkl')
print("混合模型已保存为 'hybrid_model.pkl'")