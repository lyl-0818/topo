"""
train_model.py
功能：训练 ASIC NPU 模型
算法：Gradient Boosting Classifier (适合处理复杂的物理特征边界)
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. 加载数据
print("正在加载数据集...")
try:
    df = pd.read_csv('embodied_ai_data.csv')
except FileNotFoundError:
    print("错误：未找到数据文件，请先运行 generate_dataset.py")
    exit()

# 分离特征和标签
X = df.drop('label', axis=1)
y = df['label']

# 2. 划分数据集 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 训练模型
# 使用 GradientBoosting，因为它能更好地区分重叠特征
print("正在训练 Gradient Boosting 模型 (模拟 ASIC 学习过程)...")
clf = GradientBoostingClassifier(
    n_estimators=200,     # 树的数量
    learning_rate=0.1,    # 学习率
    max_depth=5,          # 树的深度
    random_state=42
)
clf.fit(X_train, y_train)

# 4. 评估模型
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("-" * 40)
print(f"模型训练完成！测试集准确率: {acc*100:.2f}%")
print("-" * 40)
print("分类详细报告:")
print(classification_report(y_test, y_pred, target_names=['正常(Normal)', '故障(Faulty)', '过载(Overload)']))

print("-" * 40)
print("混淆矩阵 (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))
print("-" * 40)

# 5. 保存模型
model_filename = 'asic_model.pkl'
joblib.dump(clf, model_filename)
print(f"模型已固化并保存为: {model_filename}")
print("ASIC NPU 准备就绪。")