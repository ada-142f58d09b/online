import clf
import pandas as pd
import pywt
import numpy as np
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import compute_class_weight

# 读取数据
data = pd.read_excel('EMR.xlsx')  # 使用正确的文件名
data['时间 (time)'] = pd.to_datetime(data['时间 (time)'])  # 转换时间列
window = 30  # 定义滑动窗口大小
# 计算趋势特征函数
def trend_features(data, window=30):
    return {
        'mean': data.mean(),
        'std': data.std(),
        'max': data.max(),
        'min': data.min(),
        'slope': np.polyfit(range(len(data)), data, 1)[0]  # 计算斜率
    }

# 提取趋势特征
trend_features_list = []
for i in range(len(data) - window + 1):
    window_data = data['电磁辐射 (EMR)'][i:i+window]
    features = trend_features(window_data)
    trend_features_list.append(features)

# 转换为DataFrame
trend_features_df = pd.DataFrame(trend_features_list)

# 合并数据和特征
data = pd.concat([data.iloc[window-1:].reset_index(drop=True), trend_features_df.reset_index(drop=True)], axis=1)

# 清除NaN值
data.dropna(inplace=True)

# 准备特征和标签
X = data[['mean', 'std', 'max', 'min', 'slope']]
y = data['类别 (class)']  # 确保标签列名正确

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特征归一化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 计算类权重
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights_dict = dict(zip(np.unique(y_train), class_weights))

# 创建随机森林模型，应用类权重
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=weights_dict)
rf_classifier.fit(X_train_scaled, y_train)

# 进行预测
y_pred = rf_classifier.predict(X_test_scaled)

# 评估模型
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

