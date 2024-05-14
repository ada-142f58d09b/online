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
data = pd.read_excel('EMR.xlsx')  # 确保文件名和路径正确
data['时间 (time)'] = pd.to_datetime(data['时间 (time)'])  # 转换时间列


def apply_wavelet(data, wavelet='db4', level=1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    features = {}
    for i in range(len(coeffs)):
        features[f'wt_mean_{i}'] = np.mean(coeffs[i])
        features[f'wt_std_{i}'] = np.std(coeffs[i])
        features[f'wt_max_{i}'] = np.max(coeffs[i])
        features[f'wt_min_{i}'] = np.min(coeffs[i])
    return features

# 计算小波特征
wavelet_features = []
for i in range(len(data) - 29):  # 假设30个数据点窗口
    window = data['电磁辐射 (EMR)'][i:i + 30].to_numpy()
    features = apply_wavelet(window)
    wavelet_features.append(features)

# 将小波特征转换为DataFrame
wavelet_features_df = pd.DataFrame(wavelet_features)

# 合并特征
data = pd.concat([data.iloc[29:].reset_index(drop=True), wavelet_features_df.reset_index(drop=True)], axis=1)

def apply_fft(data):
    # 确保数据是NumPy数组格式
    if not isinstance(data, np.ndarray):
        data = data.to_numpy()  # 或者使用 data.values
    # 计算快速傅里叶变换
    fft_result = rfft(data)
    # 计算频率
    freqs = rfftfreq(len(data))
    # 寻找最大振幅及其对应的频率
    max_idx = np.argmax(np.abs(fft_result))
    max_freq = freqs[max_idx]
    max_amp = np.abs(fft_result[max_idx])
    return max_freq, max_amp


# 初始化储存最大频率和振幅的列表
max_frequencies = []
max_amplitudes = []

# 遍历每个窗口应用傅里叶变换
for i in range(len(data) - 29):  # 30个数据点的窗口
    window = data['电磁辐射 (EMR)'][i:i + 30]
    freq, amp = apply_fft(window)
    max_frequencies.append(freq)
    max_amplitudes.append(amp)

# 将特征添加到DataFrame中
data['max_frequency'] = pd.Series(max_frequencies, index=data.index[29:])
data['max_amplitude'] = pd.Series(max_amplitudes, index=data.index[29:])

# 计算滑动窗口的平均值和标准差
data['EMR_mean'] = data['电磁辐射 (EMR)'].rolling(window=30).mean()
data['EMR_std'] = data['电磁辐射 (EMR)'].rolling(window=30).std()

# 提取小时信息和周期性编码
data['hour'] = data['时间 (time)'].dt.hour
data['sin_time'] = np.sin(2 * np.pi * data['hour'] / 24)
data['cos_time'] = np.cos(2 * np.pi * data['hour'] / 24)

# 清除NaN值
data.dropna(inplace=True)

# 准备特征和标签
wavelet_feature_columns = [f'wt_mean_{i}' for i in range(2)] + [f'wt_std_{i}' for i in range(2)] + [f'wt_max_{i}' for i in range(2)] + [f'wt_min_{i}' for i in range(2)]
X = data[['电磁辐射 (EMR)', 'EMR_mean', 'EMR_std', 'sin_time', 'cos_time', 'max_frequency', 'max_amplitude'] + wavelet_feature_columns]
y = data['类别 (class)']


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