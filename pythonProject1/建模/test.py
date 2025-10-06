import pandas as pd
import numpy as np
import pywt
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# 读取数据
data = pd.read_excel('EMR1.xlsx', skiprows=1)  # 确保文件名和路径正确
data['时间 (time)'] = pd.to_datetime(data['时间 (time)'])  # 转换时间列

def apply_wavelet(data, wavelet='db4', level=1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    features = {}
    for i, coeff in enumerate(coeffs):
        features[f'wt_mean_{i}'] = np.mean(coeff)
        features[f'wt_std_{i}'] = np.std(coeff)
        features[f'wt_max_{i}'] = np.max(coeff)
        features[f'wt_min_{i}'] = np.min(coeff)
    return features
    

# 计算小波特征
wavelet_features = []
for i in range(len(data) - 29):  # 假设30个数据点窗口
    window = data['电磁辐射 (EMR)'][i:i + 30].to_numpy()
    features = apply_wavelet(window)
    wavelet_features.append(features)

# 将小波特征转换为DataFrame
wavelet_features_df = pd.DataFrame(wavelet_features)
data = pd.concat([data.iloc[29:].reset_index(drop=True), wavelet_features_df.reset_index(drop=True)], axis=1)

def apply_fft(data):
    if not isinstance(data, np.ndarray):
        data = data.to_numpy()
    fft_result = rfft(data)
    freqs = rfftfreq(len(data))
    max_idx = np.argmax(np.abs(fft_result))
    max_freq = freqs[max_idx]
    max_amp = np.abs(fft_result[max_idx])
    return max_freq, max_amp

# 应用FFT并提取特征
max_frequencies, max_amplitudes = [], []
for i in range(len(data) - 29):
    window = data['电磁辐射 (EMR)'][i:i + 30]
    freq, amp = apply_fft(window)
    max_frequencies.append(freq)
    max_amplitudes.append(amp)

data['max_frequency'] = pd.Series(max_frequencies, index=data.index[29:])
data['max_amplitude'] = pd.Series(max_amplitudes, index=data.index[29:])
data['AE_mean'] = data['电磁辐射 (EMR)'].rolling(window=30).mean()
data['AE_std'] = data['电磁辐射 (EMR)'].rolling(window=30).std()

# 清除NaN值
data.dropna(inplace=True)

# 准备特征
features = ['电磁辐射 (EMR)', 'AE_mean', 'AE_std', 'max_frequency', 'max_amplitude'] + [f'wt_mean_{i}' for i in range(2)] + [f'wt_std_{i}' for i in range(2)] + [f'wt_max_{i}' for i in range(2)] + [f'wt_min_{i}' for i in range(2)]
X = data[features]

# 特征归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用Isolation Forest检测异常
iso_forest = IsolationForest(n_estimators=100, random_state=42, contamination=0.01)  # 假设1%的点为异常
iso_forest.fit(X_scaled)
scores = iso_forest.decision_function(X_scaled)
outliers = iso_forest.predict(X_scaled) == -1

# 检测到的异常时间点
outlier_times = data[outliers]['时间 (time)']

# 对异常时间点进行分组，判断连续的异常时间点
outlier_intervals = outlier_times.groupby((outlier_times.diff() > pd.Timedelta('2min')).cumsum()).agg(['min', 'max'])

# 选择最早的5个时间区间
earliest_intervals = outlier_intervals.head(5)

# 输出结果
print("Earliest Anomaly Intervals:")
print(earliest_intervals)