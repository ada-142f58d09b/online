import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 读取数据
data = pd.read_excel('EMR1.xlsx', skiprows=1)  # 确保文件名和路径正确
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

# 准备特征
features = ['mean', 'std', 'max', 'min', 'slope']
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
outlier_intervals = outlier_times.groupby((outlier_times.diff() > pd.Timedelta('12min')).cumsum()).agg(['min', 'max'])

# 选择最早的5个时间区间
earliest_intervals = outlier_intervals.head(5)

# 输出结果
print("Earliest Anomaly Intervals:")
print(earliest_intervals)
