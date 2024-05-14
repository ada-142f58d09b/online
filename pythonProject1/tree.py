import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import talib

# 加载数据
data = pd.read_csv('online/TSLA.csv')

# 转换日期格式
data['Date'] = pd.to_datetime(data['Date']).map(pd.Timestamp.toordinal)

# 计算技术指标
data['SMA20'] = talib.SMA(data['Close'], timeperiod=20)
data['RSI14'] = talib.RSI(data['Close'], timeperiod=14)

# 处理NaN值（例如通过填充或删除）
data.fillna(method='bfill', inplace=True)

# 划分训练集和测试集（这里我们用全部数据来训练模型）
X_train = data.drop(['Close'], axis=1)
y_train = data['Close']

# 创建随机森林模型并训练
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 加载目标测试数据（用于模型效果对比）
target_data = pd.read_csv('online/target.csv')
target_data['Date'] = pd.to_datetime(target_data['Date']).map(pd.Timestamp.toordinal)

# 应用同样的技术指标到测试数据
target_data['SMA20'] = talib.SMA(target_data['Close'], timeperiod=20)
target_data['RSI14'] = talib.RSI(target_data['Close'], timeperiod=14)
target_data.fillna(method='bfill', inplace=True)

X_test = target_data.drop(['Close'], axis=1)
y_test = target_data['Close']

# 进行预测
predictions = model.predict(X_test)

# 计算模型的误差和相似度
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("均方误差 (MSE):", mse)
print("R^2 Score:", r2)

# 绘制预测与实际结果的比较图
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual Prices', color='blue')
plt.plot(y_test.index, predictions, label='Predicted Prices', color='red')
plt.title('Stock Price Prediction for 2024')
plt.xlabel('Date')
plt.ylabel('Prices')
plt.legend()
plt.show()
