import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import talib  # 引入 TA-Lib

# 加载数据
train_data = pd.read_csv('online/TSLA.csv')
test_data = pd.read_csv('online/target.csv')

# 转换日期格式
train_data['Date'] = pd.to_datetime(train_data['Date']).map(pd.Timestamp.toordinal)
test_data['Date'] = pd.to_datetime(test_data['Date']).map(pd.Timestamp.toordinal)

# 计算技术指标
train_data['SMA20'] = talib.SMA(train_data['Close'], timeperiod=20)
train_data['RSI14'] = talib.RSI(train_data['Close'], timeperiod=14)
test_data['SMA20'] = talib.SMA(test_data['Close'], timeperiod=20)
test_data['RSI14'] = talib.RSI(test_data['Close'], timeperiod=14)

# 处理NaN值（例如通过填充或删除）
train_data.fillna(method='bfill', inplace=True)
test_data.fillna(method='bfill', inplace=True)

# 准备训练和测试数据
X_train = train_data.drop('Close', axis=1)
y_train = train_data['Close']
X_test = test_data.drop('Close', axis=1)
y_test = test_data['Close']

# 创建随机森林模型并训练
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

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
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Prices')
plt.legend()
plt.show()
