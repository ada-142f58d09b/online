import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import talib  # 引入 TA-Lib

# 加载数据
train_data = pd.read_csv('online/TSLA.csv')
test_data = pd.read_csv('online/target.csv')

# 转换日期格式
train_data['Date'] = pd.to_datetime(train_data['Date']).map(pd.Timestamp.toordinal)
test_data['Date'] = pd.to_datetime(test_data['Date']).map(pd.Timestamp.toordinal)

# 合并训练和测试数据以计算技术指标
all_data = pd.concat([train_data, test_data])

# 计算技术指标
all_data['SMA20'] = talib.SMA(all_data['Close'], timeperiod=20)
all_data['RSI14'] = talib.RSI(all_data['Close'], timeperiod=14)

# 处理NaN值（例如通过填充或删除）
all_data.bfill(inplace=True)

# 分割回训练和测试数据
train_data = all_data.iloc[:len(train_data)]
test_data = all_data.iloc[len(train_data):]

# 准备训练和测试数据
X_train = train_data[['Date', 'SMA20', 'RSI14']]
y_train = train_data['Close']
X_test = test_data[['Date', 'SMA20', 'RSI14']]
y_test = test_data['Close']

# 特征标准化（仅对训练数据进行）
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# 对测试数据进行标准化
X_test_scaled = scaler_X.transform(X_test)

# 创建线性回归模型并训练
model = LinearRegression()
model.fit(X_train_scaled, y_train_scaled)

# 进行预测
predictions_scaled = model.predict(X_test_scaled)
predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()

# 计算模型的误差和相似度
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("均方误差 (MSE):", mse)
print("R^2 Score:", r2)

# 绘制预测与实际结果的比较图
plt.figure(figsize=(10, 5))
plt.plot(test_data.index, y_test.values, label='Actual Prices', color='blue')
plt.plot(test_data.index, predictions, label='Predicted Prices', color='red')
plt.title('Stock Price Prediction using Linear Regression')
plt.xlabel('Date')
plt.ylabel('Prices')
plt.legend()
plt.show()