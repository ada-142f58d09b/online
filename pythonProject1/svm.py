import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
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
train_data.bfill(inplace=True)
test_data.bfill(inplace=True)

# 准备训练和测试数据
X_train = train_data.drop('Close', axis=1)
y_train = train_data['Close']
X_test = test_data.drop('Close', axis=1)
y_test = test_data['Close']

# 特征标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# 网格搜索寻找最佳参数
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'epsilon': [0.01, 0.1, 0.2, 0.5]
}
grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, y_train_scaled)

# 输出最佳参数
print("Best parameters found: ", grid_search.best_params_)

# 使用最佳参数训练SVR模型
best_model = grid_search.best_estimator_
best_model.fit(X_train_scaled, y_train_scaled)

# 进行预测
predictions_scaled = best_model.predict(X_test_scaled)
predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()

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