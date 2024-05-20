import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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

# 合并训练和测试数据以计算技术指标
all_data = pd.concat([train_data, test_data])

# 计算技术指标
all_data['SMA20'] = talib.SMA(all_data['Close'], timeperiod=20)
all_data['RSI14'] = talib.RSI(all_data['Close'], timeperiod=14)
all_data['MACD'], all_data['MACD_signal'], all_data['MACD_hist'] = talib.MACD(all_data['Close'])
all_data['BB_upper'], all_data['BB_middle'], all_data['BB_lower'] = talib.BBANDS(all_data['Close'])
all_data['WILLR'] = talib.WILLR(all_data['High'], all_data['Low'], all_data['Close'], timeperiod=14)
all_data['ATR'] = talib.ATR(all_data['High'], all_data['Low'], all_data['Close'], timeperiod=14)

# 处理NaN值（例如通过填充或删除）
all_data.bfill(inplace=True)

# 分割回训练和测试数据
train_data = all_data.iloc[:len(train_data)]
test_data = all_data.iloc[len(train_data):]

# 准备训练和测试数据
features = ['Date', 'SMA20', 'RSI14', 'MACD', 'MACD_signal', 'MACD_hist', 'BB_upper', 'BB_middle', 'BB_lower', 'WILLR', 'ATR']
X_train = train_data[features]
y_train = train_data['Close']
X_test = test_data[features]
y_test = test_data['Close']

# 特征标准化（仅对训练数据进行）
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
X_test_scaled = scaler_X.transform(X_test)

# 1. SVR模型
# 网格搜索寻找最佳参数
param_grid_svr = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'epsilon': [0.01, 0.1, 0.2, 0.5]
}
grid_search_svr = GridSearchCV(SVR(kernel='rbf'), param_grid_svr, cv=5, scoring='r2')
grid_search_svr.fit(X_train_scaled, y_train_scaled)
best_model_svr = grid_search_svr.best_estimator_
best_model_svr.fit(X_train_scaled, y_train_scaled)
predictions_svr_scaled = best_model_svr.predict(X_test_scaled)
predictions_svr = scaler_y.inverse_transform(predictions_svr_scaled.reshape(-1, 1)).ravel()

# 2. 随机森林模型
# 网格搜索寻找最佳参数
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=5, scoring='r2')
grid_search_rf.fit(X_train, y_train)
best_model_rf = grid_search_rf.best_estimator_
best_model_rf.fit(X_train, y_train)
predictions_rf = best_model_rf.predict(X_test)

# 3. 线性回归模型
model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train_scaled)
predictions_lr_scaled = model_lr.predict(X_test_scaled)
predictions_lr = scaler_y.inverse_transform(predictions_lr_scaled.reshape(-1, 1)).ravel()

# 计算模型的误差和相似度
mse_svr = mean_squared_error(y_test, predictions_svr)
r2_svr = r2_score(y_test, predictions_svr)
print("SVR - 均方误差 (MSE):", mse_svr)
print("SVR - R^2 Score:", r2_svr)

mse_rf = mean_squared_error(y_test, predictions_rf)
r2_rf = r2_score(y_test, predictions_rf)
print("随机森林 - 均方误差 (MSE):", mse_rf)
print("随机森林 - R^2 Score:", r2_rf)

mse_lr = mean_squared_error(y_test, predictions_lr)
r2_lr = r2_score(y_test, predictions_lr)
print("线性回归 - 均方误差 (MSE):", mse_lr)
print("线性回归 - R^2 Score:", r2_lr)

# 绘制预测与实际结果的比较图
plt.figure(figsize=(10, 5))
plt.plot(test_data.index, y_test.values, label='Actual Prices', color='blue')
plt.plot(test_data.index, predictions_svr, label='SVR Predicted Prices', color='red')
plt.plot(test_data.index, predictions_rf, label='Random Forest Predicted Prices', color='green')
plt.plot(test_data.index, predictions_lr, label='Linear Regression Predicted Prices', color='orange')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Prices')
plt.legend()
plt.show()