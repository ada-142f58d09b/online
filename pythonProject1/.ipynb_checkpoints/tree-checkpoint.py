import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import talib
import matplotlib.pyplot as plt

# 数据预处理 - 指数平滑
def exponential_smoothing(series, alpha):
    result = [series[0]]  # 第一个值作为初始值
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

# 加载数据
data = pd.read_csv('TSLA.csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)

# 计算技术指标
data['Adj Close Smooth'] = exponential_smoothing(data['Adj Close'], 0.95)
data['RSI'] = talib.RSI(data['Adj Close Smooth'].values, timeperiod=14)
data['MACD'], data['MACD Signal'], data['MACD Hist'] = talib.MACD(data['Adj Close Smooth'].values, fastperiod=12, slowperiod=26, signalperiod=9)

# 清理数据
data.dropna(inplace=True)

# 特征和标签
X = data[['RSI', 'MACD', 'MACD Signal', 'MACD Hist']]
y = np.sign(data['Adj Close'].diff().shift(-30)).replace(-1, 0)  # 30天后的价格变化（涨：1，跌：0）

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 构建随机森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测
predictions = model.predict(X_test)

# 性能评估
accuracy = accuracy_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f'Accuracy: {accuracy}, ROC AUC: {roc_auc}')

# 绘制ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
