import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd
import torch
    

def load_time_series(data_type="santa_fe", train_set=True):    
    if data_type == "santa_fe":
        # Forward prediction task
        time_series_raw = np.load("./sk_Santa_Fe_2000.npy")
        min_ts = min(time_series_raw)        
        max_ts = max(time_series_raw)
        time_series = (time_series_raw + np.abs(min_ts)) / (max_ts - min_ts)
        time_series = time_series.flatten()

    elif data_type == "smt":
        # Memory retrieval task
        time_series = np.load("./time_series_smt.npy")

    elif data_type == "stock":
        if train_set:
            print("Loading training data...")
            # 读取数据
            data = pd.read_csv('./stock_price/traindata_stock.csv')['Open'].values.reshape(-1, 1)
        else:
            print("Loading testing data...")
            data = pd.read_csv('./stock_price/testdata_stock.csv')['Open'].values.reshape(-1, 1)
        # 初始化归一化器
        train_scaler = MinMaxScaler(feature_range=(0, 1))

        # 对训练数据进行拟合和转换
        train_data_scaled = train_scaler.fit_transform(data)

        # 将归一化后的数据转换回一维数组
        time_series = torch.tensor(train_data_scaled, dtype=torch.float32)
    else:
        raise ValueError("data_type must be 'santa_fe' or 'smt'")    
    
    return time_series


def prepare_time_series_data(data, seq_length, eta=1):
    """
    将时间序列转换为监督学习数据集
    :param data: 时间序列数据
    :param seq_length: 输入序列长度
    :return: (X, y) 输入输出对
    """
    sequences = []
    targets = []
    for i in range(len(data) - seq_length - eta + 1):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length + eta - 1])

    return torch.stack(sequences), torch.stack(targets)

stock = load_time_series('stock')
train_x, train_y = prepare_time_series_data(stock, seq_length=6, eta=1)

stock = load_time_series('stock', False)
test_x, test_y = prepare_time_series_data(stock, seq_length=6, eta=1)

with open("data/stock_price_data", "wb") as f:
    pickle.dump((train_x, train_y, test_x, test_y), f)