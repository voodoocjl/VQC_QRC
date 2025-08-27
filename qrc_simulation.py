#!/usr/bin/env python3
"""
Quantum Reservoir Computing (QRC) Simulation
Refactored from Tutorial.ipynb for easier debugging
"""

import matplotlib.pyplot as plt
import torch as pt
import numpy as np
import pandas as pd
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import time
import pickle

from tqdm import tqdm
import os
import json

# Import utilities
from utilities import *


class QRCSimulator:
    """Quantum Reservoir Computing Simulator"""
    
    def __init__(self, total_spins=6, input_sys_size=1, T_evolution=10):
        """
        Initialize QRC simulator
        
        Args:
            total_spins: Total number of quantum spins
            input_sys_size: Size of input system 
            T_evolution: Time evolution parameter
        """
        self.total_spins = total_spins
        self.input_sys_size = input_sys_size
        self.T_evolution = T_evolution
        
        # Initialize Hilbert spaces
        self._setup_hilbert_spaces()
        
        # Initialize quantum state
        self._setup_initial_state()
        
        # Setup basis change matrices
        self._setup_basis_changes()
        
        # Setup data structure
        self._setup_data_structure()
        
    def _setup_hilbert_spaces(self):
        """Setup Hilbert spaces for the quantum system"""
        self.spin_space_B = HilbertSpace(self.total_spins - self.input_sys_size, 'spin')
        self.spin_space_A = HilbertSpace(self.input_sys_size, 'spin')
        
        hilbert_spaces_dict = {
            'spin_A': self.spin_space_A,
            'spin_B': self.spin_space_B
        }
        self.composite_space = CompositeHilbertSpace(hilbert_spaces_dict)
        
    def _setup_initial_state(self):
        """Setup initial quantum state |000000>"""
        rho_0_spins = np.zeros((2**self.total_spins, 2**self.total_spins), dtype=np.complex128)
        rho_0_spins[0][0] = 1.0 + 1j*0.0
        self.rho_0 = DensityMatrix(rho_0_spins)
        
    def _setup_basis_changes(self):
        """Setup basis change matrices for X, Y measurements"""
        # Hadamard matrix for X basis
        H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)
        H_total = H
        for i in range(self.total_spins - 1):
            H_total = np.kron(H_total, H)
        self.X_basis_change = H_total
        self.X_basis_change_dag = H_total.conj().T
        
        # Phase gate for Y basis
        S = np.array([[1, 0], [0, -np.exp(1j*np.pi/2)]], dtype=complex)
        S_total = S
        for i in range(self.total_spins - 1):
            S_total = np.kron(S_total, S)
        self.Y_basis_change = H_total @ S_total
        self.Y_basis_change_dag = self.Y_basis_change.conj().T
        
    def _setup_data_structure(self):
        """Setup data structure for storing results"""
        column_length_dict = {
            'X': self.total_spins,
            'Y': self.total_spins,
            'Z': self.total_spins,
            'XX': self.total_spins * (self.total_spins + 1) // 2,
            'YY': self.total_spins * (self.total_spins + 1) // 2,
            'ZZ': self.total_spins * (self.total_spins + 1) // 2
        }
        
        self.df_columns = []
        for col in column_length_dict.keys():
            for i in range(column_length_dict[col]):
                self.df_columns.append(col + str(i))
    
    @staticmethod
    def get_couplings(n_spins, J):
        """Generate random coupling matrix"""
        J_bare = np.random.uniform(low=-0.5+J, high=0.5*J, size=(n_spins, n_spins))
        # Exclude self-interaction
        for i in range(n_spins):
            J_bare[i, i] = 0
        # Make symmetric
        J_bare = np.triu(J_bare, k=1)
        J_bare = 0.5 * (J_bare + J_bare.T)
        return J_bare
    
    def get_hamiltonian(self, h=10, J=1):
        """
        Generate reservoir Hamiltonian
        
        Args:
            h: Magnetic field strength
            J: Coupling strength
            
        Returns:
            H_reservoir: Reservoir Hamiltonian matrix
        """
        H_reservoir = np.zeros((self.composite_space.dimension, self.composite_space.dimension))
        J_bare = self.get_couplings(self.total_spins, J)
        
        # XX coupling terms
        for i in range(1, self.total_spins + 1):
            for j in range(i + 1, self.total_spins + 1):
                H_reservoir += J * J_bare[i-1, j-1] * self.composite_space.X[i] @ self.composite_space.X[j]
        
        # Z field terms
        for i in range(1, self.total_spins + 1):
            H_reservoir += 0.5 * h * self.composite_space.Z[i]
            
        return H_reservoir
    
    @staticmethod
    def M_matrix(N, g):
        """Generate measurement backaction matrix"""
        Mi = np.array([[1.0, np.exp(-g**2/2)], [np.exp(-g**2/2), 1]], dtype=np.complex128)
        M = Mi * 1
        for j in range(N - 1):
            M = np.kron(M, Mi)
        return M
    
    def run_simulation(self, time_series, g=0.5, h=1, verbose=True):
        """
        Run QRC simulation for given parameters
        
        Args:
            time_series: Input time series data
            g: Measurement strength parameter
            h: Magnetic field strength
            verbose: Whether to print progress
            
        Returns:
            features_df: DataFrame with extracted features
        """
        if verbose:
            print(f"Running QRC simulation with g={g}, h={h}")
            
        # Initialize Hamiltonian and evolution
        H_reservoir = self.get_hamiltonian(h)
        U_reservoir = scipy.linalg.expm(-1j * H_reservoir * self.T_evolution)
        U_reservoir_dag = U_reservoir.conj().T
        
        # Measurement backaction matrix
        M = self.M_matrix(self.total_spins, g)
        
        # Initialize data storage
        data = {
            'X': [], 'Y': [], 'Z': [],
            'XX': [], 'YY': [], 'ZZ': []
        }
        
        K = len(time_series)
        
        # Process time series for each observable
        for obs in ['X', 'Y', 'Z']:
            if verbose:
                print(f"Processing {obs} observable...")
                
            rho = self.rho_0.dm
            
            for k in tqdm(range(K), disable=not verbose):
                if isinstance(rho, DensityMatrix):
                    rho = rho.dm
                
                # Input encoding
                s_k = time_series[k]
                rho_B = self.composite_space.get_reduced_density_matrix(rho, ['spin_B'])
                rho_A = np.array([
                    [1. - s_k, np.sqrt((1 - s_k) * s_k)],
                    [np.sqrt((1 - s_k) * s_k), s_k]
                ])
                rho = np.kron(rho_A, rho_B)
                
                # Time evolution
                rho = U_reservoir @ rho @ U_reservoir_dag
                
                # Measurement for different observables
                if obs == 'Z':
                    rho = self._measure_Z(rho, M, data)
                elif obs == 'Y':
                    rho = self._measure_Y(rho, M, data)
                elif obs == 'X':
                    rho = self._measure_X(rho, M, data)
        
        # Convert to DataFrame
        data_arr = self._dict_to_arr(data)
        features_df = pd.DataFrame(data_arr, columns=self.df_columns)
        
        if verbose:
            print(f"Simulation completed. Features shape: {features_df.shape}")
            
        return features_df
    
    def _measure_Z(self, rho, M, data):
        """Measure Z observables"""
        # Apply measurement
        rho = np.multiply(M, rho)
        rho = DensityMatrix(rho)
        
        # Single-spin Z expectation values
        ev_Z = []
        for i in range(1, self.total_spins + 1):
            ev_Z.append(rho.get_expectation_value(self.composite_space.Z[i]))
        data['Z'].append(ev_Z)
        
        # Two-spin ZZ correlations
        ev_ZZ = []
        for i in range(1, self.total_spins + 1):
            for j in range(i, self.total_spins + 1):
                ev_ZZ.append(rho.get_expectation_value(
                    self.composite_space.Z[i] @ self.composite_space.Z[j]
                ))
        data['ZZ'].append(ev_ZZ)
        
        return rho.dm
    
    def _measure_Y(self, rho, M, data):
        """Measure Y observables"""
        # Rotate to Y basis
        rho = self.Y_basis_change @ rho @ self.Y_basis_change_dag
        # Apply measurement
        rho = np.multiply(M, rho)
        # Rotate back to Z basis
        rho = self.Y_basis_change_dag @ rho @ self.Y_basis_change
        rho = DensityMatrix(rho)
        
        # Single-spin Y expectation values
        ev_Y = []
        for i in range(1, self.total_spins + 1):
            ev_Y.append(rho.get_expectation_value(self.composite_space.Y[i]))
        data['Y'].append(ev_Y)
        
        # Two-spin YY correlations
        ev_YY = []
        for i in range(1, self.total_spins + 1):
            for j in range(i, self.total_spins + 1):
                ev_YY.append(rho.get_expectation_value(
                    self.composite_space.Y[i] @ self.composite_space.Y[j]
                ))
        data['YY'].append(ev_YY)
        
        return rho.dm
    
    def _measure_X(self, rho, M, data):
        """Measure X observables"""
        # Rotate to X basis
        rho = self.X_basis_change @ rho @ self.X_basis_change_dag
        # Apply measurement
        rho = np.multiply(M, rho)
        # Rotate back to Z basis
        rho = self.X_basis_change_dag @ rho @ self.X_basis_change
        rho = DensityMatrix(rho)
        
        # Single-spin X expectation values
        ev_X = []
        for i in range(1, self.total_spins + 1):
            ev_X.append(rho.get_expectation_value(self.composite_space.X[i]))
        data['X'].append(ev_X)
        
        # Two-spin XX correlations
        ev_XX = []
        for i in range(1, self.total_spins + 1):
            for j in range(i, self.total_spins + 1):
                ev_XX.append(rho.get_expectation_value(
                    self.composite_space.X[i] @ self.composite_space.X[j]
                ))
        data['XX'].append(ev_XX)
        
        return rho.dm
    
    @staticmethod
    def _dict_to_arr(data_dict):
        """Convert data dictionary to structured array"""
        num_lists_per_key = len(data_dict[next(iter(data_dict))])
        concatenated_lists = []
        
        for i in range(num_lists_per_key):
            current_row = []
            for key in data_dict:
                current_row.extend(data_dict[key][i])
            concatenated_lists.append(current_row)
            
        return np.array(concatenated_lists)


class QRCAnalyzer:
    """QRC Performance Analysis Tools"""
    
    @staticmethod
    def compute_capacity(features_df, time_series, eta, f_p=True):
        """
        Compute memory/prediction capacity for given eta
        
        Args:
            features_df: Features DataFrame
            time_series: Input time series
            eta: Time delay parameter
            f_p: True for prediction, False for memory
            
        Returns:
            capacity_train, capacity_test: Training and test capacities
        """
        tot_width = features_df.shape[1]
        N_raw_data = len(features_df)
        N_skip = 20
        data_size = N_raw_data - N_skip
        
        # Prepare data tensors
        X_data = pt.zeros(data_size, tot_width, dtype=float)
        Y_ini_data = pt.zeros(data_size, 1, dtype=float)
        
        for k in range(data_size):
            X_data[k] = pt.tensor(features_df.loc[k, :].values, dtype=float)
            Y_ini_data[k] = time_series[k]
        
        eta = abs(eta)
        if f_p:
            eta = -eta
        
        Y_data = pt.roll(Y_ini_data, -eta)
        
        # Train-test split
        fraction_train = 0.7
        N_train = int(fraction_train * data_size)
        
        X_train = X_data[:N_train, :].float()
        y_train = Y_data[:N_train].float()
        X_test = X_data[-data_size + N_train:, :].float()
        y_test = Y_data[-data_size + N_train:].float()
        
        # Linear regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        y_train = y_train.detach().numpy()
        y_test = y_test.detach().numpy()
        
        # Compute capacities
        capacity_train = np.cov(y_train.T, y_train_pred.T, ddof=1)[0, 1]**2 / (
            np.var(y_train, ddof=1) * np.var(y_train_pred, ddof=1)
        )
        
        capacity_test = np.cov(y_test.T, y_test_pred.T, ddof=1)[0, 1]**2 / (
            np.var(y_test, ddof=1) * np.var(y_test_pred, ddof=1)
        )
        
        return capacity_train, capacity_test
    
    @staticmethod
    def compute_sum_capacity(features_df, time_series, eta_max, f_p=True):
        """
        Compute sum capacity up to eta_max
        
        Args:
            features_df: Features DataFrame
            time_series: Input time series
            eta_max: Maximum eta value
            f_p: True for prediction, False for memory
            
        Returns:
            sum_capacity_test: Sum of test capacities
        """
        capacities_test = []
        
        for eta in range(1, eta_max + 1):
            _, capacity_test = QRCAnalyzer.compute_capacity(
                features_df, time_series, eta, f_p
            )
            capacities_test.append(capacity_test)
        
        return np.sum(capacities_test)


def load_time_series(data_type="santa_fe"):
    """
    Load and preprocess time series data
    
    Args:
        data_type: "santa_fe" or "smt"
        
    Returns:
        time_series: Preprocessed time series
    """
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
        # 读取数据
        data = pd.read_csv('./stock_price/traindata_stock.csv')['Open'].values.reshape(-1, 1)
        # data = pd.read_csv('./stock_price/testdata_stock.csv')['Open'].values.reshape(-1, 1)
        # 初始化归一化器
        train_scaler = MinMaxScaler(feature_range=(0, 1))

        # 对训练数据进行拟合和转换
        train_data_scaled = train_scaler.fit_transform(data)

        # 将归一化后的数据转换回一维数组
        time_series = train_data_scaled.flatten().tolist()
    else:
        raise ValueError("data_type must be 'santa_fe' or 'smt'")    
    
    return time_series


def load_figure_params():
    """Load figure parameters from JSON file"""
    with open('./figure_params.json') as json_file:
        return json.load(json_file)    


def main():
    """Main function for demonstration"""
    print("QRC Simulation Demo")
    print("==================")
    
    # Load time series
    time_series = load_time_series("santa_fe")
    print(f"Loaded time series with {len(time_series)} points")    
    
    # Initialize simulator
    simulator = QRCSimulator()
    
    # Run simulation with default parameters
    g, h = 0.5, 1
    features_df = simulator.run_simulation(time_series, g=g, h=h)
    
    # Analyze performance
    analyzer = QRCAnalyzer()
    
    # Single capacity
    eta = 10
    capacity_train, capacity_test = analyzer.compute_capacity(
        features_df, time_series, eta, f_p=True
    )
    print(f"Capacity for eta = {eta}: {np.round(capacity_test, 4)}")
    
    # Sum capacity
    eta_max = 20
    sum_capacity = analyzer.compute_sum_capacity(
        features_df, time_series, eta_max, f_p=True
    )
    print(f"Sum capacity for eta_max = {eta_max}: {np.round(sum_capacity, 4)}")

def get_data_loaders(dataset_file, batch_size=32):
    """获取数据加载器"""
    with open(dataset_file, 'rb') as file:
        (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor) = pickle.load(file)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def process_schemes_qrc_data(g=0.5, h=1, verbose=True):
    """
    使用QRC处理schemes_qrc_ols中的时间序列数据
    
    Args:
        g: 测量强度参数
        h: 磁场强度参数 
        verbose: 是否显示详细信息
        
    Returns:
        train_features: 训练集的81维特征矩阵
        train_targets: 训练集目标值
        test_features: 测试集的81维特征矩阵
        test_targets: 测试集目标值
    """ 
    
    if verbose:
        print("=== 使用QRC处理schemes_qrc_ols时间序列数据 ===")
        print(f"参数设置: g={g}, h={h}")
    
    # 获取数据加载器
    train_loader, test_loader = get_data_loaders('data/Santa_Fe_2000', batch_size=1)  # batch_size=1方便处理

    # 初始化QRC模拟器
    simulator = QRCSimulator(total_spins=6, input_sys_size=1, T_evolution=10)
    
    # 处理训练集
    print("\n--- 处理训练集 ---")
    train_features_list = []
    train_targets_list = []
    
    train_count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # data shape: [1, 6], target shape: [1]
        time_series = data.squeeze().numpy().flatten()  # 转换为 [6] 的numpy数组
        target_value = target.item()       
        
        # 确保数据在[0,1]范围内（QRC期望的输入范围）
        # time_series_normalized = (time_series - time_series.min()) / (time_series.max() - time_series.min() + 1e-8)
        time_series_normalized = time_series
        
        try:
            # 使用QRC模拟器处理这6个时间点
            features_df = simulator.run_simulation(time_series_normalized, g=g, h=h, verbose=False)
            
            # features_df应该有6行，每行81个特征
            # 我们可以选择不同的聚合策略：
            # 选项1: 取最后一行作为整个序列的特征表示
            # final_features = features_df.iloc[-1].values

            # 所有输入的Z期望
            final_features = features_df.iloc[:, 12:18].values.squeeze().flatten() 
            train_features_list.append(final_features)
            train_targets_list.append(target_value)
            
        except Exception as e:
            if verbose and train_count < 5:  # 只显示前几个错误
                print(f"  样本 {train_count+1} 处理失败: {e}")
            continue
        
        train_count += 1
        if train_count % 100 == 0 and verbose:
            print(f"  已处理 {train_count} 个训练样本")
       
    
    print(f"训练集处理完成，共 {len(train_features_list)} 个有效样本")
    
    # 处理测试集
    print("\n--- 处理测试集 ---")
    test_features_list = []
    test_targets_list = []
    
    test_count = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        # data shape: [1, 6], target shape: [1]
        time_series = data.squeeze().numpy().flatten()  # 转换为 [6] 的numpy数组
        target_value = target.item()
        
        # 确保数据在[0,1]范围内
        # time_series_normalized = (time_series - time_series.min()) / (time_series.max() - time_series.min() + 1e-8)
        time_series_normalized = time_series
        
        try:
            # 使用QRC模拟器处理这6个时间点
            features_df = simulator.run_simulation(time_series_normalized, g=g, h=h, verbose=False)
            
            # 取最后一行作为特征表示
            # final_features = features_df.iloc[-1].values

            final_features = features_df.iloc[:, 12:18].values.squeeze().flatten()
            
            test_features_list.append(final_features)
            test_targets_list.append(target_value)
            
        except Exception as e:
            if verbose and test_count < 5:  # 只显示前几个错误
                print(f"  测试样本 {test_count+1} 处理失败: {e}")
            continue
        
        test_count += 1
        if test_count % 100 == 0 and verbose:
            print(f"  已处理 {test_count} 个测试样本")
    
    print(f"测试集处理完成，共 {len(test_features_list)} 个有效样本")
    
    # 转换为numpy数组
    if len(train_features_list) == 0 or len(test_features_list) == 0:
        raise ValueError("没有成功处理的样本，请检查数据格式和参数设置")
    
    train_features = np.array(train_features_list)  # shape: [N_train, 81]
    train_targets = np.array(train_targets_list)    # shape: [N_train]
    test_features = np.array(test_features_list)    # shape: [N_test, 81]
    test_targets = np.array(test_targets_list)      # shape: [N_test]
    
    if verbose:
        print("\n=== 特征提取完成 ===")
        print(f"训练集特征形状: {train_features.shape}")
        print(f"训练集目标形状: {train_targets.shape}")
        print(f"测试集特征形状: {test_features.shape}")
        print(f"测试集目标形状: {test_targets.shape}")
        print(f"特征维度: {train_features.shape[1]} (应该是81)")
        
        # 显示特征统计信息
        print(f"\n特征统计:")
        print(f"  训练集特征范围: [{train_features.min():.6f}, {train_features.max():.6f}]")
        print(f"  测试集特征范围: [{test_features.min():.6f}, {test_features.max():.6f}]")
        print(f"  训练集目标范围: [{train_targets.min():.6f}, {train_targets.max():.6f}]")
        print(f"  测试集目标范围: [{test_targets.min():.6f}, {test_targets.max():.6f}]")
        
        # 检查是否有无效值
        if np.any(np.isnan(train_features)) or np.any(np.isinf(train_features)):
            print("  ⚠️ 训练集特征中包含NaN或Inf值")
        if np.any(np.isnan(test_features)) or np.any(np.isinf(test_features)):
            print("  ⚠️ 测试集特征中包含NaN或Inf值")
    
    return train_features, train_targets, test_features, test_targets




def load_qrc_features(filepath):
    """
    从文件加载QRC特征数据
    
    Args:
        filepath: 文件路径
        
    Returns:
        train_features, train_targets, test_features, test_targets
    """
    import pickle
    
    if filepath.endswith('.npz'):
        # 加载npz格式
        data = np.load(filepath)
        train_features = data['train_features']
        train_targets = data['train_targets']
        test_features = data['test_features']
        test_targets = data['test_targets']
        
    elif filepath.endswith('.pkl'):
        # 加载pickle格式
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
        train_features = data_dict['train_features']
        train_targets = data_dict['train_targets']
        test_features = data_dict['test_features']
        test_targets = data_dict['test_targets']
        
        # 显示元数据（如果有）
        if 'metadata' in data_dict:
            metadata = data_dict['metadata']
            print("加载的数据信息:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
                
    elif filepath.endswith('.csv'):
        # 加载CSV格式
        df = pd.read_csv(filepath)
        train_df = df[df['split'] == 'train']
        test_df = df[df['split'] == 'test']
        
        # 提取特征和目标
        feature_cols = [col for col in df.columns if col not in ['target', 'split']]
        train_features = train_df[feature_cols].values
        train_targets = train_df['target'].values
        test_features = test_df[feature_cols].values
        test_targets = test_df['target'].values
        
    else:
        raise ValueError("Unsupported file format. Use .npz, .pkl, or .csv")
    
    print(f"✅ 已从 {filepath} 加载数据")
    print(f"训练集: {train_features.shape[0]} 样本, {train_features.shape[1]} 特征")
    print(f"测试集: {test_features.shape[0]} 样本, {test_features.shape[1]} 特征")
    
    return train_features, train_targets, test_features, test_targets


def demo_schemes_qrc_integration(filename):
    """演示QRC与schemes_qrc_ols的集成"""
    print("QRC + schemes_qrc_ols 集成演示")
    print("=" * 50)
    
    # 处理数据  需要手动指定文件名
    train_features, train_targets, test_features, test_targets = process_schemes_qrc_data(
        g=0.5, h=1, verbose=True
    )
        
    # 显示几个样本
    print(f"\n前3个训练样本的特征(前5维):")
    for i in range(min(3, len(train_features))):
        print(f"  样本{i+1}: {train_features[i][:5]} -> 目标: {train_targets[i]:.6f}")
    
    # 保存数据到文件
    print(f"\n" + "="*50)
    save_qrc_features(train_features, train_targets, test_features, test_targets,
                     filename=filename, save_format='npz')

    # 演示加载数据
    print(f"\n" + "="*30)
    print("演示数据加载:")
    loaded_train_features, loaded_train_targets, loaded_test_features, loaded_test_targets = load_qrc_features(filename+".npz")

    # 验证数据一致性
    print(f"\n数据一致性检查:")
    print(f"训练特征一致: {np.array_equal(train_features, loaded_train_features)}")
    print(f"训练目标一致: {np.array_equal(train_targets, loaded_train_targets)}")
    print(f"测试特征一致: {np.array_equal(test_features, loaded_test_features)}")
    print(f"测试目标一致: {np.array_equal(test_targets, loaded_test_targets)}")    


def evaluate_qrc_regression(filename):
    """
    从保存的QRC特征文件中读取数据，使用线性回归进行拟合，并计算capacity
    """
    print("=== QRC特征线性回归评估 ===")
    
    # 1. 加载保存的QRC特征数据
    try:
        train_features, train_targets, test_features, test_targets = load_qrc_features(filename)
    except FileNotFoundError:
        print(f"❌ 未找到 {filename} 文件")
        print("请先运行 demo_schemes_qrc_integration() 生成特征文件")
        return None
    
    print(f"\n数据加载完成:")
    print(f"  训练集: {train_features.shape[0]} 样本, {train_features.shape[1]} 特征")
    print(f"  测试集: {test_features.shape[0]} 样本, {test_features.shape[1]} 特征")
    
    # 2. 使用线性回归拟合训练集
    print(f"\n--- 训练线性回归模型 ---")
    model = LinearRegression()
    model.fit(train_features, train_targets)
    
    print(f"✅ 线性回归模型训练完成")
    print(f"模型系数维度: {model.coef_.shape}")
    print(f"模型截距: {model.intercept_:.6f}")
    
    # 3. 进行预测
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)
    
    print(f"\n--- 预测结果统计 ---")
    print(f"训练集预测范围: [{train_predictions.min():.6f}, {train_predictions.max():.6f}]")
    print(f"测试集预测范围: [{test_predictions.min():.6f}, {test_predictions.max():.6f}]")
    print(f"训练集真值范围: [{train_targets.min():.6f}, {train_targets.max():.6f}]")
    print(f"测试集真值范围: [{test_targets.min():.6f}, {test_targets.max():.6f}]")
    
    # 4. 计算capacity（按照QRC的定义）
    def compute_capacity(y_true, y_pred):
        """
        计算capacity = (Cov(y_true, y_pred))^2 / (Var(y_true) * Var(y_pred))
        这是QRC中使用的capacity定义
        """
        # 确保是numpy数组
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # 计算协方差和方差
        covariance = np.cov(y_true, y_pred, ddof=1)[0, 1]
        var_true = np.var(y_true, ddof=1)
        var_pred = np.var(y_pred, ddof=1)
        
        # 计算capacity
        if var_true == 0 or var_pred == 0:
            return 0.0
        
        capacity = (covariance ** 2) / (var_true * var_pred)
        return capacity
    
    # 5. 计算训练集和测试集的capacity
    train_capacity = compute_capacity(train_targets, train_predictions)
    test_capacity = compute_capacity(test_targets, test_predictions)
    
    print(f"\n=== Capacity 评估结果 ===")
    print(f"训练集 Capacity: {train_capacity:.6f}")
    print(f"测试集 Capacity:  {test_capacity:.6f}")
    
    # 6. 额外的性能指标
    from sklearn.metrics import r2_score, mean_squared_error
    
    train_r2 = r2_score(train_targets, train_predictions)
    test_r2 = r2_score(test_targets, test_predictions)
    train_mse = mean_squared_error(train_targets, train_predictions)
    test_mse = mean_squared_error(test_targets, test_predictions)
    
    print(f"\n=== 额外性能指标 ===")
    print(f"训练集 R²: {train_r2:.6f}")
    print(f"测试集 R²:  {test_r2:.6f}")
    print(f"训练集 MSE: {train_mse:.6f}")
    print(f"测试集 MSE:  {test_mse:.6f}")
    
    # 7. 保存结果
    results = {
        'train_capacity': train_capacity,
        'test_capacity': test_capacity,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'model_coefficients': model.coef_,
        'model_intercept': model.intercept_,
        'train_predictions': train_predictions,
        'test_predictions': test_predictions
    }  
    
    # 8. 可视化结果（可选）
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 训练集预测 vs 真值
        axes[0].scatter(train_targets, train_predictions, alpha=0.6, s=10)
        axes[0].plot([train_targets.min(), train_targets.max()], 
                    [train_targets.min(), train_targets.max()], 'r--', lw=2)
        axes[0].set_xlabel('True Values')
        axes[0].set_ylabel('Predictions')
        axes[0].set_title(f'Training Set\nCapacity = {train_capacity:.4f}')
        axes[0].grid(True, alpha=0.3)
        
        # 测试集预测 vs 真值
        axes[1].scatter(test_targets, test_predictions, alpha=0.6, s=10)
        axes[1].plot([test_targets.min(), test_targets.max()], 
                    [test_targets.min(), test_targets.max()], 'r--', lw=2)
        axes[1].set_xlabel('True Values')
        axes[1].set_ylabel('Predictions')
        axes[1].set_title(f'Test Set\nCapacity = {test_capacity:.4f}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = f"qrc_regression_plot.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 图表已保存到: {plot_file}")
        
    except ImportError:
        print("⚠️ matplotlib未安装，跳过可视化")
    
    return results


# 添加到main函数的调用示例
def test_qrc_evaluation():
    """测试QRC评估功能"""
    print("开始QRC评估测试...")
    
    # 如果特征文件不存在，先生成
    filename = "data/santa_fe_features"
    if not os.path.exists(filename + ".npz"):
        print("特征文件不存在，先生成QRC特征...")
        demo_schemes_qrc_integration(filename)

    # 运行评估
    results = evaluate_qrc_regression(filename + ".npz")
    
    if results:
        print(f"\n🎉 QRC评估完成!")
        print(f"主要结果: 测试集Capacity = {results['test_capacity']:.6f}")


if __name__ == "__main__":
    # 原有的main函数
    # main()

    # demo_schemes_qrc_integration()
    
    # 新增的QRC评估任务
    test_qrc_evaluation()
