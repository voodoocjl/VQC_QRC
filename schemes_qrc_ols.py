import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import pickle
import random
from FusionModel import single_enta_to_design, Cell
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime

# 强制禁用CUDA
torch.cuda.is_available = lambda: False

seq_length = 6  # 预测步长
n_layers = 2
n_qubits = 6


def get_param_num(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total:', total_num, 'trainable:', trainable_num)


def display(metrics):
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'

    print(YELLOW + "\nTest NMSE: {:.6f}".format(metrics) + RESET)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_data_loaders(batch_size=32):
    """获取数据加载器"""
    # with open('data/Santa_Fe_200', 'rb') as file:
    #     (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor) = pickle.load(file)

    with open('data/stock_price_data', 'rb') as file:
        (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor) = pickle.load(file)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def nmse(y_true, y_pred):
    """计算归一化均方误差 (NMSE)"""
    return np.mean((y_true - y_pred) ** 2) / np.var(y_true)

def compute_capacity(y_true, y_pred, eps=1e-12):
    """计算容量 = 相关系数的平方"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # 计算协方差和方差
    cov = np.cov(y_true, y_pred, ddof=1)[0, 1]
    var_true = np.var(y_true, ddof=1)
    var_pred = np.var(y_pred, ddof=1)
    
    # 避免除零
    if var_true <= eps or var_pred <= eps:
        return 0.0
    
    # capacity = 相关系数的平方
    capacity = (cov ** 2) / (var_true * var_pred)
    return capacity

def train(model, train_loader, optimizer, criterion, device):
    """训练模型一个epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(-1), target)
        loss.backward()

        # 检查VQC参数的梯度
        check_vqc_gradients(model)

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        
        # 检查参数更新
        check_parameter_updates(model)

        running_loss += loss.item()
        all_preds.append(output.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())

    # 计算整个训练集的NMSE和Capacity
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    train_nmse = nmse(all_targets, all_preds)
    train_capacity = compute_capacity(all_targets, all_preds)

    return running_loss / len(train_loader), train_nmse, train_capacity


def test(model, test_loader, criterion, device):
    """测试模型"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output.squeeze(-1), target)

            running_loss += loss.item()
            all_preds.append(output.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())

    # 计算整个测试集的NMSE
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    test_nmse = nmse(all_targets, all_preds)
    test_capacity = compute_capacity(all_targets, all_preds)
    
    return running_loss / len(test_loader), test_nmse, test_capacity, all_preds, all_targets


def evaluate(model, data_loader, device):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            all_preds.append(output.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = nmse(all_targets, all_preds)
    
    return metrics


def Scheme_eval(design, weight=None, draw=False):
    """评估已训练的模型 - 使用量子特征+OLS"""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
   
    
    result = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 获取数据加载器
    train_loader, test_loader = get_data_loaders(batch_size=32)
    arch_code = [6, 2]
    model = Cell(arch_code, design, 6, 1).to(device)
    
    # 加载预训练权重
    if weight is not None:
        model.load_state_dict(weight, strict=False)       
   
    train_features = []
    train_targets = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 获取量子特征（不是最终预测）
            features = model(data, return_features=True)
            
            train_features.append(features.detach().cpu().numpy())
            train_targets.append(target.detach().cpu().numpy())
    
    train_features = np.concatenate(train_features, axis=0)
    train_targets = np.concatenate(train_targets, axis=0).flatten()  
    
    # 提取测试集特征
    test_features = []
    test_targets = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            # 获取量子特征
            features = model(data, return_features=True)
            
            test_features.append(features.detach().cpu().numpy())
            test_targets.append(target.detach().cpu().numpy())
    
    test_features = np.concatenate(test_features, axis=0)
    test_targets = np.concatenate(test_targets, axis=0).flatten()
    
 
    
    # 第三步：使用最小二乘法训练线性回归器
    regressor = LinearRegression(fit_intercept=True)
    regressor.fit(train_features, train_targets)
    
    # 计算训练集性能
    train_pred = regressor.predict(train_features)
    train_r2 = r2_score(train_targets, train_pred)
    train_nmse = nmse(train_targets, train_pred)
    train_capacity = compute_capacity(train_targets, train_pred)

    # 第四步：测试集评估
    test_pred = regressor.predict(test_features)
    
    # 计算各种指标
    test_r2 = r2_score(test_targets, test_pred)
    test_nmse = nmse(test_targets, test_pred)
    test_capacity = compute_capacity(test_targets, test_pred)
    
    print(f" Train Capacity: {train_capacity:.6f}")
    print(f" Test Capacity: {test_capacity:.6f}")

    
    # 组装结果
    result = {
        'nmse': test_nmse,
        'capacity': test_capacity,
        'r2_score': test_r2,        
        'train_nmse': train_nmse,
        'train_capacity': train_capacity,
        'train_r2': train_r2,
        'regressor': regressor,
        'test_predictions': test_pred,
        'test_targets': test_targets,
        'train_features_shape': train_features.shape,
        'test_features_shape': test_features.shape
    }

     # 可视化结果（可选）
    if draw == True:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 训练集预测 vs 真值
            axes[0].scatter(train_targets, train_pred, alpha=0.6, s=10)
            axes[0].plot([train_targets.min(), train_targets.max()], 
                        [train_targets.min(), train_targets.max()], 'r--', lw=2)
            axes[0].set_xlabel('True Values')
            axes[0].set_ylabel('Predictions')
            axes[0].set_title(f'Training Set\nCapacity = {train_capacity:.4f}')
            axes[0].grid(True, alpha=0.3)
            
            # 测试集预测 vs 真值
            axes[1].scatter(test_targets, test_pred, alpha=0.6, s=10)
            axes[1].plot([test_targets.min(), test_targets.max()], 
                        [test_targets.min(), test_targets.max()], 'r--', lw=2)
            axes[1].set_xlabel('True Values')
            axes[1].set_ylabel('Predictions')
            axes[1].set_title(f'Test Set\nCapacity = {test_capacity:.4f}')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = f"figs/qrc_regression_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.show(block=False)
            plt.pause(10)
            plt.close(fig)
            
            print(f"📊 图表已保存到: {plot_file}")
            
        except ImportError:
            print("⚠️ matplotlib未安装，跳过可视化")
        
    return result


def check_vqc_gradients(model, threshold=1e-8):
    """检查VQC参数的梯度情况"""
    vqc_grad_norms = []
    total_params = 0
    zero_grad_params = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            vqc_grad_norms.append(grad_norm)
            total_params += 1
            
            if grad_norm < threshold:
                zero_grad_params += 1
                
            # 打印关键层的梯度
            if 'quantum' in name.lower() or 'vqc' in name.lower():
                print(f"  {name}: grad_norm = {grad_norm:.8f}")
    
    if zero_grad_params > 0:
        print(f"WARNING: {zero_grad_params}/{total_params} parameters have near-zero gradients")
    
    return vqc_grad_norms


def check_parameter_updates(model, epoch_interval=10):
    """检查参数是否在更新"""
    if not hasattr(check_parameter_updates, 'prev_params'):
        # 存储前一次的参数
        check_parameter_updates.prev_params = {}
        check_parameter_updates.call_count = 0
        
    check_parameter_updates.call_count += 1
    
    if check_parameter_updates.call_count % epoch_interval == 0:
        for name, param in model.named_parameters():
            if name in check_parameter_updates.prev_params:
                param_diff = torch.norm(param.data - check_parameter_updates.prev_params[name]).item()
                if 'quantum' in name.lower() or 'vqc' in name.lower():
                    print(f"  {name}: param_change = {param_diff:.8f}")
            
            check_parameter_updates.prev_params[name] = param.data.clone()

def diagnose_gradient_explosion(model, data_loader, device):
    """诊断梯度爆炸的原因"""
    model.train()
    
    # 检查一个batch的详细情况
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        
        print("=== Gradient Explosion Diagnosis ===")
        print(f"Input data range: [{data.min():.6f}, {data.max():.6f}]")
        print(f"Target range: [{target.min():.6f}, {target.max():.6f}]")
        
        # 检查初始参数
        print("\n--- Initial Parameters ---")
        for name, param in model.named_parameters():
            print(f"{name}: range=[{param.min():.6f}, {param.max():.6f}], norm={param.norm():.6f}")
        
        # 前向传播
        output = model(data)
        print(f"\nModel output range: [{output.min():.6f}, {output.max():.6f}]")
        print(f"Output norm: {output.norm():.6f}")
        
        # 计算损失
        loss = nn.MSELoss()(output.squeeze(-1), target)
        print(f"Loss: {loss.item():.6f}")
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        print("\n--- Gradients ---")
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_max = param.grad.max().item()
                grad_min = param.grad.min().item()
                print(f"{name}: norm={grad_norm:.6f}, range=[{grad_min:.6f}, {grad_max:.6f}]")
                
                # 检查是否有异常大的梯度
                if grad_norm > 1000:
                    print(f"  ⚠️  LARGE GRADIENT in {name}")
                    print(f"  Sample gradient values: {param.grad.flatten()[:5]}")
        
        break  # 只检查第一个batch

def analyze_model_output_diversity(model, data_loader, device, sample_size=100):
    """分析模型输出的多样性"""
    model.eval()
    outputs = []
    
    sample_count = 0
    with torch.no_grad():
        for data, _ in data_loader:
            if sample_count >= sample_size:
                break
                
            data = data.to(device)
            output = model(data)
            outputs.append(output.detach().cpu().numpy())
            sample_count += len(data)
    
    outputs = np.concatenate(outputs)[:sample_size]
    
    print(f"\n=== Model Output Analysis (n={len(outputs)}) ===")
    print(f"Min: {outputs.min():.6f}, Max: {outputs.max():.6f}")
    print(f"Mean: {outputs.mean():.6f}, Std: {outputs.std():.6f}")
    print(f"Unique values (rounded to 6 decimals): {len(np.unique(np.round(outputs, 6)))}")
    print(f"Range: {outputs.max() - outputs.min():.6f}")
    
    # 检查是否接近常数预测
    if outputs.std() < 1e-6:
        print("⚠️  WARNING: Model output has very low variance (near-constant prediction)")
        return False
    else:
        print("✅ Model output shows reasonable diversity")
        return True


def enhanced_train(model, train_loader, optimizer, criterion, device, epoch):
    """增强版训练函数，包含详细监控"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(-1), target)
        loss.backward()

        # 每10个epoch检查一次梯度
        if epoch % 10 == 0 and batch_idx == 0:
            print(f"\n--- Epoch {epoch} Gradient Check ---")
            grad_norms = check_vqc_gradients(model)
            
            if len(grad_norms) > 0:
                avg_grad = np.mean(grad_norms)
                print(f"Average gradient norm: {avg_grad:.8f}")
                
                if avg_grad < 1e-8:
                    print("⚠️  WARNING: Very small gradients detected!")

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        all_preds.append(output.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())

    # 检查输出多样性
    if epoch % 20 == 0:
        analyze_model_output_diversity(model, train_loader, device)

    # 计算指标
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    train_nmse = nmse(all_targets, all_preds)
    train_capacity = compute_capacity(all_targets, all_preds)

    return running_loss / len(train_loader), train_nmse, train_capacity

def Scheme_enhanced(arch_code, design, weight='init', epochs=None, verbs=None, save=None):

    """增强版训练函数，专门用于诊断VQC训练问题"""
    seed = 42
    set_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if epochs is None:
        epochs = 100
    
    train_loader, test_loader = get_data_loaders()
    
    model = Cell(arch_code, design, 6, 1).to(device)
    
    # 打印模型结构
    print("=== Model Architecture ===")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}, requires_grad: {param.requires_grad}")
    
    if weight != 'init':
        if weight != 'base':
            model.load_state_dict(weight, strict=False)
        else:
            model.load_state_dict(torch.load('init_weights/base_qrc'), strict=False)

    # # 诊断梯度爆炸
    # print("=== Diagnosing Gradient Explosion ===")
    # diagnose_gradient_explosion(model, train_loader, device)

    # 分析初始输出
    print("\n=== Initial Model Output ===")
    analyze_model_output_diversity(model, train_loader, device)
    
    criterion = nn.MSELoss()
    
    # 尝试不同的学习率和优化器
    # optimizer = optim.Adam(model.parameters(), lr=0.01)  # 增大学习率
    # 或者尝试 SGD
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)
    
    train_losses = []
    test_losses = []
    train_nmses = []
    test_nmses = []
    train_capacitys = []
    test_capacitys = []
    best_capacity = 0

    start = time.time()
    for epoch in range(epochs):
        # 使用增强版训练函数
        train_loss, train_nmse, train_capacity = enhanced_train(model, train_loader, optimizer, criterion, device, epoch)
            
        train_losses.append(train_loss)
        train_nmses.append(train_nmse)
        train_capacitys.append(train_capacity)
        
        test_loss, test_nmse, test_capacity, test_preds, test_targets = test(model, test_loader, criterion, device)
        test_nmses.append(test_nmse)
        test_losses.append(test_loss)
        test_capacitys.append(test_capacity)

        scheduler.step(train_loss)
        
        combined_score = train_capacity

        if combined_score > best_capacity:
            best_capacity = combined_score
            if not verbs:
                print(f'Epoch [{epoch + 1}/{epochs}], LR: {optimizer.param_groups[0]["lr"]:.6f}, Train Loss: {train_loss:.6f}, Train NMSE: {train_nmse:.6f}, Train Capacity: {train_capacity:.6f}, Test NMSE: {test_nmse:.6f}, Test Capacity: {test_capacity:.6f}, saving model')
            best_model = copy.deepcopy(model)
        else:
            if not verbs:
                print(f'Epoch [{epoch + 1}/{epochs}], LR: {optimizer.param_groups[0]["lr"]:.6f}, Train Loss: {train_loss:.6f}, Train NMSE: {train_nmse:.6f}, Train Capacity: {train_capacity:.6f}, Test NMSE: {test_nmse:.6f}, Test Capacity: {test_capacity:.6f}')

    end = time.time()
    
    # 最终分析
    print("\n=== Final Model Output ===")
    analyze_model_output_diversity(best_model, test_loader, device)
    
    metrics = evaluate(best_model, test_loader, device)
    display(metrics)
    print("Running time: %s seconds" % (end - start))
    
    report = {
        'train_loss_list': train_losses,
        'test_loss_list': test_losses,
        'train_nmse_list': train_nmses,
        'test_nmse_list': test_nmses,
        'train_capacity_list': train_capacitys,
        'test_capacity_list': test_capacitys,
        'best_capacity': best_capacity,
        'metrics': metrics
    }
    
    if save:
        torch.save(best_model.state_dict(), 'init_weights/init_stock_qrc')
    
    return best_model, report

class QuantumEscapeScheduler:
    """量子模型专用的逃逸调度器"""
    
    def __init__(self, optimizer, base_lr=0.001, escape_factor=5.0, 
                 explosion_threshold=1000.0, patience=3, cooldown=5):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.escape_factor = escape_factor  # 逃逸时学习率放大倍数
        self.explosion_threshold = explosion_threshold  # 梯度爆炸阈值
        self.patience = patience  # 连续爆炸多少次触发逃逸
        self.cooldown = cooldown  # 逃逸后的冷却期
        
        # 状态跟踪
        self.explosion_count = 0
        self.escape_mode = False
        self.cooldown_counter = 0
        self.last_grad_norm = 0
        self.last_loss = 0
        
        # 历史记录
        self.grad_history = []
        self.loss_history = []
        
    def step(self, grad_norm, loss_value):
        """根据梯度范数和损失值调整学习率"""
        self.grad_history.append(grad_norm)
        self.loss_history.append(loss_value)
        
        # 保持最近10次的历史
        if len(self.grad_history) > 10:
            self.grad_history.pop(0)
            self.loss_history.pop(0)
        
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # 检测梯度爆炸
        explosion_detected = False
        
        # 方式1：绝对阈值检测
        if grad_norm > self.explosion_threshold:
            explosion_detected = True
            print(f"🔥 Gradient explosion detected: {grad_norm:.2f} > {self.explosion_threshold}")
        
        # 方式2：相对增长检测
        if len(self.grad_history) >= 2:
            recent_avg = np.mean(self.grad_history[-3:-1]) if len(self.grad_history) >= 3 else self.grad_history[-2]
            if grad_norm > 10 * recent_avg and grad_norm > 100:
                explosion_detected = True
                print(f"📈 Rapid gradient increase detected: {grad_norm:.2f} vs recent avg {recent_avg:.2f}")
        
        # 方式3：损失突然增大
        if len(self.loss_history) >= 2:
            prev_loss = self.loss_history[-2]
            if loss_value > 5 * prev_loss and loss_value > 1.0:
                explosion_detected = True
                print(f"💥 Loss explosion detected: {loss_value:.6f} vs {prev_loss:.6f}")
        
        if explosion_detected:
            self.explosion_count += 1
            print(f"爆炸计数: {self.explosion_count}/{self.patience}")
        else:
            self.explosion_count = max(0, self.explosion_count - 1)  # 缓慢衰减
        
        # 决定是否进入逃逸模式
        if self.explosion_count >= self.patience and not self.escape_mode:
            self.escape_mode = True
            self.cooldown_counter = 0
            new_lr = self.base_lr * self.escape_factor
            
            print(f"🚀 ESCAPE MODE ACTIVATED!")
            print(f"   学习率从 {current_lr:.8f} 提升到 {new_lr:.8f}")
            print(f"   逃逸倍数: {self.escape_factor}x")
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
                
        elif self.escape_mode:
            # 逃逸模式中，逐渐冷却
            self.cooldown_counter += 1
            
            if self.cooldown_counter >= self.cooldown:
                # 检查是否已经逃离陷阱
                recent_grad_avg = np.mean(self.grad_history[-3:]) if len(self.grad_history) >= 3 else grad_norm
                
                if recent_grad_avg < self.explosion_threshold / 2:  # 梯度已经稳定
                    self.escape_mode = False
                    self.explosion_count = 0
                    new_lr = self.base_lr
                    
                    print(f"✅ ESCAPE SUCCESSFUL! 梯度已稳定: {recent_grad_avg:.2f}")
                    print(f"   学习率恢复到基础值: {new_lr:.8f}")
                    
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                else:
                    print(f"🔄 继续逃逸中... 当前梯度: {recent_grad_avg:.2f}")
        
        # 正常情况下的温和调整（类似传统调度器）
        elif not self.escape_mode and len(self.loss_history) >= 5:
            recent_losses = self.loss_history[-5:]
            if all(recent_losses[i] >= recent_losses[i+1] for i in range(len(recent_losses)-1)):
                # 损失持续下降，可以稍微提高学习率
                new_lr = min(current_lr * 1.05, self.base_lr * 2)
                if new_lr != current_lr:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"📊 性能良好，学习率小幅提升: {current_lr:.8f} -> {new_lr:.8f}")
        
        self.last_grad_norm = grad_norm
        self.last_loss = loss_value
        
        return self.escape_mode


def quantum_aware_train(model, train_loader, optimizer, criterion, device, epoch, scheduler):
    """量子感知的训练函数"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    epoch_max_grad = 0
    batch_count = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(-1), target)
        
        # 检查loss异常
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️  NaN/Inf loss detected, skipping batch")
            continue
            
        loss.backward()

        # 计算梯度范数（分别计算量子和经典参数）
        quantum_grad_norm = 0
        classical_grad_norm = 0
        total_grad_norm = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_grad_norm += param_norm ** 2
                
                if 'q_params' in name or 'g_raw' in name:
                    quantum_grad_norm += param_norm ** 2
                else:
                    classical_grad_norm += param_norm ** 2
        
        total_grad_norm = total_grad_norm ** 0.5
        quantum_grad_norm = quantum_grad_norm ** 0.5
        classical_grad_norm = classical_grad_norm ** 0.5
        
        epoch_max_grad = max(epoch_max_grad, quantum_grad_norm)
        
        # 使用量子调度器
        escape_mode = scheduler.step(quantum_grad_norm, loss.item())
        
        # 自适应梯度处理
        if escape_mode:
            # 逃逸模式：使用较大的梯度裁剪阈值，允许大步长
            if epoch > 10 and quantum_grad_norm > 50000:  # 只在epoch>10后才跳过极端梯度
                print(f"💀 极端梯度 {quantum_grad_norm:.2f}，跳过此batch")
                optimizer.zero_grad()
                continue
            else:
                # 在逃逸模式下使用更大的裁剪阈值
                if quantum_grad_norm > 50000:  # 训练初期不跳过，但使用强力裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
                    print(f"🚀 逃逸模式(初期): 极大梯度={quantum_grad_norm:.2f}, 使用超强裁剪")
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
                    print(f"🚀 逃逸模式: 梯度={quantum_grad_norm:.2f}, 使用大裁剪阈值")
        else:
            # 正常模式：标准梯度裁剪
            if epoch > 10 and quantum_grad_norm > 10000:  # 只在epoch>10后才跳过大梯度
                print(f"⚠️  大梯度 {quantum_grad_norm:.2f}，跳过此batch")
                optimizer.zero_grad()
                continue
            else:
                # 训练初期不跳过，但使用适当的裁剪
                if quantum_grad_norm > 10000:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    print(f"⚠️  正常模式(初期): 大梯度={quantum_grad_norm:.2f}, 使用强裁剪")
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        batch_count += 1

        running_loss += loss.item()
        all_preds.append(output.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())

    # 计算指标
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        train_nmse = nmse(all_targets, all_preds)
        train_capacity = compute_capacity(all_targets, all_preds)
    else:
        train_nmse = float('inf')
        train_capacity = 0.0
    
    print(f"Epoch {epoch}: 最大梯度={epoch_max_grad:.2f}, 有效batch={batch_count}")

    return running_loss / max(1, len(all_preds)), train_nmse, train_capacity


def Scheme_Quantum_Escape(arch_code, design, weight='init', epochs=None, verbs=None, save=None):
    """使用量子逃逸调度器的训练方案"""
    seed = 42
    set_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if epochs is None:
        epochs = 100
    
    train_loader, test_loader = get_data_loaders(batch_size=16)
    
    model = Cell(arch_code, design, 6, 1).to(device)
    
    print("=== Model Architecture ===")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}, requires_grad: {param.requires_grad}")
    
    if weight != 'init':
        if weight != 'base':
            model.load_state_dict(weight, strict=False)
        else:
            model.load_state_dict(torch.load('init_weights/base_qrc'), strict=False)
    
    criterion = nn.MSELoss()
    
    # 使用中等学习率作为基础
    base_lr = 0.01
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    
    # 创建量子逃逸调度器
    quantum_scheduler = QuantumEscapeScheduler(
        optimizer, 
        base_lr=base_lr,
        escape_factor=3.0,  # 逃逸时学习率放大3倍
        explosion_threshold=1000.0,
        patience=2,  # 连续2次爆炸就触发逃逸
        cooldown=3   # 逃逸3个epoch后检查是否成功
    )
    
    train_losses = []
    test_losses = []
    train_nmses = []
    test_nmses = []
    train_capacitys = []
    test_capacitys = []
    best_capacity = 0
    
    start = time.time()
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1} (Quantum Escape Training) ---")
        
        # 使用量子感知训练
        train_loss, train_nmse, train_capacity = quantum_aware_train(
            model, train_loader, optimizer, criterion, device, epoch, quantum_scheduler
        )
        
        train_losses.append(train_loss)
        train_nmses.append(train_nmse)
        train_capacitys.append(train_capacity)
        
        test_loss, test_nmse, test_capacity, test_preds, test_targets = test(
            model, test_loader, criterion, device
        )
        test_nmses.append(test_nmse)
        test_losses.append(test_loss)
        test_capacitys.append(test_capacity)
        
        combined_score = train_capacity

        if combined_score > best_capacity:
            best_capacity = combined_score
            best_model = copy.deepcopy(model)
            print(f'Epoch [{epoch + 1}/{epochs}], LR: {optimizer.param_groups[0]["lr"]:.6f}, Train Loss: {train_loss:.6f}, Train NMSE: {train_nmse:.6f}, Train Capacity: {train_capacity:.6f}, Test NMSE: {test_nmse:.6f}, Test Capacity: {test_capacity:.6f}, saving model')
        else:
            print(f'Epoch [{epoch + 1}/{epochs}], LR: {optimizer.param_groups[0]["lr"]:.6f}, Train Loss: {train_loss:.6f}, Train NMSE: {train_nmse:.6f}, Train Capacity: {train_capacity:.6f}, Test NMSE: {test_nmse:.6f}, Test Capacity: {test_capacity:.6f}')

    end = time.time()
    
    metrics = evaluate(best_model, test_loader, device)
    display(metrics)
    print("Running time: %s seconds" % (end - start))
    
    return best_model, {
        'train_loss_list': train_losses,
        'test_loss_list': test_losses,
        'train_nmse_list': train_nmses,
        'test_nmse_list': test_nmses,
        'train_capacity_list': train_capacitys,
        'test_capacity_list': test_capacitys,
        'best_capacity': best_capacity,
        'nmse': metrics
    }


if __name__ == '__main__':
    single = [[i] + [1] * 1 * n_layers for i in range(1, n_qubits + 1)]
    enta = [[i] + [i + 1] * n_layers for i in range(1, n_qubits)] + [[n_qubits] + [1] * n_layers]

    arch_code = [6, 2]
    design = single_enta_to_design(single, enta, (n_qubits, n_layers))
    
    # 使用量子逃逸调度器
    print("开始量子逃逸训练...")
    # best_model, report = Scheme_Quantum_Escape(arch_code, design, 'init', 100)
    best_model, report = Scheme_enhanced(arch_code, design, 'init', 200, None, 'save')

    result = Scheme_eval(design, weight=best_model.state_dict())

    # # 保存最佳模型
    # torch.save(best_model.state_dict(), 'quantum_escape_model.pth')
    

   
