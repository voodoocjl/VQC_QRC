import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from math import pi
import torch.nn.functional as F
from torchquantum.encoding import encoder_op_list_name_dict
import numpy as np

def gen_arch(change_code, base_code):        # start from 1, not 0
    # arch_code = base_code[1:] * base_code[0]
    n_qubits = base_code[0]    
    arch_code = ([i for i in range(2, n_qubits+1, 1)] + [1]) * base_code[1]
    if change_code != None:
        if type(change_code[0]) != type([]):
            change_code = [change_code]

        for i in range(len(change_code)):
            q = change_code[i][0]  # the qubit changed
            for id, t in enumerate(change_code[i][1:]):
                arch_code[q - 1 + id * n_qubits] = t
    return arch_code

def prune_single(change_code):
    single_dict = {}
    single_dict['current_qubit'] = []
    if change_code != None:
        if type(change_code[0]) != type([]):
            change_code = [change_code]
        length = len(change_code[0])
        change_code = np.array(change_code)
        change_qbit = change_code[:,0] - 1
        change_code = change_code.reshape(-1, length)    
        single_dict['current_qubit'] = change_qbit
        j = 0
        for i in change_qbit:            
            single_dict['qubit_{}'.format(i)] = change_code[:, 1:][j].reshape(-1, 2).transpose(1,0)
            j += 1
    return single_dict

def translator(single_code, enta_code, trainable, arch_code, fold=1):
    single_code = qubit_fold(single_code, 0, fold)
    enta_code = qubit_fold(enta_code, 1, fold)
    n_qubits = arch_code[0]
    n_layers = arch_code[1]

    updated_design = {}
    updated_design = prune_single(single_code)
    net = gen_arch(enta_code, arch_code) 

    if trainable == 'full' or enta_code == None:
        updated_design['change_qubit'] = None
    else:
        if type(enta_code[0]) != type([]): enta_code = [enta_code]
        updated_design['change_qubit'] = enta_code[-1][0]

    # number of layers
    updated_design['n_layers'] = n_layers

    for layer in range(updated_design['n_layers']):
        # categories of single-qubit parametric gates
        for i in range(n_qubits):
            updated_design['rot' + str(layer) + str(i)] = 'U3'
        # categories and positions of entangled gates
        for j in range(n_qubits):
            if net[j + layer * n_qubits] > 0:
                updated_design['enta' + str(layer) + str(j)] = ('CU3', [j, net[j + layer * n_qubits]-1])
            else:
                updated_design['enta' + str(layer) + str(j)] = ('CU3', [abs(net[j + layer * n_qubits])-1, j])

    updated_design['total_gates'] = updated_design['n_layers'] * n_qubits * 2
    return updated_design

def single_enta_to_design(single, enta, arch_code, fold=1):
    """
    Generate a design list usable by QNET from single and enta codes

    Args:
        single: Single-qubit gate encoding, format: [[qubit, gate_config_layer0, gate_config_layer1, ...], ...]
                Each two bits of gate_config represent a layer: 00=Identity, 01=U3, 10=data, 11=data+U3
        enta: Two-qubit gate encoding, format: [[qubit, target_layer0, target_layer1, ...], ...]
              Each value represents the target qubit position in that layer
        arch_code_fold: [n_qubits, n_layers]

    Returns:
        design: List containing quantum circuit design info, each element is (gate_type, [wire_indices], layer)
    """
    design = []
    single = qubit_fold(single, 0, fold)
    enta = qubit_fold(enta, 1, fold)

    n_qubits, n_layers = arch_code

    # Process each layer
    for layer in range(n_layers):
        # First process single-qubit gates
        for qubit_config in single:
            qubit = qubit_config[0] - 1  # Convert to 0-based index
            # The config for each layer is at position: 1 + layer*2 and 1 + layer*2 + 1
            config_start_idx = 1 + layer 
            if config_start_idx + 1 < len(qubit_config):
                gate_config = f"{qubit_config[config_start_idx]}"

                if gate_config == '1':  # U3
                    design.append(('U3', [qubit], layer))                
                # 0 (Identity) skip

        # Then process two-qubit gates
        for qubit_config in enta:
            control_qubit = qubit_config[0] - 1  # Convert to 0-based index
            # The target qubit position in the list: 1 + layer
            target_idx = 1 + layer
            if target_idx < len(qubit_config):
                target_qubit = qubit_config[target_idx] - 1  # Convert to 0-based index

                # If control and target qubits are different, add C(U3) gate
                if control_qubit != target_qubit:
                    design.append(('C(U3)', [control_qubit, target_qubit], layer))
                # If same, skip (equivalent to Identity)

    return design

def cir_to_matrix(x, y, arch_code, fold=1):
    # x = qubit_fold(x, 0, fold)
    # y = qubit_fold(y, 1, fold)

    qubits = int(arch_code[0] / fold)
    layers = arch_code[1]
    entangle = gen_arch(y, [qubits, layers])
    entangle = np.array([entangle]).reshape(layers, qubits).transpose(1,0)
    single = np.ones((qubits, 2*layers))
    # [[1,1,1,1]
    #  [2,2,2,2]
    #  [3,3,3,3]
    #  [0,0,0,0]]

    if x != None:
        if type(x[0]) != type([]):
            x = [x]    
        x = np.array(x)
        index = x[:, 0] - 1
        index = [int(index[i]) for i in range(len(index))]
        single[index] = x[:, 1:]
    arch = np.insert(single, [(2 * i) for i in range(1, layers+1)], entangle, axis=1)
    return arch.transpose(1, 0)

def qubit_fold(jobs, phase, fold=1):
    if fold > 1:
        job_list = []
        for job in jobs:
            q = job[0]
            if phase == 0:
                job_list.append([2*q] + job[1:])
                job_list.append([2*q-1] + job[1:])
            else:
                job_1 = [2*q]
                job_2 = [2*q-1]
                for k in job[1:]:
                    if q < k:
                        job_1.append(2*k)
                        job_2.append(2*k-1)
                    elif q > k:
                        job_1.append(2*k-1)
                        job_2.append(2*k)
                    else:
                        job_1.append(2*q)
                        job_2.append(2*q-1)
                job_list.append(job_1)
                job_list.append(job_2)
    else:
        job_list = jobs
    return job_list

class TQLayer(tq.QuantumModule):
    def __init__(self, arguments, design, seq_length, n_class):
        super().__init__()
        self.args = arguments
        self.design = design
        self.n_wires = self.args.n_qubits
        
        self.uploading = [tq.GeneralEncoder(self.data_uploading(i)) for i in range(10)]

        self.q_params_rot = nn.Parameter(pi * torch.rand(self.args.n_layers, self.args.n_qubits, 3))  # each U3 gate needs 3 parameters
        self.q_params_enta = nn.Parameter(pi * torch.rand(self.args.n_layers, self.args.n_qubits, 3))  # each CU3 gate needs 3 parameters        
               
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.seq_length = seq_length
        self.fc = nn.Linear(seq_length*self.n_wires, n_class)  

    def data_uploading(self, qubit):
        input = [
            {"input_idx": [0], "func": "ry", "wires": [qubit]},
            {"input_idx": [1], "func": "rz", "wires": [qubit]},
            {"input_idx": [2], "func": "rx", "wires": [qubit]},
            {"input_idx": [3], "func": "ry", "wires": [qubit]},
        ]
        return input

    def forward(self, x):
        bsz = x.shape[0]
        kernel_size = self.args.kernel        
        if not self.args.task.startswith('QML'):
            x = F.avg_pool2d(x, kernel_size)  # 'down_sample_kernel_size' = 6
            if kernel_size == 4:
                x = x.view(bsz, 6, 6)
                tmp = torch.cat((x.view(bsz, -1), torch.zeros(bsz, 4)), dim=-1)
                x = tmp.reshape(bsz, -1, 10).transpose(1, 2)
            else:
                x = x.view(bsz, 4, 4).transpose(1, 2)
        else:
            x = x.view(bsz, self.n_wires, -1)

        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)

        
        for i in range(len(self.design)):
            if self.design[i][0] == 'U3':                
                layer = self.design[i][2]
                qubit = self.design[i][1][0]
                params = self.q_params_rot[layer][qubit].unsqueeze(0)  # 重塑为 [1, 3]
                tqf.u3(qdev, wires=self.design[i][1], params=params)
            elif self.design[i][0] == 'C(U3)':               
                layer = self.design[i][2]
                control_qubit = self.design[i][1][0]
                params = self.q_params_enta[layer][control_qubit].unsqueeze(0)  # 重塑为 [1, 3]
                tqf.cu3(qdev, wires=self.design[i][1], params=params)
            else:   # data uploading: if self.design[i][0] == 'data'
                j = int(self.design[i][1][0])
                self.uploading[j](qdev, x[:,j])

        return self.measure(qdev)

class QNet(nn.Module):
    def __init__(self, arguments, design):
        super(QNet, self).__init__()
        self.args = arguments
        self.design = design
        self.QuantumLayer = TQLayer(self.args, self.design)

    def forward(self, x_image, n_qubits, task_name):
        # exp_val = self.QuantumLayer(x_image, n_qubits, task_name)
        exp_val = self.QuantumLayer(x_image)
        output = F.log_softmax(exp_val, dim=1)        
        return output
    
class Cell(nn.Module):
    def __init__(self, arch, design, seq_length,n_class, g=1):
        super().__init__()
        self.n_wires = arch[0]
        self.n_layers = arch[1]
        self.design = design
        self.seq_length = seq_length

        # seq_length = 1
        self.fc = nn.Linear(seq_length*self.n_wires, n_class)

        self.q_params_rot = nn.Parameter(pi * torch.rand(self.n_layers, self.n_wires, 3))  # each U3 gate needs 3 parameters
        self.q_params_enta = nn.Parameter(pi * torch.rand(self.n_layers, self.n_wires, 3))  # each CU3 gate needs 3 parameters

        # 使用参数化来确保g始终为正数
        # g = softplus(g_raw) = log(1 + exp(g_raw))，确保g > 0
        self.g_raw = nn.Parameter(torch.tensor(np.log(np.exp(g) - 1), dtype=torch.float32))
        
        self.obs = [tq.PauliZ() for i in range(self.n_wires)]


    @property
    def g(self):
        """确保g始终为正数的属性"""
        return torch.nn.functional.softplus(self.g_raw)

    def get_M_matrix(self):
         # 使用可训练参数g
        g = self.g
        # 创建矩阵，避免就地操作
        exp_term = torch.exp(-g ** 2 / 2)
        
        # 构建矩阵元素，确保梯度连接
        Mi = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.complex64)
        Mi = Mi.clone()  # 确保可修改
        Mi[0, 1] = exp_term
        Mi[1, 0] = exp_term
        
        # 对所有位作用
        M = Mi  # 第一个量子比特：弱测量
        for _ in range(self.n_wires - 1):
            M = torch.kron(M, Mi)  
        return M

    

    def vqc_density_input(self, batch_rho):
        """批量处理VQC变换"""
        batch_size = batch_rho.shape[0]
        
        # 创建批量量子设备
        qdev = tq.NoiseDevice(n_wires=self.n_wires, bsz=batch_size)
        
        # 设置批量密度矩阵到设备
        # 需要根据设备的期望格式重塑密度矩阵
        target_shape = qdev.density.shape  # [2, 2, ..., 2]
        
        # 将2D密度矩阵重塑为设备期望的多维格式 - 使用clone确保安全
        batch_densities = []
        for i in range(batch_size):
            rho_reshaped = batch_rho[i].clone().reshape(target_shape)  # 去掉batch维度并克隆
            batch_densities.append(rho_reshaped)
        
        qdev.densities = torch.stack(batch_densities)
        
        # 应用量子电路
        for i in range(len(self.design)):
            if self.design[i][0] == 'U3':                
                layer = self.design[i][2]
                qubit = self.design[i][1][0]
                params = self.q_params_rot[layer][qubit].unsqueeze(0)
                tqf.u3(qdev, wires=[qubit], params=params)
            elif self.design[i][0] == 'C(U3)':               
                layer = self.design[i][2]
                qubit_pair = self.design[i][1]
                control_qubit = qubit_pair[0]
                params = self.q_params_enta[layer][control_qubit].unsqueeze(0)
                tqf.cu3(qdev, wires=qubit_pair, params=params)
        
        # ⚠️ 替换有问题的弱测量函数，使用我们自己的实现
        for i in range(self.n_wires):
            tqf.weak_measurement(qdev, wires=i, params=self.g)        
        
        expval = tq.expval_density(qdev, wires=[i for i in range(self.n_wires)], observables=self.obs)  # [batch_size]


        # 将结果转换回2D密度矩阵格式 - 使用clone确保安全
        result_rho = qdev.densities.clone().reshape(batch_size, 2**self.n_wires, 2**self.n_wires)
        
        return result_rho, expval
    
    def x_to_rho(self, x_values):
        rho_list = []
        for x in x_values:
            rho_A = torch.tensor([
                [1.0 - x, np.sqrt((1.0 - x) * x)],
                [np.sqrt((1.0 - x) * x), x]
            ], dtype=torch.complex64)
            rho_list.append(rho_A)

        # 使用 torch.kron 替代 jnp.kron
        rho=rho_list[0]
        for i in range(1,len(rho_list)):
            rho = torch.kron(rho_list[i], rho)
        return rho

    def partial_trace(self, rho, keep_wires):
        dim_keep = 2 ** keep_wires
        dim_trace = 2 ** (self.n_wires - keep_wires)

        # 将密度矩阵重塑为 (dim_trace, dim_keep, dim_trace, dim_keep)
        rho_reshaped = rho.reshape((dim_trace, dim_keep, dim_trace, dim_keep))

        # 使用 einsum 计算部分迹: 对第一个和第三个索引求和
        rho_reduced = torch.einsum('ijik->jk', rho_reshaped)

        return rho_reduced   
        
    def forward(self, x, return_features=False):
    # x.shape: [batch_size, seq_length, feature_dim]
    # 例如: [32, 6, 3] - 32个样本，每个6步时间序列，每步3个特征
    
        batch_size, seq_length, feature_dim = x.shape
        
        # 初始化所有样本的量子态 - 批量初始化
        # 创建初始的6量子比特密度矩阵，所有量子比特初始化为0
        initial_state = [0] * seq_length
        rho = self.x_to_rho(initial_state)  # [2^6, 2^6]
        
        # 扩展到批次维度 - 使用clone确保独立副本
        rho_batch = rho.unsqueeze(0).expand(batch_size, -1, -1).clone()  # [batch_size, 2^6, 2^6]
        
        all_measurements = []

        # seq_length = 1
        
        # 逐时间步处理整个批次
        for t in range(seq_length):
            current_input = x[:, t, :]  # [batch_size, feature_dim]
            
            # 1. 对所有样本进行部分迹操作（压缩历史信息）
            batch_rho_reduced = []
            keep_qubits = self.n_wires - feature_dim
            
            for i in range(batch_size):
                # 使用clone确保不修改原始张量
                rho_reduced = self.partial_trace(rho_batch[i].clone(), keep_qubits)
                batch_rho_reduced.append(rho_reduced)
            
            rho_batch = torch.stack(batch_rho_reduced)  # [batch_size, 2^keep_qubits, 2^keep_qubits]
            
            # 2. 批量编码当前输入并合并
            batch_new_rho = []
            for i in range(batch_size):
                # 编码当前输入为密度矩阵
                input_rho = self.x_to_rho(current_input[i])  # [2^feature_dim, 2^feature_dim]
                
                # 与历史信息合并 - 使用clone确保安全
                combined_rho = torch.kron(rho_batch[i].clone(), input_rho)
                batch_new_rho.append(combined_rho)
            
            rho_batch = torch.stack(batch_new_rho)  # [batch_size, 2^n_wires, 2^n_wires]
            
            # 3. 批量VQC变换
            
            batch_states = []
            for i in range(batch_size):                
                batch_states.append(rho_batch[i].clone())  # 确保独立副本
            
            # 调用VQC进行批量处理
            rho_batch, step_measurements = self.vqc_density_input(torch.stack(batch_states))            
                        
            all_measurements.append(step_measurements)
        
        # 组合所有时间步的测量结果
        all_measurements = torch.stack(all_measurements)  # [seq_length, batch_size, 3]
        all_measurements = all_measurements.transpose(0, 1)  # [batch_size, seq_length, 3]
        
        # 展平为 [batch_size, seq_length * 3]
        features = all_measurements.reshape(batch_size, -1)  # [batch_size, seq_length * 3]
        
        # 通过全连接层
        y = self.fc(features)  # [batch_size, n_class]

        if return_features:
            return features
        else:
            return y
