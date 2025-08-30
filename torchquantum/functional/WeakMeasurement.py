import functools
import torch
import numpy as np

from typing import Callable, Union, Optional, List, Dict, TYPE_CHECKING
from ..macro import C_DTYPE, F_DTYPE, ABC, ABC_ARRAY, INV_SQRT2
from ..util.utils import pauli_eigs, diag
from torchpack.utils.logging import logger
from torchquantum.util import normalize_statevector

from .gate_wrapper import gate_wrapper, apply_unitary_einsum, apply_unitary_bmm

if TYPE_CHECKING:
    from torchquantum.device import QuantumDevice
else:
    QuantumDevice = None


def weak_measurement_matrix(params):
    """Compute matrix for weak measurement gate.

    Args:
        params (torch.Tensor): The measurement strength parameter g.

    Returns:
        torch.Tensor: The computed matrix.
    """
    g = params.type(C_DTYPE)
    
    # WeakMeasurement 矩阵: [[1, exp(-g²/2)], [exp(-g²/2), 1]]
    exp_term = torch.exp(-g * g / 2)
    
    return torch.stack(
        [torch.cat([torch.ones_like(g), exp_term], dim=-1), 
         torch.cat([exp_term, torch.ones_like(g)], dim=-1)], dim=-2
    ).squeeze(0)


_weak_measurement_mat_dict = {
    "weak_measurement": weak_measurement_matrix,
}


def weak_measurement(
    q_device,
    wires,
    params=None,
    n_wires=None,
    static=False,
    parent_graph=None,
    inverse=False,
    comp_method="bmm",
):
    """Perform the weak measurement operation.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = "weak_measurement"
    mat = weak_measurement_matrix
    
    # 使用标准的 gate_wrapper，就像其他单量子比特门一样
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse,
    )
    
    # 非酉操作后的归一化：除以 trace
    if hasattr(q_device, 'densities'):
        normalize_density_after_measurement(q_device)


def normalize_density_after_measurement(q_device):
    """归一化密度矩阵，除以其迹 - 批量处理避免就地操作"""
    batch_size = q_device.densities.shape[0]
    n_qubits = (q_device.densities.dim() - 1) // 2
    total_dim = 2 ** n_qubits
    
    # 批量重塑为 2D 矩阵：[batch_size, total_dim, total_dim]
    rho_2d = q_device.densities.reshape(batch_size, total_dim, total_dim)
    
    # 批量计算迹：[batch_size]
    traces = torch.diagonal(rho_2d, dim1=-2, dim2=-1).sum(dim=-1).real
    
    # 避免除零
    traces = torch.where(torch.abs(traces) > 1e-10, traces, torch.ones_like(traces))
    
    # 批量归一化 - 创建新张量
    normalized_rho_2d = rho_2d / traces.unsqueeze(-1).unsqueeze(-1)
    
    # 重塑回原始形状并替换（不是就地操作）
    q_device.densities = normalized_rho_2d.reshape(q_device.densities.shape)
