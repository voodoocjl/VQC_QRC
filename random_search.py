import pickle
import os
import random
import json
import csv
import numpy as np
import torch
import time
from schemes_qrc_ols import Scheme_Quantum_Escape, Scheme_eval, Scheme_enhanced
from FusionModel import single_enta_to_design

n_qubits = 6
n_layers = 2

path = ['search_space/search_space_qrc_6_single', 'search_space/search_space_qrc_6_enta']

checkpoint = 'init_weights/init_weight_qrc'

epochs = 30
periods = 6
split = 3

# 断点续传文件
checkpoint_file = 'random_search_checkpoint.pkl'

def save_checkpoint(epoch, period, single, enta, weights, search_space_single, search_space_enta, global_id_counter):
    """保存搜索状态的检查点"""
    checkpoint_data = {
        'epoch': epoch,
        'period': period,
        'single': single,
        'enta': enta,
        'weights': weights,
        'search_space_single': search_space_single,
        'search_space_enta': search_space_enta,
        'global_id_counter': global_id_counter,
        'timestamp': time.time(),
        'n_qubits': n_qubits,
        'n_layers': n_layers,
        'epochs': epochs,
        'periods': periods,
        'split': split
    }
    
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"检查点已保存: Epoch {epoch+1}, Period {period+1}")

def load_checkpoint():
    """加载搜索状态的检查点"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            print(f"发现检查点文件，恢复状态...")
            print(f"  上次保存时间: {time.ctime(checkpoint_data['timestamp'])}")
            print(f"  上次进度: Epoch {checkpoint_data['epoch']+1}, Period {checkpoint_data['period']+1}")
            
            # 验证参数一致性
            if (checkpoint_data['n_qubits'] != n_qubits or 
                checkpoint_data['n_layers'] != n_layers or
                checkpoint_data['epochs'] != epochs or
                checkpoint_data['periods'] != periods or
                checkpoint_data['split'] != split):
                print("警告: 检查点参数与当前设置不一致，将重新开始")
                return None
            
            return checkpoint_data
            
        except Exception as e:
            print(f"加载检查点失败: {e}")
            print("将重新开始搜索")
            return None
    else:
        print("未发现检查点文件，从头开始搜索")
        return None

def initialize_or_restore():
    """初始化或恢复搜索状态"""
    checkpoint_data = load_checkpoint()
    
    if checkpoint_data is not None:
        # 从检查点恢复
        start_epoch = checkpoint_data['epoch']
        start_period = checkpoint_data['period'] + 1  # 从下一个period开始
        single = checkpoint_data['single']
        enta = checkpoint_data['enta']
        weights = checkpoint_data['weights']
        search_space_single = checkpoint_data['search_space_single']
        search_space_enta = checkpoint_data['search_space_enta']
        global_id_counter = checkpoint_data['global_id_counter']
        
        # 如果period超出范围，进入下一个epoch
        if start_period >= periods:
            start_epoch += 1
            start_period = 0
        
        print(f"从 Epoch {start_epoch+1}, Period {start_period+1} 继续搜索")
        
        # 检查CSV文件是否需要创建表头
        if not os.path.exists(results_file):
            with open(results_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['num_id', 'job', 'epoch', 'period', 'component_type', 'train_capacity', 'test_capacity'])
        
        if not os.path.exists(complete_results_file):
            with open(complete_results_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['network_structure', 'train_capacity', 'test_capacity'])
                
    else:
        # 全新开始
        start_epoch = 0
        start_period = 0
        single = [[i]+[1]*1*n_layers for i in range(1,n_qubits+1)]
        enta = [[i]+[i+1]*n_layers for i in range(1,n_qubits)]+[[n_qubits]+[1]*n_layers]
        weights = torch.load(checkpoint)
        global_id_counter = 0
        
        # 加载初始搜索空间
        with open(path[0], 'rb') as file:
            search_space_single = pickle.load(file)        
        with open(path[1], 'rb') as file:
            search_space_enta = pickle.load(file)
        
        # 初始化CSV文件
        with open(results_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['num_id', 'job', 'epoch', 'period', 'component_type', 'train_capacity', 'test_capacity'])
        
        with open(complete_results_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['network_structure', 'train_capacity', 'test_capacity'])
        
        print("开始新的搜索")
    
    return start_epoch, start_period, single, enta, weights, search_space_single, search_space_enta, global_id_counter

# 在文件开头定义文件名
results_file = 'results_random_search.csv'
complete_results_file = 'complete_network_results.csv'

# 其他函数保持不变...
def remove_matching_items(search_list, target_item):
    """删除列表中与target_item内容相同的所有元素"""
    filtered_list = [item for item in search_list if item not in target_item]
    return filtered_list

def insert_job(change_code, job_input):
    """参考MCTS中的insert_job方法"""
    import copy
    job = copy.deepcopy(job_input)
    if type(job[0]) == type([]):
        qubit = [sub[0] for sub in job]
    else:
        qubit = [job[0]]
        job = [job]
    if change_code != None:            
        for change in change_code:
            if change[0] not in qubit:
                job.append(change)
    return job

def create_designs_from_search_space(search_space, base_components, component_type):
    """通用函数：从搜索空间创建设计"""
    jobs = []
    designs = []
    
    for candidate_item in search_space:
        if type(candidate_item[0]) != type([]):
            job = [candidate_item]
        else:
            job = candidate_item
            
        if component_type == 'single':
            integrated_single = insert_job(base_components['single'], job)
            integrated_enta = base_components['enta']
        elif component_type == 'enta':
            integrated_single = base_components['single']
            integrated_enta = insert_job(base_components['enta'], job)
        else:
            raise ValueError("component_type must be 'single' or 'enta'")
        
        design = single_enta_to_design(integrated_single, integrated_enta, (n_qubits, n_layers))        
        jobs.append(job)
        designs.append(design)        
    
    return jobs, designs

def evaluate_designs(designs, jobs, weights, max_evaluations=None):
    """评估生成的设计"""
    results = []
    
    if max_evaluations is None:
        max_evaluations = len(designs)
    else:
        max_evaluations = min(max_evaluations, len(designs))
    
    print(f"开始评估 {max_evaluations} 个设计...")
    
    for i in range(max_evaluations):
        if designs[i] is None:
            print(f"  设计 {i+1}: 无效设计，跳过")
            results.append(None)
            continue
            
        print(f"  评估设计 {i+1}/{max_evaluations}")        
        print(f"    任务: {jobs[i]}")        
    
        result = Scheme_eval(designs[i], weight=weights)
        result['job'] = jobs[i]
        result['design_index'] = i
        results.append(result)
    return results

def save_results(results, epoch, period, component_type):
    """保存评估结果到统一的CSV文件"""
    global global_id_counter
    
    valid_results = [r for r in results if r is not None]
    
    if len(valid_results) == 0:
        print("没有有效结果可保存")
        return
    
    with open(results_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        for result in valid_results:
            global_id_counter += 1
            job_str = str(result['job'])
            train_capacity = result.get('train_capacity', 0.0)
            test_capacity = result.get('capacity', 0.0)
            
            writer.writerow([
                global_id_counter, job_str, epoch + 1, period + 1,
                component_type, train_capacity, test_capacity
            ])
    
    print(f"已追加 {len(valid_results)} 条记录到 {results_file}")

def save_complete_network_result(single, enta, result):
    """保存完整网络结构和结果到CSV文件"""
    network_structure = [single, enta]
    network_str = str(network_structure)
    train_capacity = result.get('train_capacity', 0.0)
    test_capacity = result.get('capacity', 0.0)
    
    with open(complete_results_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([network_str, train_capacity, test_capacity])
    
    print(f"已保存完整网络结构到 {complete_results_file}")
    print(f"Train Capacity: {train_capacity:.6f}")
    print(f"Test Capacity: {test_capacity:.6f}")

# 主函数
def main():
    global global_id_counter
    
    try:
        # 初始化或恢复状态
        start_epoch, start_period, single, enta, weights, search_space_single, search_space_enta, global_id_counter = initialize_or_restore()
        
        # 主搜索循环
        for i in range(start_epoch, epochs):
            # 如果是恢复的epoch，从指定period开始；否则从0开始
            period_start = start_period if i == start_epoch else 0

            # 加载初始搜索空间
            with open(path[0], 'rb') as file:
                search_space_single = pickle.load(file)        
            with open(path[1], 'rb') as file:
                search_space_enta = pickle.load(file)
            
            
            for j in range(period_start, periods):
                print(f"\n{'='*60}")
                print(f"Epoch {i+1}/{epochs}, Period {j+1}/{periods}")
                print(f"{'='*60}")
                
                base_components = {'single': single, 'enta': enta}

                if j < split:
                    # 处理single搜索空间
                    print(f"\n--- 处理Single搜索空间 ---")
                    search_space_single = remove_matching_items(search_space_single, single)
                    print(f"可用的single候选数量: {len(search_space_single)}")
                
                    if len(search_space_single) > 0:
                        single_space = random.sample(search_space_single, min(10, len(search_space_single)))
                        jobs_single, designs_single = create_designs_from_search_space(
                            single_space, base_components, 'single'
                        )
                        
                        print(f"生成了 {len(designs_single)} 个single设计")
                        
                        results = evaluate_designs(designs_single, jobs_single, weights, max_evaluations=10)
                        save_results(results, i, j, 'single')

                        valid_results = [r for r in results if r is not None]
                        if len(valid_results) > 0:                
                            max_capacity = max(r.get('capacity', 0.0) for r in valid_results)                
                            best_results = [r for r in valid_results if r.get('capacity', 0.0) == max_capacity]
                            
                            best_result = random.choice(best_results)
                            best_job = best_result['job']
                            
                            print(f"最佳test_capacity: {max_capacity:.6f}")
                            print(f"找到 {len(best_results)} 个最佳结果，随机选择: {best_job}")
                        else:
                            best_job = None
                            print("没有有效的single结果")

                        if best_job is not None:
                            single = insert_job(single, best_job)
                            qubit_used = best_job[0][0]
                            print(f"使用的qubit: {qubit_used}")            
                            search_space_single = [item for item in search_space_single if item[0] != qubit_used]

                        if j == split - 1:
                            design = single_enta_to_design(single, enta, (n_qubits, n_layers))
                            result = Scheme_eval(design, weights)

                            ols_regressor = result.get('regressor', None)
                            if ols_regressor is not None:
                                fc_weight = torch.tensor(ols_regressor.coef_, dtype=torch.float32, requires_grad=True).unsqueeze(0)
                                fc_bias = torch.tensor([ols_regressor.intercept_], dtype=torch.float32, requires_grad=True)
                                weights['fc.weight'] = fc_weight
                                weights['fc.bias'] = fc_bias

                                print("开始量子逃逸训练...")
                                # best_model, report = Scheme_Quantum_Escape((n_qubits, n_layers), design, weight=weights, epochs=30)
                                best_model, report = Scheme_enhanced((n_qubits, n_layers), design, weight=weights, epochs=30)
                                weights = best_model.state_dict()

                                result = Scheme_eval(design, weights, draw=True)
                                save_complete_network_result(single, enta, result)

                else:
                    # 处理enta搜索空间
                    print(f"\n--- 处理Enta搜索空间 ---")
                    search_space_enta = remove_matching_items(search_space_enta, enta)
                    print(f"可用的enta候选数量: {len(search_space_enta)}")
                    
                    if len(search_space_enta) > 0:
                        enta_space = random.sample(search_space_enta, min(10, len(search_space_enta)))
                        jobs_enta, designs_enta = create_designs_from_search_space(
                            enta_space, base_components, 'enta'
                        )
                        
                        print(f"生成了 {len(designs_enta)} 个enta设计")
                        
                        results = evaluate_designs(designs_enta, jobs_enta, weights, max_evaluations=10)
                        save_results(results, i, j, 'enta')

                        valid_results = [r for r in results if r is not None]
                        if len(valid_results) > 0:                
                            max_capacity = max(r.get('capacity', 0.0) for r in valid_results)                
                            best_results = [r for r in valid_results if r.get('capacity', 0.0) == max_capacity]
                            
                            best_result = random.choice(best_results)
                            best_job = best_result['job']
                            
                            print(f"最佳test_capacity: {max_capacity:.6f}")
                            print(f"找到 {len(best_results)} 个最佳结果，随机选择: {best_job}")
                            
                            enta = insert_job(enta, best_job)
                            print(f"更新后的enta: {enta}")
                            
                            qubit_used = best_job[0][0]
                            print(f"使用的qubit: {qubit_used}")
                            search_space_enta = [item for item in search_space_enta if item[0] != qubit_used]                
                            
                        else:
                            best_job = None
                            print("没有有效的enta结果")

                        if j == periods - 1:
                            design = single_enta_to_design(single, enta, (n_qubits, n_layers))                                         
                            result = Scheme_eval(design, weights)
                            ols_regressor = result.get('regressor', None)
                            if ols_regressor is not None:
                                fc_weight = torch.tensor(ols_regressor.coef_, dtype=torch.float32, requires_grad=True).unsqueeze(0)
                                fc_bias = torch.tensor([ols_regressor.intercept_], dtype=torch.float32, requires_grad=True)
                                weights['fc.weight'] = fc_weight
                                weights['fc.bias'] = fc_bias

                                print("开始量子逃逸训练...")
                                # best_model, report = Scheme_Quantum_Escape((n_qubits, n_layers), design, weight=weights, epochs=30)
                                best_model, report = Scheme_enhanced((n_qubits, n_layers), design, weight=weights, epochs=30)
                                weights = best_model.state_dict()

                                result = Scheme_eval(design, weights, draw=True)                     
                                save_complete_network_result(single, enta, result)

                # 每个period结束后保存检查点
                save_checkpoint(i, j, single, enta, weights, search_space_single, search_space_enta, global_id_counter)

        print(f"\n🎉 随机搜索完成!")
        
        # 搜索完成后删除检查点文件
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print("搜索完成，已清理检查点文件")
            
    except KeyboardInterrupt:
        print(f"\n\n⚠️  搜索被用户中断")
        print(f"检查点已保存，可以通过重新运行程序继续搜索")
        
    except Exception as e:
        print(f"\n\n❌ 搜索过程中出现错误: {e}")
        print(f"检查点已保存，可以通过重新运行程序继续搜索")
        raise

if __name__ == "__main__":
    main()
