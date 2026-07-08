import torch
import argparse
import os
from torch_geometric.nn import GNNExplainer
from Bio import SeqIO
import re
import pandas as pd
from tqdm import tqdm # 导入tqdm用于显示进度条
import sys

# 导入您自己的数据加载工具
try:
    from utils.data_processing import load_data 
except ImportError:
    print("错误: 找不到 'utils.data_processing'。请确保 'utils' 文件夹在同一目录下。")
    sys.exit(1)

# --- (核心修复: 添加模型包装器) ---
# 解决 "tuple object has no attribute 'argmax'" 错误
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        # 调用 GATModel (GCN)
        # 假设 GAT.py 中的 forward 签名是 (self, x, edge_index, edge_attr, batch, **kwargs)
        output_tuple = self.model(x, edge_index, edge_attr, batch, **kwargs)
        # 只返回元组的第一个元素 (out 预测结果)
        return output_tuple[0]

# --- 1. 清理文件名函数 (不变) ---
def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name)
# ----------------------------------------

def explain_aggregate_model(args):
    
    # --- 步骤 1: 加载数据和模型 ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1.1 加载模型
    try:
        original_model = torch.load(args.model_path).to(device)
        original_model.eval() 
        print(f"成功加载原始模型: {args.model_path}")
    except FileNotFoundError:
        print(f"\n!! 错误: 找不到模型文件: {args.model_path} !!")
        return

    # 1.2 加载 *所有* 正样本图数据
    print("正在加载 XUAMP 正样本测试集 (图数据)...")
    data_list_pos, _ = load_data(args.fasta_path, args.npz_dir, 37, 1)
    print(f"共加载 {len(data_list_pos)} 个正样本图。")

    # 1.3 加载 *所有* 正样本 FASTA 序列
    print("正在加载 FASTA 序列...")
    try:
        fasta_records = list(SeqIO.parse(args.fasta_path, "fasta"))
    except FileNotFoundError:
        print(f"\n!! 错误: 找不到 FASTA 文件: {args.fasta_path} !!")
        return
    print(f"共加载 {len(fasta_records)} 条 FASTA 序列。")
    
    # --- 步骤 2: 筛选要分析的目标样本 ---
    print(f"正在加载结果 CSV: {args.csv_path}")
    try:
        df = pd.read_csv(args.csv_path)
    except FileNotFoundError:
        print(f"\n!! 错误: 找不到 CSV 文件: {args.csv_path} !!")
        return
        
    # 筛选 True Positives (只在正样本中 (前1536行) 筛选)
    df_pos = df.iloc[:len(data_list_pos)]
    df_tp = df_pos[(df_pos['AMP_label'] == 1) & (df_pos['pred'] == 1)]
    
    # 按置信度得分降序排列
    df_tp = df_tp.sort_values(by='score', ascending=False)
    
    # 获取要分析的样本索引
    # 截取前 N 个样本
    target_indices = df_tp.index.tolist()
    if args.num_samples > 0:
        target_indices = target_indices[:args.num_samples]
        
    print(f"\n--- 将分析 {len(target_indices)} 个“真阳性” (True Positive) 样本 ---")
    print(f"（已按置信度从高到低排序，取前 {args.num_samples} 个）")

    # --- 步骤 3: 初始化 GNNExplainer 和累加器 ---
    
    # 使用包装器
    explainer_model = ModelWrapper(original_model).to(device)

    explainer = GNNExplainer(
        model=explainer_model,
        epochs=200,      
        lr=0.01,         
        log=False
    )
    
    # 20 种标准氨基酸的累加器
    aa_importance = {aa: 0.0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    aa_counts = {aa: 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}

    print("\nGNNExplainer 正在运行聚合分析...")
    # --- 步骤 4: 循环遍历所有目标样本 ---
    
    for sample_idx in tqdm(target_indices, desc="解释样本"):
        
        # 4.1. 获取数据
        data = data_list_pos[sample_idx].to(device)
        record = fasta_records[sample_idx]
        original_seq = str(record.seq)

        # 4.2. 修复 batch 属性
        if not hasattr(data, 'batch'):
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        
        # 4.3. 检查序列-图是否匹配
        if data.num_nodes != len(original_seq):
            tqdm.write(f"!! 警告: 样本 {record.id} (索引 {sample_idx}) 节点数 ({data.num_nodes}) 与序列长度 ({len(original_seq)}) 不匹配，已跳过。")
            continue
            
        # 4.4. 设置 GNNExplainer 参数
        custom_model_args = {
            "edge_attr": data.edge_attr
        }

        # 4.5. 运行解释器
        try:
            _, edge_mask = explainer.explain_graph(
                x=data.x,
                edge_index=data.edge_index,
                **custom_model_args  
            )
        except Exception as e:
            tqdm.write(f"!! 错误: 样本 {record.id} (索引 {sample_idx}) 解释失败: {e}。已跳过。")
            continue

        # 4.6. 计算节点重要性 (!!! 不进行归一化 !!!)
        # 我们希望置信度高的样本贡献更多权重，所以使用原始分数
        node_importance = torch.zeros(data.num_nodes, device=device)
        node_importance.index_add_(0, data.edge_index[0], edge_mask)
        node_importance.index_add_(0, data.edge_index[1], edge_mask)
        
        # 4.7. 累加到全局字典
        for i, aa in enumerate(original_seq):
            if aa in aa_importance:
                aa_importance[aa] += node_importance[i].item()
                aa_counts[aa] += 1

    print("聚合分析完成。")

    # --- 步骤 5: 分析和可视化结果 ---
    print("\n--- 聚合解释结果: 全局氨基酸重要性 ---")
    
    # 计算平均重要性
    avg_importance = {}
    for aa in aa_importance:
        if aa_counts[aa] > 0:
            avg_importance[aa] = aa_importance[aa] / aa_counts[aa]
        else:
            avg_importance[aa] = 0.0

    # 按平均重要性排序
    sorted_avg_importance = sorted(avg_importance.items(), key=lambda item: item[1], reverse=True)

    print("（根据 GNNExplainer 在所有被分析样本中的平均得分排名）\n")
    print(f"{'排名':<5} | {'氨基酸 (AA)':<10} | {'平均重要性得分':<20}")
    print("-" * 40)
    for i, (aa, score) in enumerate(sorted_avg_importance):
        print(f"#{i+1:<4} | {aa:<10} | {score:<20.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aggregated GNNExplainer for Light-AMP-GCN")
    
    # 1. 模型路径
    parser.add_argument('-model_path', type=str, 
                        default='saved_models/auc_GCN_XU_final.model', 
                        help='Path to the trained GCN model file')
    
    # 2. XUAMP 正样本 FASTA
    parser.add_argument('-fasta_path', type=str, 
                        default='datasets/independent test datasets/XUAMP/XU_AMP.fasta', 
                        help='Path of the positive test dataset')
    
    # 3. 对应的图和特征数据
    parser.add_argument('-npz_dir', type=str, 
                        default='data_graphs/test_data/positive', 
                        help='Path of the npz folder (saving probability graphs)')
    
    # 4. CSV 结果文件
    parser.add_argument('-csv_path', type=str, 
                        default='test_results.csv', 
                        help='Path to the test_results.csv file')
                        
    # 5. (!!!) 要分析的样本数量
    parser.add_argument('--num_samples', type=int, default=50, 
                        help='要分析的 Top True Positive 样本数量 (设为 0 则分析全部，但可能*非常*慢)')

    args = parser.parse_args()
    
    # (确保 GAT.py 中的 forward 签名包含 **kwargs)
    print("!! 重要提示: !!")
    print("请确保您的 'GAT.py' (GCN模型) 文件中的 forward 函数签名包含 '**kwargs'。")
    print("例如: def forward(self, x, edge_index, edge_attr, batch, **kwargs):")
    print("-" * 50)
    
    explain_aggregate_model(args)