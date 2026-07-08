import torch
import argparse
import os
from torch_geometric.nn import GNNExplainer
from Bio import SeqIO
import re

# 导入您自己的数据加载工具
from utils.data_processing import load_data 

# --- (!!! 核心修复 1: 添加模型包装器 !!!) ---
# 这个类的作用是“包装”您的 GATModel
# GNNExplainer 会调用它，它会调用您的模型，
# 然后只返回您的模型输出的第一个元素（logits），
# 从而解决 "tuple object has no attribute 'argmax'" 错误。

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        # 调用您原始的 GATModel (GCN)
        # 假设 GAT.py 中的 forward 签名是 (self, x, edge_index, edge_attr, batch, **kwargs)
        output_tuple = self.model(x, edge_index, edge_attr, batch, **kwargs)
        
        # (!!!) 只返回元组的第一个元素 (out 预测结果)
        return output_tuple[0]

# ----------------------------------------

# --- 1. 清理文件名函数 (不变) ---
def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name)
# ----------------------------------------

def explain_model(args):
    
    # --- 步骤 1: 加载数据和模型 ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1.1 加载您 *原始* 的模型
    try:
        # 我们称之为 original_model
        original_model = torch.load(args.model_path).to(device)
        original_model.eval() # 必须设置为评估模式
        print(f"成功加载原始模型: {args.model_path}")
    except FileNotFoundError:
        print(f"\n!! 错误: 找不到模型文件: {args.model_path} !!")
        return

    # 1.2 加载图数据 (GCN的输入)
    print("正在加载 XUAMP 正样本测试集 (图数据)...")
    data_list_pos, _ = load_data(args.fasta_path, args.npz_dir, 37, 1)
    print(f"共加载 {len(data_list_pos)} 个正样本图。")

    # 1.3 从 FASTA 文件加载原始序列 (用于显示)
    print("正在加载 FASTA 序列...")
    try:
        fasta_records = list(SeqIO.parse(args.fasta_path, "fasta"))
    except FileNotFoundError:
        print(f"\n!! 错误: 找不到 FASTA 文件: {args.fasta_path} !!")
        return
    print(f"共加载 {len(fasta_records)} 条 FASTA 序列。")
    

    # --- 步骤 2: 选择一个样本进行解释 ---
    sample_idx = args.sample_idx
    if sample_idx >= len(data_list_pos) or sample_idx >= len(fasta_records):
        print(f"!! 错误: 索引 {sample_idx} 超出范围。")
        return
        
    data = data_list_pos[sample_idx].to(device)
    record = fasta_records[sample_idx]
    
    seq_id = record.id
    original_seq = str(record.seq)

    # (修复 'batch' 属性问题 - 这一步仍然需要)
    if not hasattr(data, 'batch'):
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        print(f"已为样本 {sample_idx} 手动创建 .batch 属性。")
    
    # 检查节点数是否与序列长度一致
    if data.num_nodes != len(original_seq):
         print(f"\n!! 警告: 样本 {seq_id} 节点数 ({data.num_nodes}) 与序列长度 ({len(original_seq)}) 不匹配!")

    print(f"\n--- 正在解释样本: {seq_id} (索引 {sample_idx}) ---")
    print(f"序列: {original_seq}")
    print(f"长度: {len(original_seq)} 个氨基酸 (节点数: {data.num_nodes})")
    
    # 验证模型对该样本的原始预测 (使用 original_model)
    with torch.no_grad():
        # 您的模型返回一个元组 (out, x)
        log_logits, _ = original_model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred_class = log_logits.argmax(dim=1).item()
        print(f"模型原始预测: {'AMP' if pred_class == 1 else 'Non-AMP'} (Logits: {log_logits.cpu().numpy()})")

    
    # --- (!!! 核心修复 2: 使用包装器 !!!) ---
    # 创建包装后的模型，专门用于 GNNExplainer
    explainer_model = ModelWrapper(original_model).to(device)

    # GNNExplainer 会自己创建 'batch' 参数。
    # 我们创建一个字典，只包含它不知道的 *额外* 参数 (edge_attr)
    custom_model_args = {
        "edge_attr": data.edge_attr
    }

    # --- 步骤 3: 初始化 GNNExplainer ---
    # (!!! 核心修复 3 !!!)
    # 将 explainer_model (包装器) 传给 GNNExplainer
    explainer = GNNExplainer(
        model=explainer_model, # <--- 使用包装后的模型
        epochs=200,      
        lr=0.01,         
        log=False
    )

    # --- 步骤 4: 运行解释器 ---
    print("\nGNNExplainer 正在运行 (可能需要 1-2 分钟)...")
    
    # GNNExplainer 现在会调用 ModelWrapper
    # ModelWrapper 会调用 GATModel，并只返回第一个元素
    # 错误 'AttributeError: 'tuple' object...' 将被解决
    node_feat_mask, edge_mask = explainer.explain_graph(
        x=data.x,
        edge_index=data.edge_index,
        **custom_model_args  
    )
    print("解释完成。")

    # --- 步骤 5: 分析和可视化结果 (不变) ---
    
    node_importance = torch.zeros(data.num_nodes, device=device)
    node_importance.index_add_(0, data.edge_index[0], edge_mask)
    node_importance.index_add_(0, data.edge_index[1], edge_mask)
    
    # 归一化
    if node_importance.max() > 0:
         node_importance = node_importance / node_importance.max()
    
    print("\n--- 解释结果: 节点（氨基酸）重要性 ---")
    
    k_val = min(10, data.num_nodes)
    top_k_val, top_k_idx = torch.topk(node_importance, k_val)

    # 构建带高亮的序列字符串
    highlighted_seq = ""
    for i, aa in enumerate(original_seq):
        if i >= len(node_importance): break 
        importance = node_importance[i].item()
        if importance > 0.8: # 高重要性
            highlighted_seq += f"[\033[91m{aa}\033[0m]" # 红色
        elif importance > 0.5: # 中重要性
            highlighted_seq += f"[\033[93m{aa}\033[0m]" # 黄色
        else:
            highlighted_seq += aa # 正常
            
    print(f"高亮序列 (红色 > 80%, 黄色 > 50%):")
    print(highlighted_seq)
    
    print(f"\n重要性排名前 {k_val} 的氨基酸 (索引从 0 开始):")
    for i in range(len(top_k_idx)):
        idx = top_k_idx[i].item()
        val = top_k_val[i].item()
        if idx < len(original_seq):
            print(f"  - 索引: {idx:<3} (氨基酸: {original_seq[idx]}) - 重要性: {val:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GNNExplainer for Light-AMP-GCN")
    
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
    
    # 4. 您想解释的样本索引
    parser.add_argument('-sample_idx', type=int, default=181, 
                        help='Index of the sample in the fasta file to explain')
    
    args = parser.parse_args()
    explain_model(args)