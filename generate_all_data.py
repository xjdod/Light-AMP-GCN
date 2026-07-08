import torch
import esm
import time
import os
from Bio import SeqIO
from tqdm import tqdm
import sys
import re
import numpy as np

# --- 1. 清理文件名函数 ---
def sanitize_filename(name):
    # 只保留字母、数字、下划线、连字符
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name) 
    return name

# --- 2. 检查输入参数 ---
if len(sys.argv) != 4:
    print("错误：请提供 3 个参数！")
    print("用法: python generate_all_data.py <输入FASTA> <特征输出目录> <图输出目录>")
    sys.exit(1)

input_fasta_file = sys.argv[1]
output_feature_dir = sys.argv[2]
output_graph_dir = sys.argv[3] 

# --- 3. 加载 ESM-1b 模型 ---
print("正在加载 ESM (650M) 模型...")
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
if torch.cuda.is_available(): model = model.cuda()
model.eval()
batch_converter = alphabet.get_batch_converter()
print("模型已加载到 GPU！")

# --- 4. 准备输入和输出 ---
print(f"即将处理: {input_fasta_file}")
print(f"特征输出到: {output_feature_dir}")
print(f"图输出到:   {output_graph_dir}")
os.makedirs(output_feature_dir, exist_ok=True)
os.makedirs(output_graph_dir, exist_ok=True)

# --- 5. 处理序列 ---
sequences_to_process = list(SeqIO.parse(input_fasta_file, "fasta"))
total_sequences = len(sequences_to_process)
print(f"共找到 {total_sequences} 条序列。")

skipped_count = 0
success_count = 0 
valid_chars = set("ACDEFGHIKLMNPQRSTVWYX") # 允许 X 作为未知氨基酸

for record in tqdm(sequences_to_process, desc="处理序列"):
    seq_id_original = record.id
    
    # (!!! 核心修复 !!!) 强制转大写
    seq_str = str(record.seq).upper()
    
    # (!!! 核心修复 !!!) 过滤非法字符
    # 如果包含非氨基酸字符（比如 B, J, O, U, Z），替换为 X 或跳过
    # 这里我们做一个简单的清洗，去除非法字符
    clean_seq = "".join([c for c in seq_str if c in valid_chars])
    
    # 如果清洗后长度变短太多，或者为空，则跳过
    if len(clean_seq) == 0:
        skipped_count += 1
        continue
        
    seq_str = clean_seq # 使用清洗后的序列

    seq_id_safe = sanitize_filename(seq_id_original) 

    feature_file_path = os.path.join(output_feature_dir, f"{seq_id_safe}.pt")
    graph_file_path = os.path.join(output_graph_dir, f"{seq_id_safe}.pt")

    # 长度检查 (ESM-1b 限制)
    if len(seq_str) > 1022:
        skipped_count += 1
        continue 
        
    data = [(seq_id_original, seq_str)]
    
    try:
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        if torch.cuda.is_available(): batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)

        token_representations = results["representations"][33]
        node_features = token_representations[0, 1:-1] 
        torch.save(node_features.cpu(), feature_file_path)

        contact_map = results["contacts"][0].cpu() 
        torch.save(contact_map, graph_file_path)
        
        success_count += 1 
    
    except Exception as e:
        # 捕获所有错误 (包括 KeyError)，打印出来但不中断程序
        # tqdm.write(f"  > 警告: 序列 {seq_id_safe} 处理失败: {e}")
        skipped_count += 1
        continue

print(f"\n--- {input_fasta_file} 处理完成 ---")
print(f"  > 成功生成 (并保存): {success_count} 条序列")
print(f"  > 跳过: {skipped_count} 条序列")