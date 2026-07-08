import torch
import torch.nn.functional as F
import numpy as np
import joblib
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score
from models.GAT import GATModel
from utils.data_processing import load_data
from train_ml_baseline import get_features # 复用特征提取函数
from Bio import SeqIO
import warnings
import sys

warnings.filterwarnings("ignore")

# --- 配置 ---
TEST_POS = r"datasets\independent test datasets\XUAMP\XU_AMP.fasta"
TEST_NEG = r"datasets\independent test datasets\XUAMP\XU_nonAMP.fasta"

# 模型路径
GCN_MODEL_PATH = "saved_models/auc_GCN_XU_final.model" # 您的 V1 SOTA 模型
RF_MODEL_PATH = "saved_models/RF_XU_final.pkl"       # 刚才训练的 RF 模型
SCALER_PATH = "saved_models/RF_scaler.pkl"

# 融合权重
ALPHA = 1

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # -----------------------------------------
    # 1. 获取 GCN 的预测结果
    # -----------------------------------------
    print("\n>>> 1. 运行 GCN 模型预测...")
    
    # 加载数据 (V1 路径逻辑)
    # 注意：确保这里的路径参数与您 V1 训练时一致
    pos_data, _ = load_data(TEST_POS, "data_graphs/test_data/positive/", 37, 1)
    neg_data, _ = load_data(TEST_NEG, "data_graphs/test_data/negative/", 37, 0)
    gcn_dataset = pos_data + neg_data
    gcn_loader = DataLoader(gcn_dataset, batch_size=256, shuffle=False)
    
    # (!!! 核心修复: 直接加载整个模型对象 !!!)
    try:
        gcn_model = torch.load(GCN_MODEL_PATH)
        gcn_model = gcn_model.to(device)
        gcn_model.eval()
        print(f"成功加载全模型: {GCN_MODEL_PATH}")
    except Exception as e:
        print(f"尝试全模型加载失败: {e}")
        print("尝试加载 state_dict...")
        # 备选方案：万一它是 state_dict
        gcn_model = GATModel(1280, 64, 2, 0.5, 8).to(device)
        gcn_model.load_state_dict(torch.load(GCN_MODEL_PATH))
        gcn_model.eval()

    gcn_probs = []
    y_true = []
    
    with torch.no_grad():
        for data in gcn_loader:
            data = data.to(device)
            # V1 模型返回 (out, x)
            output = gcn_model(data.x, data.edge_index, data.edge_attr, data.batch)
            
            # 兼容处理
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
                
            prob = F.softmax(out, dim=1)[:, 1]
            gcn_probs.extend(prob.cpu().numpy())
            y_true.extend(data.y.cpu().numpy())
            
    gcn_auc = roc_auc_score(y_true, gcn_probs)
    print(f"   GCN AUC: {gcn_auc:.4f}")

    # -----------------------------------------
    # 2. 获取 Random Forest 的预测结果
    # -----------------------------------------
    print("\n>>> 2. 运行 Random Forest 模型预测...")
    rf_model = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # 提取特征
    seqs = []
    for r in SeqIO.parse(TEST_POS, "fasta"): seqs.append(str(r.seq))
    for r in SeqIO.parse(TEST_NEG, "fasta"): seqs.append(str(r.seq))
    
    # 确保样本数量一致
    if len(seqs) != len(y_true):
        print(f"警告: GCN样本数({len(y_true)}) 与 RF样本数({len(seqs)}) 不一致！")
        print("请检查数据加载逻辑。")
        sys.exit(1)
    
    X_ml = [get_features(s) for s in seqs]
    X_ml = scaler.transform(np.array(X_ml))
    
    rf_probs = rf_model.predict_proba(X_ml)[:, 1]
    
    rf_auc = roc_auc_score(y_true, rf_probs)
    print(f"   RF AUC:  {rf_auc:.4f}")

    # -----------------------------------------
    # 3. 集成融合 (Ensemble)
    # -----------------------------------------
    print(f"\n>>> 3. 计算集成结果 (Alpha = {ALPHA})...")
    
    # 加权平均
    final_probs = np.array(gcn_probs) * ALPHA + np.array(rf_probs) * (1 - ALPHA)
    
    final_auc = roc_auc_score(y_true, final_probs)
    
    print("=" * 40)
    print(f"最终集成 AUC: {final_auc:.4f}")
    print("=" * 40)
    
    if final_auc > gcn_auc:
        print(f"🎉 成功！集成模型提升了性能: +{final_auc - gcn_auc:.4f}")
    else:
        print("未能提升。尝试调整 ALPHA 权重。")

if __name__ == "__main__":
    main()