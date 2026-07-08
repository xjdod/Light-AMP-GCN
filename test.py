import numpy as np
import torch
from torch_geometric.data import DataLoader
import argparse
from models.GAT import GATModel 
from utils.data_processing import load_data 
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
import torch.nn.functional as F
import pandas as pd
import warnings
import sys
warnings.filterwarnings("ignore")

def independent_test(args):
    threshold = args.d 

    # --- 1. 加载数据 ---
    # 使用 load_data 加载正负样本
    # 注意：load_data 现在会自动推断 feature_dir (data_esm)，只要传入正确的 npz_dir (data_graphs)
    
    # 加载正样本
    fasta_path_positive = args.pos_t
    npz_dir_positive = args.pos_npz
    data_list_pos, labels_pos = load_data(fasta_path_positive, npz_dir_positive, threshold, 1)

    # 加载负样本
    fasta_path_negative = args.neg_t
    npz_dir_negative = args.neg_npz
    data_list_neg, labels_neg = load_data(fasta_path_negative, npz_dir_negative, threshold, 0)

    # 合并
    data_list = data_list_pos + data_list_neg
    labels = np.concatenate((labels_pos, labels_neg), axis=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    test_loader = DataLoader(data_list, batch_size=args.b, shuffle=False)
    print(f"Total test samples: {len(data_list)}")

    # --- 2. 加载模型 (核心修复) ---
    print(f"正在加载模型: {args.save}")
    try:
        # 步骤 A: 初始化模型架构 (必须与训练时一致)
        # node_feature_dim=1280 (ESM-1b/ESM-2)
        model = GATModel(
            node_feature_dim=1280, 
            hidden_dim=args.hd, 
            num_classes=2, 
            dropout_rate=args.drop, 
            num_heads=args.heads
        ).to(device)
        
        # 步骤 B: 加载训练好的权重 (state_dict)
        model.load_state_dict(torch.load(args.save))
        print("✅ 模型加载成功！")
        
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        print("提示: 这可能是因为模型路径错误，或者 hidden_dim/heads 参数与训练时不匹配。")
        sys.exit(1)
        
    model.eval()
    
    # --- 3. 运行预测 ---
    y_true_test = []
    y_pred_test = []
    y_score_test = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            
            # 运行模型
            # 注意：V1 模型返回 (out, x)
            output = model(data.x, data.edge_index, data.edge_attr, data.batch)
            
            # 兼容性处理
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            
            score = F.softmax(out, dim=1)[:, 1] 
            pred = out.max(dim=1)[1]
            
            y_true_test.append(data.y.cpu().numpy())
            y_pred_test.append(pred.cpu().numpy())
            y_score_test.append(score.cpu().numpy())

    y_true_test = np.concatenate(y_true_test)
    y_pred_test = np.concatenate(y_pred_test)
    y_score_test = np.concatenate(y_score_test)
    
    # --- 4. 计算指标 ---
    test_auc = roc_auc_score(y_true_test, y_score_test)
    test_acc = accuracy_score(y_true_test, y_pred_test)
    test_f1 = f1_score(y_true_test, y_pred_test)
    test_mcc = matthews_corrcoef(y_true_test, y_pred_test)
    
    tn, fp, fn, tp = confusion_matrix(y_true_test, y_pred_test).ravel()
    test_sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    test_sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("-" * 50)
    print(f"> [Benchmark1 测试] 最终评估结果:")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test ACC: {test_acc:.4f}")
    print(f"Test MCC: {test_mcc:.4f}")
    print(f"Test Sn:  {test_sn:.4f}")
    print(f"Test Sp:  {test_sp:.4f}")
    print("-" * 50)

    # --- 5. 保存结果 ---
    if args.o:
        try:
            df = pd.DataFrame({'AMP_label': y_true_test, 'score': y_score_test, 'pred': y_pred_test})
            df.to_csv(args.o, index=False)
            print(f"结果已保存到: {args.o}")
        except Exception as e:
            print(f"保存 CSV 失败: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 路径参数
    parser.add_argument('-pos_t', type=str, required=True)
    parser.add_argument('-pos_npz', type=str, required=True)
    parser.add_argument('-neg_t', type=str, required=True)
    parser.add_argument('-neg_npz', type=str, required=True)
    parser.add_argument('-save', type=str, required=True)
    parser.add_argument('-o', type=str, default='test_results.csv')
    
    # 模型参数 (默认值需与 train.py 保持一致)
    parser.add_argument('-b', type=int, default=512)
    parser.add_argument('-drop', type=float, default=0.5)
    parser.add_argument('-hd', type=int, default=64)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-d', type=int, default=37)
    
    args = parser.parse_args()
    independent_test(args)