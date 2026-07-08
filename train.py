import numpy as np
import torch
# 适配 PyG 1.7.2 的 DataLoader
from torch_geometric.data import DataLoader 
import argparse
from models.GAT import GATModel 
from utils.data_processing import load_data 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import warnings
import os # (修复: 之前缺了这个)

warnings.filterwarnings("ignore")

def train(args):
    threshold = args.d 

    # --- 1. 加载训练数据 ---
    # 正样本
    fasta_path_positive = args.pos_t
    npz_dir_positive = args.pos_npz
    data_list_pos, labels_pos = load_data(fasta_path_positive, npz_dir_positive, threshold, 1)

    # 负样本
    fasta_path_negative = args.neg_t
    npz_dir_negative = args.neg_npz
    data_list_neg, labels_neg = load_data(fasta_path_negative, npz_dir_negative, threshold, 0)
    
    # 合并
    data_list = data_list_pos + data_list_neg
    labels = np.concatenate((labels_pos, labels_neg), axis=0)
    
    print(f"Total samples loaded: {len(data_list)}")
    
    # --- 2. 自动划分验证集 (10%) ---
    # 这样可以避免手动指定验证集路径导致的“找不到文件”错误
    data_train, data_val, _, _ = train_test_split(data_list, labels, test_size=0.1, random_state=1)
        
    print(f'Train size: {len(data_train)}, Val size: {len(data_val)}')
    
    train_loader = DataLoader(data_train, batch_size=args.b, shuffle=True)
    val_loader = DataLoader(data_val, batch_size=args.b, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    
    # --- 3. 初始化模型 ---
    # 自动获取特征维度 (通常是 1280)
    input_dim = data_train[0].x.shape[1]
    model = GATModel(node_feature_dim=input_dim, hidden_dim=args.hd, num_classes=2, dropout_rate=args.drop, num_heads=args.heads).to(device)
    
    if args.pretrained_model != "":
        model.load_state_dict(torch.load(args.pretrained_model))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = torch.nn.CrossEntropyLoss()

    best_val_auc = 0
    best_epoch = 0
    
    print('Start training...')
    
    # --- 4. 训练循环 ---
    for epoch in range(args.e):
        model.train()
        total_loss = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # V1 模型调用
            out, x = model(data.x, data.edge_index, data.edge_attr, data.batch)
            
            loss = loss_func(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        
        train_loss = total_loss / len(train_loader.dataset)

        # --- 5. 验证 ---
        model.eval()
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out, x = model(data.x, data.edge_index, data.edge_attr, data.batch)
                
                total_preds = torch.cat((total_preds, out.cpu()), 0)
                total_labels = torch.cat((total_labels, data.y.cpu()), 0)
        
        val_auc = roc_auc_score(total_labels, F.softmax(total_preds, dim=1)[:, 1])

        print(f"Epoch {epoch+1}/{args.e}, Train Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}")

        # --- 6. 保存最佳模型 ---
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            # 保存权重 state_dict
            torch.save(model.state_dict(), args.save) 
            print(f'  -> (New Best! Saved to {args.save})')
            
    print(f'训练完成。Best Epoch: {best_epoch}, Best Val AUC: {best_val_auc:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 必需参数
    parser.add_argument('-pos_t', type=str, required=True, help='正样本 FASTA')
    parser.add_argument('-pos_npz', type=str, required=True, help='正样本 图数据路径')
    parser.add_argument('-neg_t', type=str, required=True, help='负样本 FASTA')
    parser.add_argument('-neg_npz', type=str, required=True, help='负样本 图数据路径')
    
    # 兼容性参数 (不再实际使用，但保留以防报错)
    parser.add_argument('-pos_v', type=str, default="")
    parser.add_argument('-neg_v', type=str, default="")
    parser.add_argument('-pos_v_npz', type=str, default="")
    parser.add_argument('-neg_v_npz', type=str, default="")

    # 模型保存路径
    parser.add_argument('-save', type=str, default='saved_models/model_final.model')

    # 超参数
    parser.add_argument('-lr', type=float, default=0.001) 
    parser.add_argument('-drop', type=float, default=0.5)
    parser.add_argument('-e', type=int, default=50)
    parser.add_argument('-b', type=int, default=512)
    parser.add_argument('-hd', type=int, default=64)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-d', type=int, default=37)
    parser.add_argument('-pretrained_model', type=str, default="")
    parser.add_argument('-o', type=str, default='log.txt')
    
    args = parser.parse_args()
    
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)

    train(args)