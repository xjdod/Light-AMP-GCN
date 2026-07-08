import numpy as np
import torch
from torch_geometric.data import DataLoader
import argparse
from models.GAT_CL import GATModel_CL # 导入新的对比学习模型
from utils.data_processing import load_data 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import warnings
import os

warnings.filterwarnings("ignore")

# --- 核心：对比学习损失函数 (InfoNCE) ---
def contrastive_loss(z1, z2, temperature=0.5):
    # z1: 序列视图特征 [Batch, Dim]
    # z2: 图视图特征 [Batch, Dim]
    
    # 1. 归一化 (Cosine Similarity 需要)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # 2. 计算相似度矩阵 [Batch, Batch]
    # 每一行代表一个序列视图与所有图视图的相似度
    logits = torch.matmul(z1, z2.T) / temperature
    
    # 3. 标签：对角线是正样本 (自己对比自己)
    batch_size = z1.shape[0]
    labels = torch.arange(batch_size).to(z1.device)
    
    # 4. 计算交叉熵损失
    loss = F.cross_entropy(logits, labels)
    return loss

def train(args):
    threshold = args.d 

    # 1. 加载数据 (使用 XUAMP 训练集)
    fasta_path_positive = args.pos_t
    npz_dir_positive = args.pos_npz
    data_list_pos, labels_pos = load_data(fasta_path_positive, npz_dir_positive, threshold, 1)

    fasta_path_negative = args.neg_t
    npz_dir_negative = args.neg_npz
    data_list_neg, labels_neg = load_data(fasta_path_negative, npz_dir_negative, threshold, 0)
    
    data_list = data_list_pos + data_list_neg
    labels = np.concatenate((labels_pos, labels_neg), axis=0)
    
    data_train, data_val, _, _ = train_test_split(data_list, labels, test_size=0.1, random_state=1)
    
    # 注意：对比学习需要较大的 Batch Size 才有好的负样本效果
    train_loader = DataLoader(data_train, batch_size=args.b, shuffle=True, drop_last=True) 
    val_loader = DataLoader(data_val, batch_size=args.b, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化对比学习模型
    model = GATModel_CL(node_feature_dim=1280, hidden_dim=args.hd, num_classes=2, dropout_rate=args.drop, num_heads=args.heads).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion_cls = torch.nn.CrossEntropyLoss() # 分类损失

    best_val_auc = 0
    
    print('\n--- 开始训练 (Graph-Sequence Contrastive Learning) ---')
    
    for epoch in range(args.e):
        model.train()
        total_loss = 0
        total_cls_loss = 0
        total_cl_loss = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # 获取三个输出
            out, z_seq, z_graph = model(data.x, data.edge_index, data.edge_attr, data.batch)
            
            # 1. 计算分类损失
            loss_cls = criterion_cls(out, data.y)
            
            # 2. 计算对比损失
            loss_cl = contrastive_loss(z_seq, z_graph, temperature=0.1)
            
            # 3. 总损失 = 分类 + Lambda * 对比
            # lambda_cl 是对比损失的权重，通常取 0.1 到 1.0
            loss = loss_cls + (args.lambda_cl * loss_cl)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.num_graphs
            total_cls_loss += loss_cls.item() * data.num_graphs
            total_cl_loss += loss_cl.item() * data.num_graphs
        
        avg_loss = total_loss / len(train_loader.dataset)
        avg_cls = total_cls_loss / len(train_loader.dataset)
        avg_cl = total_cl_loss / len(train_loader.dataset)

        # 验证 (只看分类 AUC)
        model.eval()
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out, _, _ = model(data.x, data.edge_index, data.edge_attr, data.batch)
                total_preds = torch.cat((total_preds, out.cpu()), 0)
                total_labels = torch.cat((total_labels, data.y.cpu()), 0)
        
        val_auc = roc_auc_score(total_labels, F.softmax(total_preds, dim=1)[:, 1])

        print(f"Epoch {epoch+1}/{args.e} | Loss: {avg_loss:.4f} (Cls: {avg_cls:.4f}, CL: {avg_cl:.4f}) | Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            torch.save(model.state_dict(), args.save)
            print(f'  -> (New Best! Saved to {args.save})')
            
    print(f'训练完成。Best Val AUC: {best_val_auc:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 路径参数
    parser.add_argument('-pos_t', type=str, required=True)
    parser.add_argument('-pos_npz', type=str, required=True)
    parser.add_argument('-neg_t', type=str, required=True)
    parser.add_argument('-neg_npz', type=str, required=True)
    parser.add_argument('-save', type=str, default='saved_models/GCN_CL_final.model')

    # 超参数
    parser.add_argument('-lr', type=float, default=0.0005) # 对比学习通常需要小一点的学习率
    parser.add_argument('-drop', type=float, default=0.5)
    parser.add_argument('-e', type=int, default=50)
    parser.add_argument('-b', type=int, default=256) # 较大的 batch size 对对比学习更好
    parser.add_argument('-hd', type=int, default=64)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-d', type=int, default=37)
    
    # [新增] 对比损失权重
    parser.add_argument('-lambda_cl', type=float, default=0.5, help='Weight for contrastive loss')
    
    parser.add_argument('-pretrained_model', type=str, default="")
    parser.add_argument('-o', type=str, default='log.txt')
    
    args = parser.parse_args()
    
    train(args)