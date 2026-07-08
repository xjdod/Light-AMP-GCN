import os
import numpy as np
import joblib
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

# --- 配置 ---
# (使用 XUAMP 数据集作为示例，因为它是您的主战场)
TRAIN_POS = r"datasets\train datasets\XUAMP\XU_train_positive.fasta"
TRAIN_NEG = r"datasets\train datasets\XUAMP\XU_train_negative.fasta"
TEST_POS = r"datasets\independent test datasets\XUAMP\XU_AMP.fasta"
TEST_NEG = r"datasets\independent test datasets\XUAMP\XU_nonAMP.fasta"

MODEL_SAVE_PATH = "saved_models/RF_XU_final.pkl"
SCALER_SAVE_PATH = "saved_models/RF_scaler.pkl"

# --- 特征提取函数 (AAC + 理化性质) ---
def get_features(seq):
    # 1. 过滤非法字符
    valid_aa = "ACDEFGHIKLMNPQRSTVWY"
    seq = "".join([c for c in seq.upper() if c in valid_aa])
    if len(seq) == 0: return np.zeros(25) # 20(AAC) + 5(Phys)
    
    analyser = ProteinAnalysis(seq)
    
    # 2. 氨基酸组分 (AAC) - 20维
    aac_dict = analyser.get_amino_acids_percent()
    aac = [aac_dict.get(aa, 0.0) for aa in valid_aa]
    
    # 3. 理化性质 - 5维
    try:
        charge = analyser.charge_at_pH(7.4)
        gravy = analyser.gravy()
        iso = analyser.isoelectric_point()
        aro = analyser.aromaticity()
        # 简单的脂肪族指数估算
        ala = seq.count('A')
        val = seq.count('V')
        ile = seq.count('I')
        leu = seq.count('L')
        ali_ind = 100 * (ala + 2.9 * val + 3.9 * (ile + leu)) / len(seq)
    except:
        charge, gravy, iso, aro, ali_ind = 0, 0, 0, 0, 0
        
    return np.array(aac + [charge, gravy, iso, aro, ali_ind])

def load_and_extract(pos_file, neg_file, desc="Loading"):
    X = []
    y = []
    
    # 加载正样本
    for r in tqdm(SeqIO.parse(pos_file, "fasta"), desc=f"{desc} Positive"):
        feat = get_features(str(r.seq))
        X.append(feat)
        y.append(1)
        
    # 加载负样本
    for r in tqdm(SeqIO.parse(neg_file, "fasta"), desc=f"{desc} Negative"):
        feat = get_features(str(r.seq))
        X.append(feat)
        y.append(0)
        
    return np.array(X), np.array(y)

def main():
    print(">>> 1. 准备训练数据...")
    X_train, y_train = load_and_extract(TRAIN_POS, TRAIN_NEG, "Train")
    
    print("\n>>> 2. 准备测试数据...")
    X_test, y_test = load_and_extract(TEST_POS, TEST_NEG, "Test")
    
    # 数据归一化 (对 ML 很重要)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 保存标尺
    joblib.dump(scaler, SCALER_SAVE_PATH)
    
    print("\n>>> 3. 训练随机森林 (Random Forest)...")
    # n_estimators=500: 500棵树
    # n_jobs=-1: 使用所有 CPU 核心加速
    clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    
    # 保存模型
    joblib.dump(clf, MODEL_SAVE_PATH)
    print(f"模型已保存: {MODEL_SAVE_PATH}")
    
    # 评估
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    
    print("-" * 30)
    print(f"[Random Forest 单独表现]")
    print(f"Test AUC: {auc:.4f}")
    print(f"Test ACC: {acc:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()