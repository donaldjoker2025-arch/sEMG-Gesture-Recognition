import numpy as np
from data_loader import load_subject_data, extract_gesture_data
from signal_processor import clean_emg_signal
from feature_extractor import extract_features_sliding_window
from model_trainer import GestureClassifier


def create_training_dataset(subject_filename='female_1.mat'):
    """
    组装一个用于训练的小型数据集（包含圆柱体抓握和球形抓握）。
    """
    print(f"正在加载 {subject_filename} 的数据...")
    mat_data = load_subject_data(subject_filename)
    if not mat_data:
        return None, None

    X_features = []
    y_labels = []

    # 动作1: 圆柱体抓握 (Cylindrical), 我们给标签 0
    cyl_ch1, cyl_ch2 = extract_gesture_data(mat_data, 'cyl')
    if cyl_ch1 is not None:
        # 取前 5 次动作数据（为了演示速度快）
        for i in range(5):
            # 处理通道1
            clean_ch1 = clean_emg_signal(cyl_ch1[i])
            feat_ch1 = extract_features_sliding_window(clean_ch1)
            # 处理通道2
            clean_ch2 = clean_emg_signal(cyl_ch2[i])
            feat_ch2 = extract_features_sliding_window(clean_ch2)

            # 将两个通道的特征拼接起来作为一个样本的特征
            # 注意: 假设 feat_ch1 和 feat_ch2 长度一致
            for f1, f2 in zip(feat_ch1, feat_ch2):
                X_features.append([f1, f2])
                y_labels.append(0)  # 0 代表 Cylindrical

    # 动作2: 球形抓握 (Spherical), 我们给标签 1
    sph_ch1, sph_ch2 = extract_gesture_data(mat_data, 'spher')
    if sph_ch1 is not None:
        for i in range(5):
            clean_ch1 = clean_emg_signal(sph_ch1[i])
            feat_ch1 = extract_features_sliding_window(clean_ch1)
            clean_ch2 = clean_emg_signal(sph_ch2[i])
            feat_ch2 = extract_features_sliding_window(clean_ch2)

            for f1, f2 in zip(feat_ch1, feat_ch2):
                X_features.append([f1, f2])
                y_labels.append(1)  # 1 代表 Spherical

    return np.array(X_features), np.array(y_labels)


if __name__ == "__main__":
    print("🚀 开始测试大作业流水线...\n")

    # 1. 准备数据
    X, y = create_training_dataset('female_1.mat')

    if X is not None and len(X) > 0:
        print(f"特征矩阵形状: {X.shape}, 标签数组形状: {y.shape}")

        # 2. 初始化分类器
        classifier = GestureClassifier()

        # 3. 训练模型 (这里为了演示，直接用全部数据训练)
        classifier.train(X, y)

        # 4. 模拟一次预测 (拿第一条数据测一下)
        test_sample = X[0].reshape(1, -1)  # 变成 2D 数组送入预测
        pred = classifier.predict(test_sample)

        if pred[0] == 0:
            print("\n🎉 预测结果: 圆柱体抓握 (Cylindrical)")
        else:
            print("\n🎉 预测结果: 球形抓握 (Spherical)")

    else:
        print("❌ 数据加载失败，请检查文件路径是否正确 (../sEMG-signal-classification-master/sEMG-signal-classification-masterdata/data/Database 1/female_1.mat)")