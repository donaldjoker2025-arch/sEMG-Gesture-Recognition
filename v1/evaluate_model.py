import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 导入我们的核心模块
from data_loader import load_subject_data, extract_gesture_data
from signal_processor import clean_emg_signal
from feature_extractor import extract_features_sliding_window
from model_trainer import GestureClassifier


def create_full_dataset(subject_filename='female_1.mat'):
    """组装包含4种手势的完整数据集"""
    print(f"正在加载 {subject_filename} 的数据...")
    mat_data = load_subject_data(subject_filename)
    if not mat_data: return None, None

    X_features = []
    y_labels = []

    # 定义我们要提取的 4 种手势及其对应标签
    gestures = {
        'cyl': 0,  # 圆柱体抓握
        'spher': 1,  # 球形抓握
        'hook': 2,  # 钩状抓握
        'tip': 3  # 指尖捏握
    }

    for prefix, label in gestures.items():
        ch1, ch2 = extract_gesture_data(mat_data, prefix)
        if ch1 is not None:
            # 遍历该动作的每一次重复 (通常有30次)
            for i in range(len(ch1)):
                # 处理并提取特征
                clean_1 = clean_emg_signal(ch1[i])
                feat_1 = extract_features_sliding_window(clean_1)

                clean_2 = clean_emg_signal(ch2[i])
                feat_2 = extract_features_sliding_window(clean_2)

                # 将两个通道的 MAV 组合成特征向量
                for f1, f2 in zip(feat_1, feat_2):
                    X_features.append([f1, f2])
                    y_labels.append(label)

    return np.array(X_features), np.array(y_labels)


if __name__ == "__main__":
    # 1. 获取数据
    X, y = create_full_dataset('female_1.mat')

    # 2. 划分训练集 (80%) 和 测试集 (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")

    # 3. 训练模型
    classifier = GestureClassifier()
    classifier.train(X_train, y_train)

    # 4. 在测试集上进行预测
    y_pred = classifier.predict(X_test)

    # 5. 计算并打印准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🏆 当前基础模型的测试集准确率: {accuracy * 100:.2f}%\n")

    # 打印详细的分类报告 (大作业报告利器)
    print("📊 详细分类报告:")
    target_names = ['Cylindrical', 'Spherical', 'Hook', 'Tip']
    print(classification_report(y_test, y_pred, target_names=target_names))