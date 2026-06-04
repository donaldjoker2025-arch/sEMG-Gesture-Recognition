import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from data_loader import load_subject_data, extract_gesture_data
from signal_processor import clean_emg_signal
# 注意这里改成 V3
from feature_extractor_v3 import extract_features_sliding_window
from model_trainer_v3 import GestureClassifier  # 模型本身不用变


def create_full_dataset_v3(subject_filename='female_1.mat'):
    print(f"正在加载 {subject_filename} 的数据 (V3 终极三维特征融合版)...")
    mat_data = load_subject_data(subject_filename)
    if not mat_data: return None, None

    X_features = []
    y_labels = []
    gestures = {'cyl': 0, 'spher': 1, 'hook': 2, 'tip': 3}

    for prefix, label in gestures.items():
        ch1, ch2 = extract_gesture_data(mat_data, prefix)
        if ch1 is not None:
            for i in range(len(ch1)):
                clean_1 = clean_emg_signal(ch1[i])
                feat_1 = extract_features_sliding_window(ch1[i], clean_1)

                clean_2 = clean_emg_signal(ch2[i])
                feat_2 = extract_features_sliding_window(ch2[i], clean_2)

                # f1 此时是 [mav1, zc1, wl1]
                # f2 此时是 [mav2, zc2, wl2]
                for f1, f2 in zip(feat_1, feat_2):
                    # 拼接后变成 6 维特征向量！
                    combined_features = f1 + f2
                    X_features.append(combined_features)
                    y_labels.append(label)

    return np.array(X_features), np.array(y_labels)


if __name__ == "__main__":
    X, y = create_full_dataset_v3('female_1.mat')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")

    classifier = GestureClassifier()
    classifier.train(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n🏆 【V3 突破版】测试集准确率: {accuracy * 100:.2f}%\n")
    print("📊 V3 详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=['Cylindrical', 'Spherical', 'Hook', 'Tip']))