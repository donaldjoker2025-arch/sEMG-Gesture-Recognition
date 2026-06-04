import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class GestureClassifier:
    def __init__(self):
        """
        初始化分类器。
        使用 SVM (支持向量机)，核函数设为 'rbf' (径向基核函数)。
        这是大作业报告“方法”部分的得分点。
        """
        self.model = SVC(kernel='rbf', probability=True)
        self.scaler = StandardScaler()  # 数据标准化，对 SVM 很重要
        self.is_trained = False

    def train(self, X_train, y_train):
        """
        训练模型。
        X_train: 特征矩阵 (n_samples, n_features)
        y_train: 标签数组 (n_samples,)
        """
        if len(X_train) == 0:
            print("❌ 错误: 训练数据为空。")
            return False

        # 1. 标准化数据
        X_scaled = self.scaler.fit_transform(X_train)
        # 2. 训练 SVM
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        print(f"✅ 模型训练完成！共使用了 {len(X_train)} 个样本。")
        return True

    def predict(self, X_test):
        """
        预测新数据。
        """
        if not self.is_trained:
            print("⚠️ 警告: 模型尚未训练，无法预测。")
            return None

        # 必须使用训练时的 scaler 进行标准化
        X_scaled = self.scaler.transform(X_test)
        # 返回预测类别
        prediction = self.model.predict(X_scaled)
        return prediction