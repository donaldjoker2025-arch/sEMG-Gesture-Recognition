import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class GestureClassifier:
    def __init__(self):
        """
        初始化分类器 v2。
        维持使用 SVM (RBF核)，因为其在小样本非线性分类上表现极佳。
        """
        self.model = SVC(kernel='rbf', probability=True)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X_train, y_train):
        if len(X_train) == 0:
            print("❌ 错误: 训练数据为空。")
            return False

        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        print(f"✅ V2 模型训练完成！共使用了 {len(X_train)} 个样本。")
        return True

    def predict(self, X_test):
        if not self.is_trained:
            print("⚠️ 警告: 模型尚未训练，无法预测。")
            return None

        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)