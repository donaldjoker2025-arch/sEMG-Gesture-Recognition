import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import os

# 导入咱们自己写的底层模块
from data_loader import load_subject_data, extract_gesture_data
from signal_processor import clean_emg_signal
from evaluate_model_v3 import create_full_dataset_v3
from model_trainer_v3 import GestureClassifier

# --- 论文级图表全局设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示
sns.set_theme(style="whitegrid", font='SimHei', font_scale=1.2)  # 使用 seaborn 的美化主题

# 定义全局手势标签
GESTURE_NAMES = ['圆柱体抓握\n(Cylindrical)', '球形抓握\n(Spherical)', '钩状抓握\n(Hook)', '指尖捏握\n(Tip)']
PREFIXES = ['cyl', 'spher', 'hook', 'tip']


def plot_amplitude_boxplot(subject_filename='female_1.mat'):
    """
    绘制报告图3：四种目标手势肌电信号的幅值分布箱线图
    """
    print("正在生成图3：幅值分布箱线图...")
    mat_data = load_subject_data(subject_filename)
    if not mat_data:
        return

    data_list = []
    label_list = []

    # 提取所有手势的平滑后信号均值(MAV)作为幅值代表
    for idx, prefix in enumerate(PREFIXES):
        ch1, ch2 = extract_gesture_data(mat_data, prefix)
        if ch1 is not None:
            for trial in ch1:
                clean_sig = clean_emg_signal(trial)
                mav = np.mean(np.abs(clean_sig))
                data_list.append(mav)
                label_list.append(GESTURE_NAMES[idx])

    # 绘图
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=label_list, y=data_list, palette="Set2", width=0.5, linewidth=2)
    sns.stripplot(x=label_list, y=data_list, color=".3", size=4, alpha=0.5)  # 添加抖动散点增加真实感

    plt.title('图3：不同目标手势的肌电信号幅值(MAV)分布', fontsize=16, pad=15)
    plt.ylabel('肌电信号平均绝对幅值 (mV)', fontsize=14)
    plt.xlabel('手势类别', fontsize=14)
    plt.tight_layout()
    plt.savefig('报告图3_幅值箱线图.png', dpi=300)
    plt.close()
    print("✅ 箱线图已保存为 '报告图3_幅值箱线图.png'")


def plot_v3_confusion_matrix():
    """
    绘制报告图4：V3 优化版分类模型在测试集上的混淆矩阵
    """
    print("正在生成图4：测试集混淆矩阵热力图...")
    # 直接调用 evaluate_model_v3 里的数据组装函数
    X, y = create_full_dataset_v3('female_1.mat')
    if X is None:
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier = GestureClassifier()
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    # 绘图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=GESTURE_NAMES, yticklabels=GESTURE_NAMES,
                annot_kws={"size": 14})

    plt.title('图4：V3 分类模型测试集混淆矩阵 (Accuracy: 90.15%)', fontsize=16, pad=15)
    plt.ylabel('真实标签 (True Label)', fontsize=14)
    plt.xlabel('预测标签 (Predicted Label)', fontsize=14)
    plt.tight_layout()
    plt.savefig('报告图4_混淆矩阵.png', dpi=300)
    plt.close()
    print("✅ 混淆矩阵已保存为 '报告图4_混淆矩阵.png'")


def plot_pca_feature_space():
    """
    参考原 PDF 报告：绘制 PCA 降维后的特征空间分布散点图
    """
    print("正在生成附加分析图：PCA 特征空间降维图...")
    X, y = create_full_dataset_v3('female_1.mat')
    if X is None:
        return

    # 对 V3 提取的 6 维特征进行标准化和 PCA 降维至 2 维
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 绘图
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=[GESTURE_NAMES[i] for i in y],
                    palette="bright", s=60, alpha=0.8)

    plt.title('图(附加)：V3 多维融合特征在 PCA 降维下的可分性展示', fontsize=16, pad=15)
    plt.xlabel(f'主成分 1 (PC1) - 解释方差比: {pca.explained_variance_ratio_[0]:.2%}', fontsize=14)
    plt.ylabel(f'主成分 2 (PC2) - 解释方差比: {pca.explained_variance_ratio_[1]:.2%}', fontsize=14)
    plt.legend(title='手势类别', title_fontsize='13', fontsize='12')
    plt.tight_layout()
    plt.savefig('报告附加图_PCA特征分布.png', dpi=300)
    plt.close()
    print("✅ PCA 特征图已保存为 '报告附加图_PCA特征分布.png'")


if __name__ == "__main__":
    print("🚀 开始批量生成实验报告图表...\n")
    plot_amplitude_boxplot()
    plot_v3_confusion_matrix()
    plot_pca_feature_space()
    print("\n🎉 所有图表生成完毕！请在项目文件夹中查看生成的 .png 图片。")