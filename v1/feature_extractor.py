import numpy as np


def extract_features_from_window(window_data):
    """
    从一个数据窗口中提取特征。
    这里我们提取最经典的 MAV (Mean Absolute Value, 平均绝对值)。
    你大作业报告里可以写：“本系统采用 MAV 作为时域特征”。
    """
    # 计算均值作为特征
    mav = np.mean(np.abs(window_data))
    return mav


def extract_features_sliding_window(signal_data, window_size=500, step_size=250):
    """
    使用滑动窗口机制遍历整个信号，提取特征序列。

    参数:
    signal_data (np.ndarray): 处理后的一维肌电信号
    window_size (int): 窗口大小 (例如500个采样点，对应1秒)
    step_size (int): 步进大小 (例如250个采样点，对应0.5秒重叠)

    返回:
    list: 提取出的特征值列表
    """
    features = []
    num_samples = len(signal_data)

    # 滑动窗口循环
    for start_idx in range(0, num_samples - window_size + 1, step_size):
        end_idx = start_idx + window_size
        window = signal_data[start_idx:end_idx]
        feature_val = extract_features_from_window(window)
        features.append(feature_val)

    return features