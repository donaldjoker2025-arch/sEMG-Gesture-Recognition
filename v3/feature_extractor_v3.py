import numpy as np


def extract_features_sliding_window(raw_signal, clean_signal, window_size=500, step_size=250):
    """
    【V3 终极突破版】特征融合：MAV (振幅) + ZC (频率) + WL (复杂度)
    """
    features = []
    num_samples = min(len(raw_signal), len(clean_signal))

    for start_idx in range(0, num_samples - window_size + 1, step_size):
        end_idx = start_idx + window_size

        raw_window = raw_signal[start_idx:end_idx]
        clean_window = clean_signal[start_idx:end_idx]

        # 1. 特征一：MAV (平均绝对值 - 振幅)
        mav = np.mean(np.abs(clean_window))

        # 2. 特征二：ZC (过零率 - 频率)
        signs = np.sign(raw_window)
        signs[signs == 0] = -1
        zero_crossings = np.sum(np.abs(np.diff(signs)) > 0)

        # 3. 特征三：WL (波形长度 - 复杂度/抖动程度)
        # 严格按照数学公式：计算相邻两点差值的绝对值之和
        # 使用 clean_window (平滑信号) 计算 WL 更加稳定，能抗干扰
        wl = np.sum(np.abs(np.diff(clean_window)))

        # 将三个特征打包返回
        features.append([mav, zero_crossings, wl])

    return features