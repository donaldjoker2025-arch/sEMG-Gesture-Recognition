import numpy as np


def extract_features_sliding_window(raw_signal, clean_signal, window_size=500, step_size=250):
    """
    【V2.1 终极版】同步滑动窗口。
    ZC (频域) 必须在原始信号上计算；
    MAV (时域) 在平滑信号上计算。
    """
    features = []
    # 确保两个信号长度一致
    num_samples = min(len(raw_signal), len(clean_signal))

    for start_idx in range(0, num_samples - window_size + 1, step_size):
        end_idx = start_idx + window_size

        # 截取两个版本的窗口
        raw_window = raw_signal[start_idx:end_idx]
        clean_window = clean_signal[start_idx:end_idx]

        # 1. 计算 MAV (使用平滑后的信号)
        mav = np.mean(np.abs(clean_window))

        # 2. 计算 ZC (必须使用有正有负的原始信号)
        signs = np.sign(raw_window)
        signs[signs == 0] = -1  # 防止 0 值干扰
        zero_crossings = np.sum(np.abs(np.diff(signs)) > 0)

        features.append([mav, zero_crossings])

    return features