import numpy as np
import pandas as pd


def apply_absolute_value(signal_data):
    """
    对肌电信号取绝对值（全波整流）。
    这是 sEMG 处理的标准第一步。
    """
    return np.abs(signal_data)


def apply_exponential_smoothing(signal_data, alpha=0.1):
    """
    应用指数平滑算法 (Holt-Winters 的一种简单形式) 来提取信号包络。

    参数:
    signal_data (np.ndarray): 一维信号数组 (例如某一次动作的一个通道数据)
    alpha (float): 平滑系数，0 到 1 之间。越小越平滑。
    """
    # 将 NumPy 数组转为 Pandas Series 以便使用 ewm 函数
    series = pd.Series(signal_data)
    # 使用指数加权移动平均，adjust=False 对应标准的指数平滑公式
    smoothed = series.ewm(alpha=alpha, adjust=False).mean()
    return smoothed.values


def clean_emg_signal(raw_signal, alpha=0.1):
    """
    一键化信号清理流水线：绝对值化 -> 指数平滑。
    这对应你大作业要求的“信号处理”环节。
    """
    abs_signal = apply_absolute_value(raw_signal)
    smoothed_signal = apply_exponential_smoothing(abs_signal, alpha=alpha)
    return smoothed_signal