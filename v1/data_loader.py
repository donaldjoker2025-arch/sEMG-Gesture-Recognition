import scipy.io as sio
import os
import numpy as np


def load_subject_data(subject_filename, base_path='../sEMG-signal-classification-master/sEMG-signal-classification-master/data/Database 1/'):
    """
    加载单个受试者的 .mat 数据文件。

    参数:
    subject_filename (str): 文件名，例如 'female_1.mat'
    base_path (str): 数据所在的相对路径

    返回:
    dict: 包含受试者各手势和通道数据的字典。
          如果文件不存在，返回 None。
    """
    file_path = os.path.join(base_path, subject_filename)
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 {file_path}。请检查路径。")
        return None

    try:
        mat_data = sio.loadmat(file_path)
        # 清理不必要的系统级 key (如 '__header__', '__version__' 等)
        clean_data = {k: v for k, v in mat_data.items() if not k.startswith('__')}
        return clean_data
    except Exception as e:
        print(f"❌ 错误: 读取文件 {file_path} 时出错。原因: {e}")
        return None


def extract_gesture_data(mat_dict, gesture_prefix):
    """
    从加载的字典中提取特定手势的两个通道数据。

    参数:
    mat_dict (dict): load_subject_data 返回的字典
    gesture_prefix (str): 手势前缀，例如 'cyl' (圆柱体), 'hook' (钩状)

    返回:
    tuple: (通道1数据矩阵, 通道2数据矩阵)
    """
    try:
        ch1_data = mat_dict[f'{gesture_prefix}_ch1']
        ch2_data = mat_dict[f'{gesture_prefix}_ch2']
        return ch1_data, ch2_data
    except KeyError:
        print(f"⚠️ 警告: 数据字典中找不到以 '{gesture_prefix}' 开头的动作数据。")
        return None, None