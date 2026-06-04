# 面向智能假肢的表面肌电（sEMG）手势意图识别系统

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-0.24+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

本项目是一个面向智能假肢控制场景的表面肌电（sEMG）信号处理与手势分类系统。通过采集并解析人体前臂的表面肌电信号，系统能够高精度识别四类精细抓握手势（圆柱体抓握、球形抓握、钩状抓握、指尖捏握），并提供了一个交互式的 Web 可视化临床分析看板。

> **🎉 在线体验：** 本系统已部署至云端，无需配置环境，请点击下方徽章直接体验交互式意图识别！

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://semg-gesture-demo.streamlit.app)

## ✨ 核心特性与创新点

相比于传统的全局离线分析，本项目围绕真实临床需求进行了深度重构与创新：
- **双轨制信号预处理**：首创“幅值轨 + 频率轨”双通道处理策略，在提取平滑包络（MAV特征）的同时，完好保留了原始信号的高频颤动信息（ZC、WL特征）。
- **多维特征融合**：从“强度-节律-复杂度”三个物理维度提取 MAV + ZC + WL 特征，构建 6 维特征空间。
- **伪实时滑动窗口**：设计了重叠率为 50% 的滑动窗口（Window=500, Step=250），满足假肢控制 300ms 以内的低延迟实时性要求。
- **高精度分类引擎**：基于径向基核函数（RBF）的支持向量机（SVM），在独立测试集上突破了 **90.15%** 的识别准确率。
- **交互式 Web 看板**：基于 Streamlit 开发了数据可视化界面，支持动态波形渲染、特征指标监控与意图实时解码。

## 📂 项目结构

项目分为多个迭代版本（v1-v3），其中 `v3` 为最终的高性能多维特征融合版。

```text
├── v3/                             # 最终版本核心代码目录
│   ├── app.py                      # Streamlit Web 前端主程序
│   ├── data_loader.py              # .mat 数据解析与加载模块
│   ├── signal_processor.py         # 信号清洗与双轨预处理模块
│   ├── feature_extractor_v3.py     # 滑动窗口与多维特征提取模块
│   ├── model_trainer_v3.py         # SVM 模型训练与封装模块
│   ├── evaluate_model_v3.py        # 模型性能评估与指标计算
│   ├── generate_report_figures.py  # 实验报告图表自动化生成脚本
│   └── 报告图表...                 # 生成的评估可视化图片
├── v1/                             # 早期单特征探索版本（归档）
├── v2/                             # 双特征迭代版本（归档）
├── .gitignore                      # Git 忽略配置文件
└── README.md                       # 项目说明文档
```
## 🗄️ 数据集获取与配置（必读）

**注意：** 为避免仓库体积过大，本项目并未包含体积庞大的 `.mat` 原始数据集文件。在运行代码前，请按照以下步骤配置数据：

1. 访问加州大学欧文分校（UCI）机器学习库，下载 **sEMG for Basic Hand movements Data Set**。
2. 在项目根目录下，手动创建一个名为 `sEMG-signal-classification-master/data/Database 1/` 的多级文件夹结构。
3. 将下载的 `female_1.mat`, `male_1.mat` 等受试者数据放入该 `Database 1` 文件夹中。

数据存放路径示例：
`./sEMG-signal-classification-master/data/Database 1/female_1.mat`

## 🚀 环境依赖与运行指南

本项目在 Windows 操作系统下开发测试，推荐使用 PyCharm 等 JetBrains IDE 以获得最佳的代码补全与调试体验。

### 1. 安装依赖环境
请在终端（Terminal）中运行以下命令安装必备的 Python 库：
`pip install numpy pandas scipy scikit-learn matplotlib seaborn streamlit`

### 2. 评估模型性能与生成图表
进入 `v3` 目录，运行评估脚本即可查看模型在测试集上的精确率、召回率及 F1 分数，并一键生成论文级图表：
`cd v3`
`python evaluate_model_v3.py`
`python generate_report_figures.py`

### 3. 启动交互式可视化看板
如需体验包含动态波形渲染与实时预测的 UI 界面，请运行：
`streamlit run app.py`

## 📊 实验结果

经过优化，V3 版本的模型在多分类任务中表现优异，尤其是在识别细微发力差异的动作（如指尖捏握与钩状抓握）时，波形长度（WL）特征的引入极大提升了分类边界的清晰度。全局测试准确率稳定在 **90.15%**。

## 🙏 致谢与参考

本项目的早期数据探索参考了 Hayden Cornell 与 Numaer Zaker 的开源离线分析框架。在此对原作者及 UCI 机器学习库提供的高质量开源数据表示衷心感谢。本项目在此基础上进行了算法流水线重构、多维特征工程升级以及系统级可视化开发。

## 👨‍💻 开发者

**Zhanyan Ren** - *医学电子(2)课程设计项目*