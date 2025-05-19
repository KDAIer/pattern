# PCA 图像压缩与重建实验

本项目基于自实现的 PCA（主成分分析）算法，完成了对 EigenFace 灰度人脸数据集和一张彩色图片（woman.jpg）的降维压缩与重建实验，分析不同主成分数量对重建效果的影响，并将结果以图片形式保存。

---

## 📁 项目目录结构

.
├── data/                         # 数据文件夹
│   ├── faces.mat                 # EigenFace 灰度人脸数据集
│   └── woman.jpg                 # 彩色测试图像
│
├── results/
│   └── PCA/                      # 实验结果图片保存目录
│       ├── eigen\_faces.jpg       # 前 49 个特征脸组成的网格图
│       ├── recovered\_faces\_top\_10.jpg
│       ├── recovered\_faces\_top\_50.jpg
│       ├── recovered\_faces\_top\_100.jpg
│       ├── recovered\_faces\_top\_150.jpg

│       ├── woman_original.jpg       # 原始woman图像
│       ├── recovered\_woman\_top\_10.jpg
│       ├── recovered\_woman\_top\_50.jpg
│       ├── recovered\_woman\_top\_100.jpg
│       └── recovered\_woman\_top\_150.jpg
│
├── src/                          # 源码目录
│   ├── pca.py                    # 自实现 PCA 算法类
│   ├── run\_faces.py              # EigenFace 数据集压缩与重建主程序
│   ├── run\_woman.py              # 彩色图片压缩与重建主程序
│
├── Homework2
├── README.md                     # 项目说明文档

---

## 📦 功能说明

- **pca.py**：实现 `PCA` 类，包含 `fit`、`transform`、`inverse_transform` 等方法。
- **run_faces.py**：对 `data/faces.mat` 中人脸图像进行 PCA 压缩、重建，并保存不同维度下的重建效果。
- **run_woman.py**：对 `data/woman.jpg` 彩色图片的 RGB 通道分别做 PCA 降维重建，保存不同维度的重建图。
- **results/PCA/**：保存所有压缩重建结果图像。

---

## 📖 使用方法

1. 安装依赖库：

   ```bash
   pip install numpy matplotlib scipy
   ```
2. 运行人脸数据压缩与重建：

   ```bash
   python src/run_faces.py
   ```
3. 运行 woman 图片压缩与重建：

   ```bash
   python src/run_woman.py
   ```
4. 结果图片保存在 `results/PCA/` 目录下。

---

## 📌 环境

* Python 3.8+
* numpy
* matplotlib
* scipy

---
