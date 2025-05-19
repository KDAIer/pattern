# 图像拼接实验

本项目实现了基于 Harris 角点检测、SIFT 和 HOG 特征描述子的图像拼接系统，支持两图及多图自动拼接，包含角点检测、特征提取、匹配、RANSAC估计仿射/单应矩阵、图像配准及融合等完整流程。

## 项目结构

```bash
.
├── src/
│   ├── harris.py                     # Harris角点检测算法实现
│   ├── sift.py                       # 基于OpenCV封装的SIFT特征提取模块
│   ├── hog.py                        # HOG特征描述子提取模块
│   ├── ransac.py                     # 特征匹配与RANSAC估计模块
│   ├── match.py                       # SIFT 和 HOG的关键点描述和匹配以及二图拼接
│   ├── _connect.py                      # 多图拼接

├── images/ ... (略)

├── results/
│   └── 1/
│       ├── sudoku_keypoints.png             # Harris检测结果
│       ├── uttower1_keypoints.jpg           # Harris检测结果
│       ├── uttower2_keypoints.jpg           # Harris检测结果
│       ├── uttower_match_sift.png           # SIFT关键点匹配
│       ├── uttower_match_hog.png            # HOG关键点匹配
│       ├── uttower_stitching_sift.png       # SIFT拼接结果
│       ├── uttower_stitching_hog.png        # HOG拼接结果
│       └── yosemite_stitching.png           # 多图拼接结果

├── README.md                     # 项目说明文件（本文件）
```
