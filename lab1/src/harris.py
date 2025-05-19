import cv2
import numpy as np


def harris_corner_detection(gray, block_size=3, ksize=3, k=0.04):
    """
    手动实现 Harris 角点检测
    参数:
        gray: 灰度图像，类型 float32
        block_size: 用于加权求和的窗口大小（这里作为高斯滤波核大小的一部分）
        ksize: Sobel 算子核大小
        k: Harris 算子中的经验系数
    返回:
        R: 角点响应图，类型 float32
    """
    # 定义 Sobel 滤波核
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    # 计算梯度
    Ix = cv2.filter2D(gray, -1, sobel_x)
    Iy = cv2.filter2D(gray, -1, sobel_y)

    # 计算梯度的二次项
    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix * Iy

    # 利用高斯滤波器对二次项平滑
    Sxx = cv2.GaussianBlur(Ixx, (block_size, block_size), sigmaX=1)
    Syy = cv2.GaussianBlur(Iyy, (block_size, block_size), sigmaX=1)
    Sxy = cv2.GaussianBlur(Ixy, (block_size, block_size), sigmaX=1)

    # 计算角点响应 R = det(M) - k * trace(M)^2
    detM = Sxx * Syy - Sxy**2
    traceM = Sxx + Syy
    R = detM - k * (traceM**2)

    return R


def non_maximum_suppression(R, threshold_ratio=0.01, window_size=3):
    """
    对角点响应图进行阈值处理和非极大值抑制
    参数:
        R: 角点响应图
        threshold_ratio: 使用 R 最大值的比例作为阈值
        window_size: 非极大值抑制时邻域的尺寸
    返回:
        corners: 一个二值图像，值为 True 的位置为角点
    """
    threshold = threshold_ratio * R.max()
    R_threshold = (R > threshold) * R
    dilated = cv2.dilate(R_threshold, np.ones((window_size, window_size), np.uint8))
    corners = (R_threshold == dilated) & (R_threshold > 0)
    return corners


def extract_harris_keypoints(img_gray_float, threshold_ratio=0.01):
    """
    利用 Harris 角点检测提取关键点，并转换为 cv2.KeyPoint 列表
    参数:
        img_gray_float: 灰度图，float32 类型
        threshold_ratio: Harris 角点响应的阈值比例
    返回:
        keypoints: cv2.KeyPoint 列表
    """
    R = harris_corner_detection(img_gray_float)
    corners = non_maximum_suppression(R, threshold_ratio=threshold_ratio)
    # np.nonzero 返回 (y, x) 坐标
    y_coords, x_coords = np.nonzero(corners)
    keypoints = [
        cv2.KeyPoint(float(x), float(y), 6) for x, y in zip(x_coords, y_coords)
    ]
    return keypoints
