import cv2
import numpy as np


def compute_hog_descriptor_for_patch(patch):
    """
    对输入 patch（期望尺寸为32x32）计算 HOG 描述子
    """
    if patch.shape[0] != 32 or patch.shape[1] != 32:
        patch = cv2.resize(patch, (32, 32))
    hog = cv2.HOGDescriptor(
        _winSize=(32, 32),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    descriptor = hog.compute(patch)
    return descriptor.flatten()


def compute_hog_descriptors(gray_uint8, keypoints, patch_size=32):
    """
    对给定灰度图和 Harris 检测到的关键点列表，
    提取每个关键点周围的 patch，并计算 HOG 描述子；
    返回有效关键点列表及描述子矩阵
    """
    descriptors_list = []
    valid_kps = []
    h, w = gray_uint8.shape
    half = patch_size // 2
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if x - half < 0 or x + half > w or y - half < 0 or y + half > h:
            continue
        patch = gray_uint8[y - half : y + half, x - half : x + half]
        hog_desc = compute_hog_descriptor_for_patch(patch)
        descriptors_list.append(hog_desc)
        valid_kps.append(cv2.KeyPoint(float(x), float(y), patch_size))
    if len(descriptors_list) == 0:
        return valid_kps, None
    descriptors = np.array(descriptors_list, dtype=np.float32)
    return valid_kps, descriptors
