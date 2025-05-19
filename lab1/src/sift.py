import cv2


def compute_sift_descriptors(gray_uint8, keypoints):
    """
    对给定的关键点使用 SIFT 描述子进行描述
    参数:
        gray_uint8: 8位灰度图
        keypoints: cv2.KeyPoint 列表
    返回:
        keypoints, descriptors
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.compute(gray_uint8, keypoints)
    return keypoints, descriptors
