import os
import cv2
import numpy as np

# 导入之前的模块
import harris
import sift
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# hog 和 ransac 模块在本例中不做修改，如有需要可保留
# import hog
# import ransac


def auto_crop(image):
    """
    自动裁剪掉图像中全黑（背景）区域
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 非零区域设为白（255）
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    cropped = image[y : y + h, x : x + w]
    return cropped


def stitch_multiple_images(image_paths, output_path):
    """
    1. 从 image_paths 中依次读取图像，以第一幅图为基准，
       依次计算每幅图到基准图的累积单应性矩阵。
    2. 根据所有变换后图像的角点确定公共画布大小与平移量。
    3. 将各幅图按照其累积变换投影到公共画布上，并进行简单融合，
       最后自动裁剪黑色区域，并保存结果到 output_path。
    """
    # 依次读取各幅图像
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"无法加载图像: {path}")
            return
        images.append(img)

    num_images = len(images)
    # cum_H_list[i] 表示第 i 幅图（i=0为基准）到基准图（第0幅图）的变换矩阵
    cum_H_list = [np.eye(3)]

    # 依次计算第 i 图（i>=1）到第 i-1 图之间的单应矩阵，然后累积转换
    for i in range(1, num_images):
        img_prev = images[i - 1]
        img_curr = images[i]
        # 转换为灰度图，供 Harris 与 SIFT 使用
        gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)

        # 利用 Harris 角点检测提取候选关键点
        harris_kp_prev = harris.extract_harris_keypoints(
            np.float32(gray_prev), threshold_ratio=0.01
        )
        harris_kp_curr = harris.extract_harris_keypoints(
            np.float32(gray_curr), threshold_ratio=0.01
        )

        # 用 SIFT 对候选关键点计算描述子
        kp_prev, des_prev = sift.compute_sift_descriptors(gray_prev, harris_kp_prev)
        kp_curr, des_curr = sift.compute_sift_descriptors(gray_curr, harris_kp_curr)

        if des_prev is None or des_curr is None:
            print("描述子计算失败")
            return

        # 使用 BFMatcher（交叉验证）进行匹配
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des_prev, des_curr)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < 4:
            print("匹配点不足，无法计算单应性")
            return

        # 构造匹配点对；注意 findHomography 的参数顺序为：将当前图像点转换到上一图像坐标系
        pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )
        pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )
        H, mask = cv2.findHomography(pts_curr, pts_prev, cv2.RANSAC, 5.0)
        if H is None:
            print("无法求解单应性矩阵")
            return
        # 累积变换：从当前图到基准图 = (从上一图到基准图) * (从当前图到上一图)
        cum_H = np.dot(cum_H_list[-1], H)
        cum_H_list.append(cum_H)
        print(f"第 {i+1} 张图与前一图匹配完成")

    # 计算所有图像在基准图坐标系下的角点位置，确定全局画布范围
    all_corners = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        pts_trans = cv2.perspectiveTransform(pts, cum_H_list[i])
        all_corners.append(pts_trans)
    all_corners = np.concatenate(all_corners, axis=0)
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # 构造平移矩阵：将所有图像平移使其全部位于正坐标区域
    translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float32)
    canvas_width = xmax - xmin
    canvas_height = ymax - ymin

    # 建立黑色画布，尺寸为全局范围
    panorama = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # 将各幅图像投影到全局画布中（依次叠加）
    for i, img in enumerate(images):
        warped = cv2.warpPerspective(
            img, translation.dot(cum_H_list[i]), (canvas_width, canvas_height)
        )
        # 简单叠加：非黑像素覆盖
        mask = (warped > 0).sum(axis=2) > 0  # 找到非黑区域
        panorama[mask] = warped[mask]

    # 自动裁剪掉依然存在的黑边
    panorama = auto_crop(panorama)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, panorama)
    print(f"多图拼接完成，结果已保存至 {output_path}")


def main():
    # 拼接 yosemite 系列图像
    yosemite_paths = [
        os.path.join(BASE_DIR, "images", "1", "yosemite1.jpg"),
        os.path.join(BASE_DIR, "images", "1", "yosemite2.jpg"),
        os.path.join(BASE_DIR, "images", "1", "yosemite3.jpg"),
        os.path.join(BASE_DIR, "images", "1", "yosemite4.jpg"),
    ]
    output_path = os.path.join(BASE_DIR, "results", "1", "yosemite_stitching.png")
    stitch_multiple_images(yosemite_paths, output_path)


if __name__ == "__main__":
    main()
