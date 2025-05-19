import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 导入自定义模块
import harris
import sift
import hog
import ransac


def main():
    output_dir = os.path.join(BASE_DIR, "results", "1")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 图像路径设置
    img1_path = os.path.join(BASE_DIR, "images", "1", "uttower1.jpg")
    img2_path = os.path.join(BASE_DIR, "images", "1", "uttower2.jpg")

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print("无法加载图像，请检查路径！")
        return

    # 转换为灰度图
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # 对 Harris 角点检测需要用 float32 格式
    gray1_float = np.float32(gray1)
    gray2_float = np.float32(gray2)

    # ---------------------------
    # Harris 角点检测
    # ---------------------------
    harris_kp1 = harris.extract_harris_keypoints(gray1_float, threshold_ratio=0.01)
    harris_kp2 = harris.extract_harris_keypoints(gray2_float, threshold_ratio=0.01)
    print(f"图像1 Harris 角点数量: {len(harris_kp1)}")
    print(f"图像2 Harris 角点数量: {len(harris_kp2)}")

    # 绘制 Harris 角点结果并保存
    img1_harris = cv2.drawKeypoints(img1, harris_kp1, None, color=(0, 0, 255))
    img2_harris = cv2.drawKeypoints(img2, harris_kp2, None, color=(0, 0, 255))
    cv2.imwrite(os.path.join(output_dir, "uttower1_keypoints.jpg"), img1_harris)
    cv2.imwrite(os.path.join(output_dir, "uttower2_keypoints.jpg"), img2_harris)

    # ---------------------------
    # SIFT 描述子与匹配
    # ---------------------------
    kp1_sift, des1_sift = sift.compute_sift_descriptors(gray1, harris_kp1)
    kp2_sift, des2_sift = sift.compute_sift_descriptors(gray2, harris_kp2)
    sift_match_path = os.path.join(output_dir, "uttower_match_sift.png")
    sift_matches = ransac.match_and_draw(
        img1,
        kp1_sift,
        des1_sift,
        img2,
        kp2_sift,
        des2_sift,
        sift_match_path,
        desc_type="SIFT",
    )

    # 基于 SIFT 匹配关键点利用 RANSAC 进行拼接
    stitch_sift_path = os.path.join(output_dir, "uttower_stitching_sift.png")
    ransac.stitch_images(
        img1, img2, kp1_sift, kp2_sift, sift_matches, stitch_sift_path, desc_type="SIFT"
    )

    # ---------------------------
    # HOG 描述子与匹配
    # ---------------------------
    kp1_hog, des1_hog = hog.compute_hog_descriptors(gray1, harris_kp1, patch_size=32)
    kp2_hog, des2_hog = hog.compute_hog_descriptors(gray2, harris_kp2, patch_size=32)
    hog_match_path = os.path.join(output_dir, "uttower_match_hog.png")
    hog_matches = ransac.match_and_draw(
        img1,
        kp1_hog,
        des1_hog,
        img2,
        kp2_hog,
        des2_hog,
        hog_match_path,
        desc_type="HOG",
    )

    # 基于 HOG 匹配关键点利用 RANSAC 进行拼接
    stitch_hog_path = os.path.join(output_dir, "uttower_stitching_hog.png")
    ransac.stitch_images(
        img1, img2, kp1_hog, kp2_hog, hog_matches, stitch_hog_path, desc_type="HOG"
    )

    # ---------------------------
    # 分析对比（在论文或报告中阐述，不在代码中显示）
    # ---------------------------
    """
    对比分析：
    
    1. SIFT 描述子:
       - 具备较强的尺度与旋转不变性，生成的128维描述子在匹配时表现鲁棒，
         RANSAC 求解出的仿射变换较为准确，因此拼接结果较为自然、平滑。
    
    2. HOG 描述子:
       - 主要依靠梯度方向直方图，描述子维度较低，对局部纹理信息敏感，
         但在尺度和旋转不变性上不如 SIFT，
         因此匹配过程可能出现误匹配，进而导致拼接结果在过渡区域存在瑕疵。
    
    总结：在本实验中 SIFT 特征的匹配与拼接效果普遍优于 HOG 特征，
    而 HOG 特征虽然计算简单，但匹配稳定性相对较差。
    """


if __name__ == "__main__":
    main()
