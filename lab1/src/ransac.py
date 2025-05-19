import cv2
import numpy as np


def match_and_draw(img1, kp1, des1, img2, kp2, des2, output_path, desc_type="SIFT"):
    """
    利用 BFMatcher（欧几里得距离）匹配描述子，并绘制匹配结果
    参数:
        img1, img2: 原图（彩色）
        kp1, kp2: 关键点列表
        des1, des2: 描述子
        output_path: 匹配结果图像保存路径
        desc_type: 描述子类型字符串，仅用于显示
    """
    if des1 is None or des2 is None:
        print(f"{desc_type} 描述子为空，无法匹配！")
        return None
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
    cv2.imwrite(output_path, img_matches)
    print(f"{desc_type} 匹配结果保存至: {output_path}")
    return matches


def stitch_images(img1, img2, kp1, kp2, matches, output_path, desc_type="SIFT"):
    """
    根据匹配关键点对利用 RANSAC 求仿射变换矩阵，
    将 img2 进行仿射变换并拼接到 img1 上，保存拼接结果。
    """
    if matches is None or len(matches) < 3:
        print(f"{desc_type} 匹配不足，无法拼接图像！")
        return

    # 提取匹配点对
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # 利用 RANSAC 求解仿射变换矩阵
    M, inliers = cv2.estimateAffine2D(
        pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=5
    )
    if M is None:
        print(f"{desc_type} RANSAC 无法求解仿射矩阵！")
        return

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    # 将 img2 的角点映射到 img1 坐标系
    pts_img2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    pts_img2_trans = cv2.transform(pts_img2, M)
    pts_img1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    all_pts = np.concatenate((pts_img1, pts_img2_trans), axis=0)
    [xmin, ymin] = np.int32(all_pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_pts.max(axis=0).ravel() + 0.5)

    # 构造平移矩阵 T (3x3)
    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float32)
    # 将 M 扩展为 3x3 齐次矩阵
    M_hom = np.vstack([M, [0, 0, 1]])
    # 组合变换：先仿射变换 M，再平移 T
    combined_transform = T.dot(M_hom)  # 3x3 矩阵

    output_size = (xmax - xmin, ymax - ymin)
    warped_img2 = cv2.warpAffine(img2, combined_transform[:2, :], output_size)

    # 将 img1 放入拼接图中
    stitched_img = warped_img2.copy()
    stitched_img[-ymin : h1 - ymin, -xmin : w1 - xmin] = img1

    cv2.imwrite(output_path, stitched_img)
    print(f"{desc_type} 拼接结果已保存至: {output_path}")
