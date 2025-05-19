import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pca import PCA


def load_faces(mat_path: str) -> np.ndarray:
    """从 .mat 文件加载人脸数据，返回形状 (n_samples, n_features)"""
    data = sio.loadmat(mat_path)
    X = data.get("X", None)
    if X is None:
        raise KeyError("MAT 文件中未找到键 'X'")
    # 如果是 (1024, N)，转为 (N, 1024)
    if X.shape[0] == 1024:
        X = X.T
    return X  # (n_samples, 1024)


def save_grid(
    images: np.ndarray, grid_shape: tuple, img_size: tuple, out_path: str, cmap="gray"
):
    """
    将 images 按 grid_shape 排列并保存为一张图。
    images: (N, H*W)
    grid_shape: (rows, cols)
    img_size: (H, W)
    """
    rows, cols = grid_shape
    H, W = img_size
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx < images.shape[0]:
            ax.imshow(images[idx].reshape(H, W), cmap=cmap)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved grid to {out_path}")


def main():
    data_path = os.path.join("data", "faces.mat")
    out_dir = os.path.join("results", "PCA")
    os.makedirs(out_dir, exist_ok=True)

    # 1. 加载人脸数据
    faces = load_faces(data_path)  # (n_samples, 1024)
    n_samples, n_features = faces.shape
    side = int(np.sqrt(n_features))
    assert side * side == n_features, f"不能将 {n_features} 重塑为正方形"

    # 2. 保存前 49 张原始人脸（7*7 网格）
    save_grid(
        faces[:49], (7, 7), (side, side), os.path.join(out_dir, "original_faces.jpg")
    )

    # # 3. 利用 PCA 提取前 49 个主成分并保存（7×7 网格）
    pca49 = PCA(n_components=49)
    pca49.fit(faces)
    eigenfaces = pca49.components_.T  # 转置后 shape=(49,1024)
    save_grid(
        eigenfaces, (7, 7), (side, side), os.path.join(out_dir, "eigen_faces.jpg")
    )

    # 4. 对不同 k 值进行压缩和重建，保存前 49 张重建结果（7×7 网格）
    for k in [10, 50, 100, 150]:
        pca = PCA(n_components=k)
        Z = pca.fit_transform(faces)  # (n_samples, k)
        faces_rec = pca.inverse_transform(Z)  # (n_samples, 1024)
        save_grid(
            faces_rec[:49],
            (7, 7),
            (side, side),
            os.path.join(out_dir, f"recovered_faces_top_{k}.jpg"),
        )


if __name__ == "__main__":
    main()
