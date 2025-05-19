import os
import numpy as np
from PIL import Image
from pca import PCA


def compress_and_reconstruct_image(img_array: np.ndarray, k: int) -> np.ndarray:
    H, W, C = img_array.shape
    reconstructed = np.zeros_like(img_array, dtype=np.float64)

    # 对 R/G/B 三个通道分别处理
    for ch in range(C):
        channel = img_array[:, :, ch].astype(np.float64)
        pca = PCA(n_components=k)
        Z = pca.fit_transform(channel)  # shape=(H, k)
        rec_channel = pca.inverse_transform(Z)  # shape=(H, W)
        reconstructed[:, :, ch] = rec_channel

    # 裁剪并转换为 uint8
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    return reconstructed


def main():
    img_path = os.path.join("data", "woman.png")
    out_dir = os.path.join("results", "PCA")
    os.makedirs(out_dir, exist_ok=True)

    # 原始图像
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)
    orig_out = os.path.join(out_dir, "woman_original.jpg")
    img.save(orig_out)
    print(f"Saved original image to {orig_out}")

    # 对不同 k 值进行压缩重建
    for k in [10, 50, 100, 150]:
        rec = compress_and_reconstruct_image(img_array, k)
        rec_img = Image.fromarray(rec)
        out_path = os.path.join(out_dir, f"recovered_woman_top_{k}.jpg")
        rec_img.save(out_path)
        print(f"Saved reconstructed image with k={k} to {out_path}")


if __name__ == "__main__":
    main()
