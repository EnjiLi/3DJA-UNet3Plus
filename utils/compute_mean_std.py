import os
from PIL import Image
import numpy as np


def main():
    img_channels = 3
    img_dir = "./INRIA/train/image"
    roi_dir = "./INRIA/train/label"
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."
    assert os.path.exists(roi_dir), f"roi dir: '{roi_dir}' does not exist."

    img_name_list = [i for i in os.listdir(img_dir) if i.endswith(".png")]
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        roi_path = os.path.join(roi_dir, img_name)
        img = np.array(Image.open(img_path)) / 255.
        roi_img = np.array(Image.open(roi_path).convert('L'))


        # img = img[roi_img == 255]
        mean = np.nanmean(img, axis=(0, 1))
        std = np.nanstd(img, axis=(0, 1))
        cumulative_mean += mean
        cumulative_std += std

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean: {mean}")
    print(f"std: {std}")


if __name__ == '__main__':
    main()
