import os
import cv2
import argparse
from PIL import Image
import numpy as np

def resize_with_antialiasing(img, size=(224, 224)):
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(size, Image.LANCZOS)
    return np.array(pil_img)

def apply_preprocessing(img, mode='clahe', image_size=(224, 224)):
    if mode == 'antialias':
        img = resize_with_antialiasing(img, image_size)
    else:
        img = cv2.resize(img, image_size)

    if mode == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
    elif mode == 'gaussian':
        img = cv2.GaussianBlur(img, (5, 5), 0)
    elif mode == 'sharpen':
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)
    elif mode == 'none':
        pass  
    elif mode == 'antialias':
        pass 
    else:
        raise ValueError(f"Unsupported preprocessing mode: {mode}")

    return img

def preprocess_images(input_dir, output_dir, mode='clahe', image_size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)

    for cls in os.listdir(input_dir):
        input_cls_path = os.path.join(input_dir, cls)

        if not os.path.isdir(input_cls_path):
            continue

        output_cls_path = os.path.join(output_dir, cls)
        os.makedirs(output_cls_path, exist_ok=True)

        for fname in os.listdir(input_cls_path):
            if not fname.lower().endswith('.png'):
                continue
            in_path = os.path.join(input_cls_path, fname)
            out_path = os.path.join(output_cls_path, fname)

            img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"跳过损坏图像：{in_path}")
                continue

            img_processed = apply_preprocessing(img, mode=mode, image_size=image_size)
            cv2.imwrite(out_path, img_processed)

    print(f"[{mode}] 预处理完成，图像保存至：{output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/raw')
    parser.add_argument('--output_dir', type=str, default='data/processed')
    parser.add_argument('--mode', type=str, default='none', choices=['clahe', 'gaussian', 'sharpen', 'clahe', 'antialias'],
                        help='预处理方式')
    args = parser.parse_args()

    preprocess_images(args.input_dir, args.output_dir, mode=args.mode)
