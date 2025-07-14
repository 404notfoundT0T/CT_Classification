#preprocess.py
import os
import cv2
import argparse
from PIL import Image
import numpy as np
import pywt

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='data/raw')
parser.add_argument('--output_dir', type=str, default='data/processed')
parser.add_argument('--mode', type=str, default='none',
                    choices=['none', 'spatial', 'frequency','bayesian'],  
                    help='预处理模式: none-原始基准, spatial-空间域去噪, frequency-频域去噪')
parser.add_argument('--data_dir', type=str, default='data/processed_none')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)
args = parser.parse_args()

def resize_with_antialiasing(img, size=(224, 224)):
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(size, Image.LANCZOS)
    return np.array(pil_img)

def bayes_thresh(detail_coeff):
    n = detail_coeff.size
    if n < 20:  
        return 0.1
    sorted_coeff = np.sort(np.abs(detail_coeff.flatten()))
    sigma = np.median(sorted_coeff[-n//20:]) / 0.6745
    return sigma * np.sqrt(2 * np.log(n))

def bayesian_wavelet_denoise(img, wavelet='bior3.3', level=4):
    """
    基于贝叶斯估计的自适应小波去噪
    参数：
        img: 输入灰度图像(0-255)
        wavelet: 使用的小波基，推荐'bior3.3'或'sym4'
        level: 小波分解层数(3-5)
    返回：
        去噪后的图像(0-255)
    """
    img_float = img.astype(np.float32) / 255.0
    
    coeffs = pywt.wavedec2(img_float, wavelet, level=level)
    
    new_coeffs = [coeffs[0]] 
    for i in range(1, len(coeffs)):
        threshold = bayes_thresh(coeffs[i][0])
        
        processed_detail = tuple(
            pywt.threshold(c, value=threshold, mode='soft') 
            for c in coeffs[i]
        )
        new_coeffs.append(processed_detail)
    
    denoised = pywt.waverec2(new_coeffs, wavelet)
    
    denoised = np.clip(denoised, 0, 1) * 255
    return denoised.astype(np.uint8)

def apply_preprocessing(img,mode=args.mode, image_size=(224, 224)):
    img = cv2.resize(img, image_size)
    
    if mode == 'spatial':
        img = cv2.bilateralFilter(
            img, 
            d=5, 
            sigmaColor=25, 
            sigmaSpace=25
        )
        
    if mode == 'frequency':

        coeffs = pywt.wavedec2(img, 'sym4', level=3)
    
        level_thresholds = [15, 10, 5]  
    
        def adaptive_threshold(coeff):
            sigma = np.median(np.abs(coeff)) / 0.6745
            return sigma * np.sqrt(2 * np.log(len(coeff)))
    
        new_coeffs = [coeffs[0]]  
        for i, detail in enumerate(coeffs[1:]):
            fixed_thresh = level_thresholds[i]
            adaptive_thresh = adaptive_threshold(detail[0])
            final_thresh = min(fixed_thresh, adaptive_thresh)
            
            processed_detail = tuple(
                pywt.threshold(d, value=final_thresh, mode='soft')
                for d in detail
            )
            new_coeffs.append(processed_detail)
        
        img = pywt.waverec2(new_coeffs, 'sym4')
        img = np.clip(img, 0, 255).astype(np.uint8)
    
        img = cv2.equalizeHist(img)

    if mode == 'bayesian_wavelet':
        img = bayesian_wavelet_denoise(img)
        img = cv2.equalizeHist(img) 

    return img

def preprocess_images(input_dir, output_dir, mode=args.mode, image_size=(224, 224)):
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

            img_processed = apply_preprocessing(img, mode=mode, image_size=image_size)
            cv2.imwrite(out_path, img_processed)

    print(f"[{mode}] 预处理完成，图像保存至：{output_dir}")


if __name__ == '__main__':
    
    preprocess_images(args.input_dir, args.output_dir, mode=args.mode)
