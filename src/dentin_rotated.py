import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from sklearn.linear_model import LinearRegression
from math import atan, degrees

# 改進的對比度增強函數
def enhance_contrast(img, clipLimit=2.0, tileGridSize=(8, 8)):
    """
    增強對比度函數。
    :param img: 輸入圖像（灰度圖像）。
    :param clipLimit: CLAHE 的對比度限制參數。
    :param tileGridSize: CLAHE 的局部網格大小。
    :return: 增強對比度後的圖像。
    """
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img)

# 圖片預處理：增強對比度 + 高斯模糊 + 均值濾波
def preprocess_image(img):
    # 第一步：增強對比度
    img = enhance_contrast(img, clipLimit=4.0, tileGridSize=(4, 4))
    
    # 第二步：高斯模糊
    kernel_size = max(3, min(11, img.shape[1] // 50 | 1))  # 動態設置內核大小
    img_blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 1)
    
    # 第三步：均值濾波
    mean_kernel_size = max(3, min(7, img.shape[1] // 100 | 1))  # 動態設置內核大小
    img_mean = cv2.blur(img_blur, (mean_kernel_size, mean_kernel_size))
    return img_mean

# 垂直積分投影與峰值檢測
def integral_projection(img, method='gaussian', peak_params=None):
    projection = np.sum(img, axis=0).astype(np.float32)
    if peak_params is None:
        peak_params = {'distance': img.shape[1] // 20, 'prominence': 0.05 * np.max(projection)}
    
    if method == 'gaussian':
        smoothed_projection = cv2.GaussianBlur(projection.reshape(1, -1), (1, 21), 0).flatten()
    elif method == 'savgol':
        smoothed_projection = savgol_filter(projection, 61, 3)
    else:
        smoothed_projection = projection

    peaks, _ = find_peaks(-smoothed_projection, **peak_params)
    return peaks, smoothed_projection

def assign_sets(peaks, rows, cols, img):
    sets = {i: [] for i in range(len(peaks))}
    sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

    sobel_mean = np.mean(np.abs(sobel))
    sobel_threshold = max(20, sobel_mean * 1.2)

    for i, peak in enumerate(peaks):
        sets[i].append((rows // 6, peak))

    for row in range(rows // 6 + 1, 2 * rows // 3 + 1):
        row_values = img[row, :]
        sobel_values = np.abs(sobel[row, :]) * 1.5
        for k in sets.keys():
            last_col = sets[k][-1][1]
            boundary_points = [
                col for col in range(max(0, last_col - 20), min(cols, last_col + 20))
                if sobel_values[col] > sobel_threshold and row_values[col] < np.mean(row_values)
            ]
            if boundary_points:
                nearest_point = min(boundary_points, key=lambda x: abs(x - last_col))
                sets[k].append((row, nearest_point))
    return sets

def linear_regression_on_sets(sets):
    angles = []
    for points in sets.values():
        y = np.array([p[0] for p in points]).reshape(-1, 1)
        x = np.array([p[1] for p in points]).reshape(-1, 1)
        reg = LinearRegression().fit(x, y)
        slope = reg.coef_[0][0]
        angle = degrees(atan(slope))
        angles.append(angle)
    
    mean_angle = np.mean(angles)
    limited_angle = max(-15, min(15, mean_angle))  # 放寬角度範圍
    return limited_angle

# 旋轉校正圖片
def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)  # 計算旋轉矩陣
    rotated_img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rotated_img

# 投影結果可視化（用綠色線表示峰值）
def visualize_projection(peaks, projection, img_height):
    plt.figure(figsize=(10, 6))
    plt.plot(projection, label='Projection', color='gray')
    plt.plot(peaks, projection[peaks], 'go', label='Peaks', markersize=8)
    for peak in peaks:
        plt.axvline(x=peak, color='Green', linestyle='--', linewidth=0.8)  # 綠色虛線
    plt.title('Projection with Peaks')
    plt.xlabel('Column Index')
    plt.ylabel('Sum of Gray Levels')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_segmentation(img, sets, peaks):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 將圖片轉換為RGB顏色圖

    # 假設 peaks 包含了所有綠線的位置，從左到右排序
    peaks.sort()

    # 計算紅線（線性回歸直線）的位置
    for points in sets.values():
        x_vals = np.array([p[1] for p in points]).reshape(-1, 1)  # x值為列索引
        y_vals = np.array([p[0] for p in points])  # y值為行索引

        # 線性回歸
        reg = LinearRegression().fit(x_vals, y_vals)

        # 根據回歸結果畫出紅色直線
        y_pred = reg.predict(x_vals)
        for i in range(1, len(x_vals)):
            pt1 = (x_vals[i-1][0], int(y_pred[i-1]))
            pt2 = (x_vals[i][0], int(y_pred[i]))
            cv2.line(img, pt1, pt2, (255, 0, 255), 5)  # 紅色直線

        # 計算整條直線的長度
        length = np.sqrt((x_vals[-1][0] - x_vals[0][0])**2 + (y_pred[-1] - y_pred[0])**2)
        
        # 將長度標註到圖像上（取小數點後兩位）
        mid_point = ((x_vals[0][0] + x_vals[-1][0]) // 2, int((y_pred[0] + y_pred[-1]) // 2))
        cv2.putText(img, f"{length:.2f}", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 畫出綠線
    for peak in peaks:
        cv2.line(img, (peak, 0), (peak, img.shape[0]), (0, 255, 0), 5)  # 綠線
    return img

# 顯示過程中的圖片
def display_process_images(img, projection, peaks, sets, rotated_img, angles):
    img_with_lines = visualize_segmentation(img, sets, peaks)
    
    plt.figure(figsize=(12, 12))
    
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.plot(projection, label='Projection', color='gray')
    plt.plot(peaks, projection[peaks], 'go', label='Peaks', markersize=8)
    for peak in peaks:
        plt.axvline(x=peak, color='green', linestyle='--', linewidth=0.8)
    plt.title("Integral Projection")
    plt.xlabel('Column Index')
    plt.ylabel('Sum of Gray Levels')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.imshow(img_with_lines)
    plt.title("Regression Line with Peaks")
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(rotated_img, cmap='gray')
    plt.title(f"Rotated Image\nAngle: {angles:.2f} degrees")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 處理資料夾內的所有圖片
def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    images = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Error loading image {img_name}")
            continue
        
        preprocessed_img = preprocess_image(img)

        peaks, projection = integral_projection(preprocessed_img)
        sets = assign_sets(peaks, img.shape[0], img.shape[1], preprocessed_img)
        rotation_angle = linear_regression_on_sets(sets)
        rotated_img = rotate_image(preprocessed_img, rotation_angle)

        display_process_images(preprocessed_img, projection, peaks, sets, rotated_img, rotation_angle)

        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, rotated_img)

# 主程式
if __name__ == "__main__":
    input_dir = 'data/pic/5_pic'
    output_dir = 'data/pic/rotated'
    process_images(input_dir, output_dir)