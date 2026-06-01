import os
import cv2
import numpy as np

def remove_large_shadows_keep_edge_shadows(L_channel, kernel_sigma):
    """
    去除大块阴影，保留边界线阴影。
    L_channel: 亮度通道 (0-255, uint8)
    kernel_sigma: 高斯模糊的标准差，需大于大块阴影的尺寸（像素）。
                   例如阴影直径约100像素，则sigma取50左右。
    返回: 校正后的亮度通道
    """
    # 1. 提取低频背景（包含大块阴影）
    background = cv2.GaussianBlur(L_channel, (0, 0), sigmaX=kernel_sigma, sigmaY=kernel_sigma)
    
    # 2. 减法去除大块阴影（保留高频细节）
    L_float = L_channel.astype(np.float32)
    bg_float = background.astype(np.float32)
    corrected = L_float - bg_float
    
    # 3. 偏移并归一化到 0-255 范围（保留对比度）
    min_val = np.min(corrected)
    max_val = np.max(corrected)
    if max_val > min_val:
        corrected = 255 * (corrected - min_val) / (max_val - min_val)
    else:
        corrected = np.zeros_like(corrected)
    
    return corrected.astype(np.uint8)

def preprocess_keep_edge_shadows(image_path, output_path):
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取: {image_path}")
        return

    # 2. 转换到 LAB，分离 L 通道
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # 3. 去除大块阴影，保留边界阴影（多次不同尺度处理）
    L_corrected = remove_large_shadows_keep_edge_shadows(L, kernel_sigma=101)
    L_corrected = remove_large_shadows_keep_edge_shadows(L_corrected, kernel_sigma=51)
    L_corrected = remove_large_shadows_keep_edge_shadows(L_corrected, kernel_sigma=12)
    #做轻微高斯模糊
    L_corrected = cv2.GaussianBlur(L_corrected, (0, 0), sigmaX=5, sigmaY=5)
    # 4. CLAHE 增强局部对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_eq = clahe.apply(L_corrected)

    # 5. 边缘增强（非锐化掩膜 + 拉普拉斯）
    blurred = cv2.GaussianBlur(L_eq, (0, 0), sigmaX=3, sigmaY=3)
    unsharp = cv2.addWeighted(L_eq, 1.5, blurred, -0.5, 0)
    laplacian = cv2.Laplacian(unsharp, cv2.CV_16S, ksize=3)
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    L_enhanced = cv2.addWeighted(unsharp, 1.0, laplacian_abs, 0.3, 0)
    L_enhanced = remove_large_shadows_keep_edge_shadows(L_enhanced, kernel_sigma=100)

    # 6. 合并 LAB 并转回 BGR
    lab_out = cv2.merge([L_enhanced, A, B])
    img_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    # 7. 对最终彩色图提取灰度图并增强对比度
    img_out1 = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
    
    # canny算子提取边缘
    edges = cv2.Canny(img_out1, 120, 230)
    #  对edges做颜色反转
    edges = cv2.bitwise_not(edges)

    # 8. 修正：将单通道灰度图转为 3 通道，再加权融合
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    img_out = cv2.addWeighted(img_out, 1, edges_3ch, 0.3, 0)
    #img_out = cv2.GaussianBlur(img_out, (0, 0), sigmaX=1, sigmaY=1)  # 轻微模糊融合结果
    # 

    # 9. 保存结果
    cv2.imwrite(output_path, img_out)

# 批量处理
output_dir = "project_image_preprocess"
os.makedirs(output_dir, exist_ok=True)
input_dir = "project_image"
valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

for filename in os.listdir(input_dir):
    if filename.lower().endswith(valid_exts):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        preprocess_keep_edge_shadows(input_path, output_path)
        print(f"已处理: {filename}")