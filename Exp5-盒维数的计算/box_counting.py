"""
项目5: 盒计数法估算分形维数
实现盒计数算法计算分形图像的盒维数

任务说明：
1. 实现盒计数算法计算分形图像的盒维数
2. 完成以下函数实现
3. 在main函数中测试你的实现
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_and_binarize_image(image_path, threshold=128):
    """
    加载图像并转换为二值数组
    
    参数：
    image_path : str
        图像文件路径，支持常见格式（PNG/JPG等）
    threshold : int, optional
        二值化阈值（0-255），默认128。大于等于阈值的像素设为1，否则为0
    
    返回：
    binary : ndarray
        二值化的NumPy数组，数据类型为uint8，值域{0, 1}

    """
    # 加载并转换图像
    img = Image.open(image_path).convert('L')  # 转为灰度图像
    img_array = np.array(img)  # 转为numpy数组
    
    # 二值化处理（注意：这里将True(1)设为前景，可根据图像实际情况反转）
    binary = (img_array >= threshold).astype(np.uint8)
    
    # 可选：显示处理后的图像用于调试
    # Image.fromarray(binary * 255).show()  
    return binary

def box_count(binary_image, box_sizes):
    """
    盒计数算法核心实现
    
    参数：
    binary_image : ndarray
        二值图像数组，形状为(H, W)，元素为0或1
    box_sizes : list of int
        要测试的盒子尺寸列表（建议使用递减尺寸）
    
    返回：
    counts : dict
        字典，键为盒子尺寸，值为对应的非空盒子数量

    """
    counts = {}
    h, w = binary_image.shape
    
    for s in box_sizes:
        if s <= 0:  # 跳过无效尺寸
            continue
            
        count = 0
        # 计算纵向和横向的网格数（向上取整）
        rows = (h + s - 1) // s  
        cols = (w + s - 1) // s
        
        for i in range(rows):
            # 计算当前网格纵向范围
            r_start = i * s
            r_end = min(r_start + s, h)  # 防止越界
            
            for j in range(cols):
                # 计算当前网格横向范围
                c_start = j * s
                c_end = min(c_start + s, w)
                
                # 提取当前盒子区域
                box = binary_image[r_start:r_end, c_start:c_end]
                
                # 检测是否包含至少一个前景像素（值为1）
                if np.any(box == 1):
                    count += 1
        
        counts[s] = count
    
    return counts

def calculate_fractal_dimension(binary_image, min_box_size=1, max_box_size=None, num_sizes=10):
    """
    计算分形维数及相关数据
    
    参数：
    binary_image : ndarray
        二值图像数组
    min_box_size : int, optional
        最小盒子尺寸，默认1（最小分辨率）
    max_box_size : int, optional
        最大盒子尺寸，默认取图像短边的一半
    num_sizes : int, optional
        生成的盒子尺寸数量，默认10
    
    返回：
    D : float
        估算的分形维数
    results : tuple
        (epsilons, N_epsilons, slope, intercept)
        - epsilons: 实际使用的盒子尺寸数组
        - N_epsilons: 对应的盒子计数数组
        - slope: 拟合直线斜率
        - intercept: 拟合直线截距

    """
    h, w = binary_image.shape
    
    # 设置最大盒子尺寸
    if max_box_size is None:
        max_box_size = min(h, w) // 2  # 防止盒子尺寸过大导致网格划分过粗
    max_box_size = max(max_box_size, min_box_size)  # 确保max >= min
    
    # 生成等比数列（几何级数），确保尺寸逐渐减小
    epsilons = np.geomspace(
        start=max_box_size,  # 从最大尺寸开始
        stop=min_box_size,  # 到最小尺寸结束
        num=num_sizes
    ).astype(int)
    
    # 去重并排序（从大到小）
    epsilons = np.unique(epsilons)[::-1]  
    
    # 过滤超出范围的尺寸
    epsilons = epsilons[(epsilons >= min_box_size) & (epsilons <= max_box_size)]
    
    # 执行盒计数
    counts_dict = box_count(binary_image, epsilons)
    epsilons = np.array(list(counts_dict.keys()))
    N_epsilons = np.array(list(counts_dict.values()))
    
    # 数据有效性检查（至少需要2个数据点）
    valid = N_epsilons > 0
    epsilons = epsilons[valid]
    N_epsilons = N_epsilons[valid]
    
    if len(epsilons) < 2:
        raise ValueError("有效数据点不足，无法进行线性回归")
    
    # 对数变换
    log_eps = np.log(epsilons)
    log_N = np.log(N_epsilons)
    
    # 线性回归（一阶多项式拟合）
    slope, intercept = np.polyfit(log_eps, log_N, 1)
    D = -slope  # 分形维数
    
    return D, (epsilons, N_epsilons, slope, intercept)

def plot_log_log(epsilons, N_epsilons, slope, intercept, save_path=None):
    """
    绘制log-log图用于可视化验证
    
    参数：
    epsilons : array_like
        盒子尺寸数组
    N_epsilons : array_like
        对应的盒子计数数组
    slope : float
        拟合直线斜率
    intercept : float
        拟合直线截距
    save_path : str, optional
        图像保存路径，如未指定则显示在窗口中
    
    图像说明：
    - 散点：实际数据点(log(ε), log(N(ε)))
    - 红线：拟合直线，标注计算得到的分形维数
    """
    plt.figure(figsize=(8, 6))
    
    # 计算坐标
    log_eps = np.log(epsilons)
    log_N = np.log(N_epsilons)
    fit_line = slope * log_eps + intercept
    
    # 绘制散点图
    plt.scatter(
        log_eps, log_N,
        color='blue', marker='o',
        label='Data points',
        zorder=10  # 确保散点在前景
    )
    
    # 绘制拟合直线
    plt.plot(
        log_eps, fit_line,
        color='red', linestyle='--',
        label=f'Fit line (D = {-slope:.3f})'
    )
    
    # 图例和标签
    plt.xlabel('log(ε)', fontsize=12)
    plt.ylabel('log(N(ε))', fontsize=12)
    plt.title('Box Counting Method - log-log plot', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # 使用说明 -------------------------------------------------
    # 1. 将box.png放在项目images目录下
    # 2. 根据需要调整阈值参数
    
    # 配置参数
    IMAGE_PATHS = "box.png"
    binary_img = load_and_binarize_image(IMAGE_PATHS)
    
    # 2. 计算分形维数
    D, (epsilons, N_epsilons, slope, intercept) = calculate_fractal_dimension(binary_img)
    
    # 3. 输出结果
    print("盒计数结果:")
    for eps, N in zip(epsilons, N_epsilons):
        print(f"ε = {eps:4d}, N(ε) = {N:6d}, log(ε) = {np.log(eps):.3f}, log(N) = {np.log(N):.3f}")
    
    print(f"\n拟合斜率: {slope:.5f}")
    print(f"估算的盒维数 D = {D:.5f}")
    
    # 4. 绘制log-log图
    plot_log_log(epsilons, N_epsilons, slope, intercept, "log_log_plot.png")

