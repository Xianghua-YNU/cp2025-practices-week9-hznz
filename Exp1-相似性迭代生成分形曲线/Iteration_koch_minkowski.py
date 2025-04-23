import numpy as np
import matplotlib.pyplot as plt

def koch_generator(u, level):
    """
    递归/迭代生成科赫曲线的点序列。

    参数:
        u: 初始线段的端点数组（复数表示）
        level: 迭代层数

    返回:
        numpy.ndarray: 生成的所有点（复数数组）
    """
    # TODO: 实现科赫曲线生成算法
    points = np.array([0, 1j]) # 初始竖直线段
    
    if level <= 0:
        return points
        
    theta = np.pi/3 # 旋转角度
    for _ in range(level):
        new_points = []
        for i in range(len(points)-1):
            start = points[i]
            end = points[i+1]
            segment = end - start
            
            # 科赫曲线的生成规则：将线段分为4段，中间插入一个等边三角形的两边
            p0 = start
            p1 = start + segment / 3
            p2 = p1 + (segment / 3) * np.exp(1j * np.pi/3)
            p3 = start + 2 * segment / 3
            p4 = end
            
            new_points.extend([p0, p1, p2, p3,p4])
        
        points = np.array(new_points)
    return points

def minkowski_generator(u, level):
    """
    递归/迭代生成闵可夫斯基香肠曲线的点序列。

    参数:
        u: 初始线段的端点数组（复数表示）
        level: 迭代层数

    返回:
        numpy.ndarray: 生成的所有点（复数数组）
    """
    # TODO: 实现闵可夫斯基香肠曲线生成算法
    points = np.array([0, 1]) # 初始水平线段
    
    theta = np.pi/2 # 旋转角度
    for _ in range(level):
        new_points = []
        for i in range(len(points)-1):
            start = points[i]
            end = points[i+1]
            segment = end - start
            
            # 闵可夫斯基香肠曲线的生成规则：将线段替换为8段特定形状
            p0 = start
            p1 = start + segment / 4
            p2 = p1 + (segment / 4) * 1j
            p3 = p2 + segment / 4
            p4 = p3 + (segment / 4) * (-1j)
            p5 = p4 + segment / 4
            p6 = p5 + (segment / 4) * (-1j)
            p7 = p6 + segment / 4
            p8 = p7 + (segment / 4) * 1j
            p9 = end
            
            new_points.extend([p0, p1, p2, p3, p4, p5, p6, p7,p8,p9])
        
        points = np.array(new_points)
    return points

if __name__ == "__main__":
    # 初始线段
    init_u = np.array([0, 1])

    # 绘制不同层级的科赫曲线
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(4):
        # TODO: 调用koch_generator生成点
        koch_points = None  # 替换为实际生成的点
        axs[i//2, i%2].plot(
            np.real(koch_points), np.imag(koch_points), 'k-', lw=1
        )
        axs[i//2, i%2].set_title(f"Koch Curve Level {i+1}")
        axs[i//2, i%2].axis('equal')
        axs[i//2, i%2].axis('off')
    plt.tight_layout()
    plt.show()

    # 绘制不同层级的闵可夫斯基香肠曲线
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(4):
        # TODO: 调用minkowski_generator生成点
        minkowski_points = None  # 替换为实际生成的点
        axs[i//2, i%2].plot(
            np.real(minkowski_points), np.imag(minkowski_points), 'k-', lw=1
        )
        axs[i//2, i%2].set_title(f"Minkowski Sausage Level {i+1}")
        axs[i//2, i%2].axis('equal')
        axs[i//2, i%2].axis('off')
    plt.tight_layout()
    plt.show()
