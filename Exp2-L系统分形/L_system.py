import matplotlib.pyplot as plt
import math

def apply_rules(axiom, rules, iterations):
    """
    生成L-System字符串
    :param axiom: 初始字符串
    :param rules: 替换规则的字典
    :param iterations: 迭代次数
    :return: 迭代后的字符串
    """
    current = axiom
    for _ in range(iterations):
        next_str = []
        for char in current:
            next_str.append(rules.get(char, char))  
        current = ''.join(next_str)
    return current

def draw_l_system(instructions, angle, step, start_pos=(0,0), start_angle=0, savefile=None, tree_mode=False):
    """
    根据指令字符串绘图
    :param instructions: 生成的指令字符串
    :param angle: 转向角度（度）
    :param step: 步长
    :param start_pos: 起始坐标
    :param start_angle: 初始角度
    :param savefile: 保存文件名
    :param tree_mode: 分形树模式（测试代码兼容参数）
    """
    x, y = start_pos
    current_angle = start_angle  # 初始角度
    stack = []  # 保存状态的栈
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.axis('off')

    for cmd in instructions:
        if cmd in ['F', '0', '1']:  # 处理画线指令
            rad = math.radians(current_angle)
            dx = step * math.cos(rad)
            dy = step * math.sin(rad)
            nx = x + dx
            ny = y + dy
            ax.plot([x, nx], [y, ny], color='black', linewidth=1)
            x, y = nx, ny  # 更新当前位置
        elif cmd == '+':
            current_angle += angle  # 左转
        elif cmd == '-':
            current_angle -= angle  # 右转
        elif cmd == '[':
            stack.append((x, y, current_angle))  # 保存当前状态
            current_angle += angle  # 分形树左转分支
        elif cmd == ']':
            if stack:
                x, y, current_angle = stack.pop()  # 恢复状态，分形树右转

    if savefile:
        plt.savefig(savefile, dpi=200, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    # 示例1：科赫曲线（初始方向向右）
    axiom = "F"
    rules = {"F": "F+F--F+F"}
    iterations = 3
    angle = 60
    step = 5
    instr = apply_rules(axiom, rules, iterations)
    draw_l_system(instr, angle, step, savefile="l_system_koch.png")

    # 示例2：分形树（初始方向向上）
    axiom = "0"
    rules = {"1": "11", "0": "1[0]0"}
    iterations = 5
    angle = 45
    step = 3  # 步长更小以适应多次迭代
    instr = apply_rules(axiom, rules, iterations)
    draw_l_system(instr, angle, step, start_angle=90, savefile="fractal_tree.png")
