import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 配置部分 ---
# 这里填写你的日志文件名
FILE_PATH = './load_metric/no-drive.jsonl' 
# 如果你有另一个对比文件（比如提供的 iso_12_8），可以填在这里，否则设为 None
FILE_PATH_2 = './load_metric/CORE-2.jsonl' 

# 平滑系数 (0~1)，越大越平滑，论文图中通常设为 0.6 到 0.99
SMOOTH_FACTOR = 0.997  

def load_data(file_path):
    """
    读取 jsonl 文件并提取 eval_return, eval_success, eval_length
    """
    data_list = []
    
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return pd.DataFrame()

    with open(file_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                # 我们只关心包含评估数据 (eval_success) 的行
                if 'train_success' in entry:
                    data_list.append({
                        'step': entry['step'],
                        'success': entry['train_success'],
                        'length': entry['train_first_success_step']
                    })
            except json.JSONDecodeError:
                continue
    
    df = pd.DataFrame(data_list)
    return df

def smooth_curve(points, factor=0.9):
    """
    使用指数移动平均 (EMA) 进行平滑，让曲线更好看（类似 TensorBoard）
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_metrics(df_dict):
    """
    绘制对比图
    """
    # 设置画图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 创建一个包含 2 个子图的画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = ['#d62728', '#1f77b4'] # 红色 (LS-Imagine 常用色), 蓝色
    
    for idx, (label, df) in enumerate(df_dict.items()):
        if df.empty:
            continue
            
        steps = df['step']
        success = smooth_curve(df['success'], SMOOTH_FACTOR)
        length = smooth_curve(df['length'], SMOOTH_FACTOR)
        
        color = colors[idx % len(colors)]

        # --- 图 1: Success Rate (成功率) ---
        ax1.plot(steps, success, label=label, color=color, linewidth=2)
        # 如果你想模拟论文中的阴影（标准差），因为只有一个种子(seed)，我们可以用虚线画原始数据
        ax1.plot(steps, df['success'], color=color, alpha=0.2, linewidth=1) 

        # --- 图 2: Steps per episode (每回合步数) ---
        ax2.plot(steps, length, label=label, color=color, linewidth=2)
        ax2.plot(steps, df['length'], color=color, alpha=0.2, linewidth=1)

    # --- 设置图 1 属性 ---
    ax1.set_title('Success Rate', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Environment Steps', fontsize=14)
    ax1.set_ylabel('Success Rate', fontsize=14)
    ax1.set_ylim(-0.05, 1.05) # 成功率通常在 0 到 1 之间
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) # X轴用科学计数法
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # --- 设置图 2 属性 ---
    ax2.set_title('Steps per Episode', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Environment Steps', fontsize=14)
    ax2.set_ylabel('Steps', fontsize=14)
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('ls_imagine_metrics.png', dpi=300) # 保存为高清图片
    print("绘图完成！图片已保存为 ls_imagine_metrics.png")
    plt.show()

# --- 主程序执行 ---
if __name__ == "__main__":
    # 1. 加载数据
    df1 = load_data(FILE_PATH)
    df2 = load_data(FILE_PATH_2) if FILE_PATH_2 else pd.DataFrame()

    data_to_plot = {}
    
    if not df1.empty:
        data_to_plot['LS-Imagine (Seed 0)'] = df1
        print(f"加载了 {len(df1)} 条评估数据 (文件1)")
    
    if not df2.empty:
        data_to_plot['LS-Imagine (ISO)'] = df2
        print(f"加载了 {len(df2)} 条评估数据 (文件2)")

    # 2. 开始画图
    if data_to_plot:
        plot_metrics(data_to_plot)
    else:
        print("没有加载到有效数据，请检查文件名或文件内容。")