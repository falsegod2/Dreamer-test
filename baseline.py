import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取数据
data = []
file_path = './load_metric/metrics_seed2_11_26_16_29.jsonl'  # 你的文件路径

try:
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
except FileNotFoundError:
    print(f"错误：找不到文件 {file_path}")
    # 为了演示，生成一些假数据 (如果文件不存在)
    # 实际运行时请确保 metrics.jsonl 存在
    pass

df = pd.DataFrame(data)

# 2. 提取关键指标：Step 和 Train Length
# LS-Imagine 的红色曲线对应的是你的 train_length
if 'train_length' in df.columns:
    plot_df = df[['step', 'train_length']].dropna()
    
    # 3. 数据平滑处理 (模拟论文风格)
    # 论文中的曲线非常平滑，通常是多组 Seed 的平均，或者单 Seed 的大窗口滑动平均
    window_size = 50  # 窗口越大曲线越平滑，阴影带越宽
    
    # 计算滑动平均值 (实线)
    plot_df['mean'] = plot_df['train_length'].rolling(window=window_size, min_periods=1).mean()
    
    # 计算滑动标准差 (阴影范围)
    plot_df['std'] = plot_df['train_length'].rolling(window=window_size, min_periods=1).std()
    
    # 4. 开始画图
    plt.figure(figsize=(8, 6))
    
    # 设置风格，模仿论文背景
    plt.style.use('seaborn-v0_8-whitegrid') # 如果报错，可用 'ggplot' 或删掉这行
    
    # 绘制阴影区域 (Mean ± Std)
    plt.fill_between(
        plot_df['step'], 
        plot_df['mean'] - plot_df['std'], 
        plot_df['mean'] + plot_df['std'], 
        color='red', 
        alpha=0.2, # 透明度，越小越淡
        label='Variance (Std Dev)'
    )
    
    # 绘制主曲线 (Mean)
    plt.plot(
        plot_df['step'], 
        plot_df['mean'], 
        color='red', 
        linewidth=2, 
        label='LS-Imagine (Ours)'
    )

    # 绘制最大步数参考线 (通常是 1000)
    plt.axhline(y=1000, color='gray', linestyle='--', alpha=0.5, label='Max Steps')

    # 5. 设置图表标签和范围
    plt.title('(a) Harvest log in plains', fontsize=14, y=-0.2) # 模仿论文标题位置
    plt.xlabel('Environment steps', fontsize=12)
    plt.ylabel('Steps per episode', fontsize=12)
    
    # 设置 X 轴刻度显示为 1e5 或 1e6 格式
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    # 设置 Y 轴范围 (根据你的最大步数，一般是 0 到 1000+一点点)
    plt.ylim(0, 1050)
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # 保存并显示
    plt.savefig('figure_5_style_plot.png', dpi=300)
    plt.show()
    print("图表已生成：figure_5_style_plot.png")
    
else:
    print("数据中未找到 'train_length' 列，无法绘图。")