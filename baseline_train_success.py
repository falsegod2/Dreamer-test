import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_log_data(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip().startswith('{'):
                    continue
                try:
                    entry = json.loads(line)
                    # 修改：读取 train_success 而不是 eval_success
                    if 'train_success' in entry and 'step' in entry:
                        data.append({
                            'step': entry['step'],
                            'success_rate': entry['train_success']
                        })
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"警告: 找不到文件 {file_path}")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values('step')
    return df

def plot_robust_multi_seed_train(files, task_name="MineDojo Training Success Rate (Multi-Seed)"):
    # 1. 读取所有种子数据
    dfs = []
    for f in files:
        df = read_log_data(f)
        if not df.empty:
            # 训练数据点通常更多且更嘈杂，适当增大平滑窗口
            df['success_rate'] = df['success_rate'].rolling(window=100, min_periods=1).mean()
            dfs.append(df)

    if not dfs:
        print("没有找到有效数据。")
        return

    # 2. 创建公共时间轴 (Common Grid)
    min_step = min(df['step'].min() for df in dfs)
    max_step = max(df['step'].max() for df in dfs)
    
    common_steps = np.linspace(min_step, max_step, 1000)
    
    # 3. 插值 (Interpolation)
    interpolated_results = []
    for df in dfs:
        df = df.drop_duplicates('step')
        interp_vals = np.interp(common_steps, df['step'], df['success_rate'])
        interpolated_results.append(interp_vals)
    
    results_array = np.array(interpolated_results)
    
    # 4. 计算均值和标准差
    mean_success = np.mean(results_array, axis=0)
    std_success = np.std(results_array, axis=0)
    
    # 5. 绘图
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('default')
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.5

    fig, ax = plt.subplots(figsize=(10, 6))
    
    line_color = "#D62728" # 红色
    fill_color = "#D62728"
    
    # 绘制均值线
    ax.plot(common_steps, mean_success, linewidth=2.5, color=line_color, label='LS-Imagine (Train Mean)')
    
    # 绘制阴影带 (Mean ± Std)
    ax.fill_between(common_steps, 
                    (mean_success - std_success).clip(0, 1), 
                    (mean_success + std_success).clip(0, 1), 
                    color=fill_color, alpha=0.15, linewidth=0, label='Standard Deviation')
    
    # 格式化
    ax.set_xlim(min_step, max_step)
    ax.set_ylim(0, 1.05)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    ax.set_xlabel("Environment steps", fontsize=14, fontweight='bold')
    ax.set_ylabel("Training Success Rate", fontsize=14, fontweight='bold')
    ax.set_title(task_name, fontsize=16, fontweight='bold')
    
    ax.legend(loc='upper left', fontsize=12, frameon=True, framealpha=0.9, edgecolor='white')
    plt.tight_layout()
    
    save_path = 'robust_multi_seed_train_plot.png'
    plt.savefig(save_path, dpi=300)
    print(f"训练成功率鲁棒图表已生成并保存为: {save_path}")
    plt.show()

if __name__ == "__main__":
    seed_files = [
        './load_metric/metrics_seed0.jsonl',
        './load_metric/metrics_seed1_11_26_16_29.jsonl',
        './load_metric/metrics_seed2_11_27_7_37.jsonl'
    ]
    plot_robust_multi_seed_train(seed_files)