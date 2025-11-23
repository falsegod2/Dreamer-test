import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns

#sns.set(style="whitegrid")

# 读取原始训练数据（LS-Imagine）
file_path = './load_metric/metrics.jsonl'
data = pd.read_json(file_path, lines=True)
data_clean = data.dropna(subset=['step', 'train_success'])

# 每 10000 步分组
data_clean['step_group'] = (data_clean['step'] // 10000) * 10000
grouped = data_clean.groupby('step_group')['train_success'].mean().reset_index()

# 平滑 LS-Imagine
grouped['smooth_success'] = grouped['train_success'].rolling(window=3, min_periods=1).mean()

steps = grouped['step_group'].values



# ==== 绘图 ====
plt.figure(figsize=(10, 6))
plt.plot(grouped['step_group'], grouped['smooth_success'],
         label='LS-Imagine (ours)', color='green', linewidth=2, zorder=3)


plt.xlabel('Training Steps')
plt.ylabel('Success Rate')
plt.ylim(0, 1.05)
plt.title('Training Success Rate Comparison: LS-Imagine vs Baselines (Cut Tree Task)')
plt.legend()
plt.tight_layout()
plt.show()
