import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np

# === 1. 您的完整训练数据 ===
raw_log_data = """
[2500] eval_return 0.0 / eval_length 1000.3 / eval_episodes 3.0 / eval_success 0.0 / eval_first_success_step 1000.0
[2500] model_loss 749.4 ...
[4388] dataset_size 4388.0 / train_return 0.0 / train_length 1991.0 / train_episodes 3.0 / train_success 0.0
[6285] dataset_size 6285.0 / train_return 0.0 / train_length 1897.0 / train_episodes 4.0 / train_success 0.0
[7294] dataset_size 7294.0 / train_return 0.0 / train_length 1009.0 / train_episodes 5.0 / train_success 0.0
[9205] dataset_size 9205.0 / train_return 0.0 / train_length 1911.0 / train_episodes 6.0 / train_success 0.0
[10808] dataset_size 10808.0 / train_return 0.0 / train_length 1603.0 / train_episodes 7.0 / train_success 0.0
[12500] eval_return 0.0 / eval_length 1000.3 / eval_episodes 3.0 / eval_success 0.0
[12789] dataset_size 12789.0 / train_return 0.0 / train_length 1981.0 / train_episodes 8.0 / train_success 0.0
[14776] dataset_size 14776.0 / train_return 0.0 / train_length 1987.0 / train_episodes 9.0 / train_success 0.0
[16409] dataset_size 16409.0 / train_return 0.0 / train_length 1633.0 / train_episodes 10.0 / train_success 0.0
[17752] dataset_size 17752.0 / train_return 0.0 / train_length 1343.0 / train_episodes 11.0 / train_success 0.0
[18807] dataset_size 18807.0 / train_return 0.0 / train_length 1055.0 / train_episodes 12.0 / train_success 0.0
[19848] dataset_size 19848.0 / train_return 0.0 / train_length 1041.0 / train_episodes 13.0 / train_success 0.0
[21708] dataset_size 21708.0 / train_return 0.0 / train_length 1860.0 / train_episodes 14.0 / train_success 0.0
[22500] eval_return 0.0 / eval_length 1000.3 / eval_episodes 3.0 / eval_success 0.0
[23492] dataset_size 23492.0 / train_return 0.0 / train_length 1784.0 / train_episodes 15.0 / train_success 0.0
[24849] dataset_size 24849.0 / train_return 0.0 / train_length 1357.0 / train_episodes 16.0 / train_success 0.0
[26049] dataset_size 26049.0 / train_return 0.0 / train_length 1200.0 / train_episodes 17.0 / train_success 0.0
[27744] dataset_size 27744.0 / train_return 0.0 / train_length 1695.0 / train_episodes 18.0 / train_success 0.0
[27948] dataset_size 27948.0 / train_return 2.0 / train_length 204.0 / train_episodes 19.0 / train_success 1.0
[29331] dataset_size 29331.0 / train_return 0.0 / train_length 1383.0 / train_episodes 20.0 / train_success 0.0
[29523] dataset_size 29523.0 / train_return 2.0 / train_length 192.0 / train_episodes 21.0 / train_success 1.0
[31414] dataset_size 31414.0 / train_return 0.0 / train_length 1891.0 / train_episodes 22.0 / train_success 0.0
[32500] eval_return 0.0 / eval_length 1000.3 / eval_episodes 3.0 / eval_success 0.0
[32722] dataset_size 32722.0 / train_return 0.0 / train_length 1308.0 / train_episodes 23.0 / train_success 0.0
[34642] dataset_size 34642.0 / train_return 0.0 / train_length 1920.0 / train_episodes 24.0 / train_success 0.0
[36450] dataset_size 36450.0 / train_return 0.0 / train_length 1808.0 / train_episodes 25.0 / train_success 0.0
[37450] dataset_size 37450.0 / train_return 0.0 / train_length 1000.0 / train_episodes 26.0 / train_success 0.0
[38450] dataset_size 38450.0 / train_return 0.0 / train_length 1000.0 / train_episodes 27.0 / train_success 0.0
[40116] dataset_size 40116.0 / train_return 0.0 / train_length 1666.0 / train_episodes 28.0 / train_success 0.0
[42101] dataset_size 42101.0 / train_return 0.0 / train_length 1985.0 / train_episodes 29.0 / train_success 0.0
[42500] eval_return 0.0 / eval_length 1000.3 / eval_episodes 3.0 / eval_success 0.0
[43933] dataset_size 43933.0 / train_return 0.0 / train_length 1832.0 / train_episodes 30.0 / train_success 0.0
[45796] dataset_size 45796.0 / train_return 0.0 / train_length 1863.0 / train_episodes 31.0 / train_success 0.0
[46985] dataset_size 46985.0 / train_return 0.0 / train_length 1189.0 / train_episodes 32.0 / train_success 0.0
[48056] dataset_size 48056.0 / train_return 0.0 / train_length 1071.0 / train_episodes 33.0 / train_success 0.0
[49563] dataset_size 49563.0 / train_return 0.0 / train_length 1507.0 / train_episodes 34.0 / train_success 0.0
[50874] dataset_size 50874.0 / train_return 0.0 / train_length 1311.0 / train_episodes 35.0 / train_success 0.0
[52500] eval_return 0.0 / eval_length 1000.3 / eval_episodes 3.0 / eval_success 0.0
[52588] dataset_size 52588.0 / train_return 0.0 / train_length 1714.0 / train_episodes 36.0 / train_success 0.0
[54289] dataset_size 54289.0 / train_return 0.0 / train_length 1701.0 / train_episodes 37.0 / train_success 0.0
[55414] dataset_size 55414.0 / train_return 0.0 / train_length 1125.0 / train_episodes 38.0 / train_success 0.0
[56941] dataset_size 56941.0 / train_return 0.0 / train_length 1527.0 / train_episodes 39.0 / train_success 0.0
[58220] dataset_size 58220.0 / train_return 0.0 / train_length 1279.0 / train_episodes 40.0 / train_success 0.0
[59911] dataset_size 59911.0 / train_return 0.0 / train_length 1691.0 / train_episodes 41.0 / train_success 0.0
[60989] dataset_size 60989.0 / train_return 0.0 / train_length 1078.0 / train_episodes 42.0 / train_success 0.0
[62494] dataset_size 62494.0 / train_return 0.0 / train_length 1505.0 / train_episodes 43.0 / train_success 0.0
[62500] eval_return 0.0 / eval_length 1000.3 / eval_episodes 3.0 / eval_success 0.0
[64187] dataset_size 64187.0 / train_return 0.0 / train_length 1693.0 / train_episodes 44.0 / train_success 0.0
[65307] dataset_size 65307.0 / train_return 0.0 / train_length 1120.0 / train_episodes 45.0 / train_success 0.0
[66873] dataset_size 66873.0 / train_return 0.0 / train_length 1566.0 / train_episodes 46.0 / train_success 0.0
[68383] dataset_size 68383.0 / train_return 0.0 / train_length 1510.0 / train_episodes 47.0 / train_success 0.0
[70107] dataset_size 70107.0 / train_return 0.0 / train_length 1724.0 / train_episodes 48.0 / train_success 0.0
[70868] dataset_size 70868.0 / train_return 2.0 / train_length 761.0 / train_episodes 49.0 / train_success 1.0
[71934] dataset_size 71934.0 / train_return 0.0 / train_length 1066.0 / train_episodes 50.0 / train_success 0.0
[72500] eval_return 0.0 / eval_length 1000.3 / eval_episodes 3.0 / eval_success 0.0
[73051] dataset_size 73051.0 / train_return 0.0 / train_length 1117.0 / train_episodes 51.0 / train_success 0.0
[74052] dataset_size 74052.0 / train_return 0.0 / train_length 1001.0 / train_episodes 52.0 / train_success 0.0
[76051] dataset_size 76051.0 / train_return 0.0 / train_length 1999.0 / train_episodes 53.0 / train_success 0.0
[77864] dataset_size 77864.0 / train_return 0.0 / train_length 1813.0 / train_episodes 54.0 / train_success 0.0
[79756] dataset_size 79756.0 / train_return 0.0 / train_length 1892.0 / train_episodes 55.0 / train_success 0.0
[80915] dataset_size 80915.0 / train_return 0.0 / train_length 1159.0 / train_episodes 56.0 / train_success 0.0
[82270] dataset_size 82270.0 / train_return 0.0 / train_length 1355.0 / train_episodes 57.0 / train_success 0.0
[82500] eval_return 0.7 / eval_length 759.0 / eval_episodes 3.0 / eval_success 0.3
[83986] dataset_size 83986.0 / train_return 0.0 / train_length 1716.0 / train_episodes 58.0 / train_success 0.0
[84986] dataset_size 84986.0 / train_return 0.0 / train_length 1000.0 / train_episodes 59.0 / train_success 0.0
[85464] dataset_size 85464.0 / train_return 2.0 / train_length 478.0 / train_episodes 60.0 / train_success 1.0
[86786] dataset_size 86786.0 / train_return 0.0 / train_length 1322.0 / train_episodes 61.0 / train_success 0.0
[87071] dataset_size 87071.0 / train_return 4.0 / train_length 285.0 / train_episodes 62.0 / train_success 1.0
[87477] dataset_size 87477.0 / train_return 2.0 / train_length 406.0 / train_episodes 63.0 / train_success 1.0
[88477] dataset_size 88477.0 / train_return 0.0 / train_length 1000.0 / train_episodes 64.0 / train_success 0.0
[89450] dataset_size 89450.0 / train_return 2.0 / train_length 973.0 / train_episodes 65.0 / train_success 1.0
[90007] dataset_size 90007.0 / train_return 2.0 / train_length 557.0 / train_episodes 66.0 / train_success 1.0
[91392] dataset_size 91392.0 / train_return 0.0 / train_length 1385.0 / train_episodes 67.0 / train_success 0.0
[92153] dataset_size 92153.0 / train_return 2.0 / train_length 761.0 / train_episodes 68.0 / train_success 1.0
[92500] eval_return 0.0 / eval_length 1000.3 / eval_episodes 3.0 / eval_success 0.0
[93401] dataset_size 93401.0 / train_return 0.0 / train_length 1248.0 / train_episodes 69.0 / train_success 0.0
[94959] dataset_size 94959.0 / train_return 0.0 / train_length 1558.0 / train_episodes 70.0 / train_success 0.0
[96918] dataset_size 96918.0 / train_return 0.0 / train_length 1959.0 / train_episodes 71.0 / train_success 0.0
[97195] dataset_size 97195.0 / train_return 2.0 / train_length 277.0 / train_episodes 72.0 / train_success 1.0
[99030] dataset_size 99030.0 / train_return 0.0 / train_length 1835.0 / train_episodes 73.0 / train_success 0.0
[99203] dataset_size 99203.0 / train_return 2.0 / train_length 173.0 / train_episodes 74.0 / train_success 1.0
[100214] dataset_size 100214.0 / train_return 0.0 / train_length 1011.0 / train_episodes 75.0 / train_success 0.0
[102165] dataset_size 102165.0 / train_return 0.0 / train_length 1951.0 / train_episodes 76.0 / train_success 0.0
[102500] eval_return 0.0 / eval_length 1000.3 / eval_episodes 3.0 / eval_success 0.0
[103355] dataset_size 103355.0 / train_return 0.0 / train_length 1190.0 / train_episodes 77.0 / train_success 0.0
[105204] dataset_size 105204.0 / train_return 0.0 / train_length 1849.0 / train_episodes 78.0 / train_success 0.0
[106204] dataset_size 106204.0 / train_return 0.0 / train_length 1000.0 / train_episodes 79.0 / train_success 0.0
[108200] dataset_size 108200.0 / train_return 0.0 / train_length 1996.0 / train_episodes 80.0 / train_success 0.0
[110199] dataset_size 110199.0 / train_return 0.0 / train_length 1999.0 / train_episodes 81.0 / train_success 0.0
[111251] dataset_size 111251.0 / train_return 0.0 / train_length 1052.0 / train_episodes 82.0 / train_success 0.0
[112500] eval_return 0.0 / eval_length 1000.3 / eval_episodes 3.0 / eval_success 0.0
[112829] dataset_size 112829.0 / train_return 0.0 / train_length 1578.0 / train_episodes 83.0 / train_success 0.0
[114127] dataset_size 114127.0 / train_return 0.0 / train_length 1298.0 / train_episodes 84.0 / train_success 0.0
[114785] dataset_size 114785.0 / train_return 2.0 / train_length 658.0 / train_episodes 85.0 / train_success 1.0
[115785] dataset_size 115785.0 / train_return 0.0 / train_length 1000.0 / train_episodes 86.0 / train_success 0.0
[116077] dataset_size 116077.0 / train_return 2.0 / train_length 292.0 / train_episodes 87.0 / train_success 1.0
[117077] dataset_size 117077.0 / train_return 0.0 / train_length 1000.0 / train_episodes 88.0 / train_success 0.0
[117763] dataset_size 117763.0 / train_return 2.0 / train_length 686.0 / train_episodes 89.0 / train_success 1.0
[118739] dataset_size 118739.0 / train_return 2.0 / train_length 976.0 / train_episodes 90.0 / train_success 1.0
[119304] dataset_size 119304.0 / train_return 4.0 / train_length 565.0 / train_episodes 91.0 / train_success 1.0
[119976] dataset_size 119976.0 / train_return 2.0 / train_length 672.0 / train_episodes 92.0 / train_success 1.0
[120460] dataset_size 120460.0 / train_return 2.0 / train_length 484.0 / train_episodes 93.0 / train_success 1.0
[121831] dataset_size 121831.0 / train_return 0.0 / train_length 1371.0 / train_episodes 94.0 / train_success 0.0
[122500] eval_return 0.0 / eval_length 1000.0 / eval_episodes 3.0 / eval_success 0.0
[123448] dataset_size 123448.0 / train_return 0.0 / train_length 1617.0 / train_episodes 95.0 / train_success 0.0
[125420] dataset_size 125420.0 / train_return 0.0 / train_length 1972.0 / train_episodes 96.0 / train_success 0.0
[126420] dataset_size 126420.0 / train_return 0.0 / train_length 1000.0 / train_episodes 97.0 / train_success 0.0
[126723] dataset_size 126723.0 / train_return 2.0 / train_length 303.0 / train_episodes 98.0 / train_success 1.0
[127457] dataset_size 127457.0 / train_return 0.0 / train_length 734.0 / train_episodes 99.0 / train_success 0.0
[127574] dataset_size 127574.0 / train_return 2.0 / train_length 117.0 / train_episodes 100.0 / train_success 1.0
[127865] dataset_size 127865.0 / train_return 2.0 / train_length 291.0 / train_episodes 101.0 / train_success 1.0
[128764] dataset_size 128764.0 / train_return 2.0 / train_length 899.0 / train_episodes 102.0 / train_success 1.0
[130663] dataset_size 130663.0 / train_return 0.0 / train_length 1899.0 / train_episodes 103.0 / train_success 0.0
[130836] dataset_size 130836.0 / train_return 2.0 / train_length 173.0 / train_episodes 104.0 / train_success 1.0
[131837] dataset_size 131837.0 / train_return 0.0 / train_length 1001.0 / train_episodes 105.0 / train_success 0.0
[132457] dataset_size 132457.0 / train_return 2.0 / train_length 620.0 / train_episodes 106.0 / train_success 1.0
[132500] eval_return 0.0 / eval_length 1000.3 / eval_episodes 3.0 / eval_success 0.0
[133057] dataset_size 133057.0 / train_return 2.0 / train_length 600.0 / train_episodes 107.0 / train_success 1.0
[134323] dataset_size 134323.0 / train_return 0.0 / train_length 1266.0 / train_episodes 108.0 / train_success 0.0
[135323] dataset_size 135323.0 / train_return 0.0 / train_length 1000.0 / train_episodes 109.0 / train_success 0.0
[135632] dataset_size 135632.0 / train_return 2.0 / train_length 309.0 / train_episodes 110.0 / train_success 1.0
[136632] dataset_size 136632.0 / train_return 0.0 / train_length 1000.0 / train_episodes 111.0 / train_success 0.0
[137244] dataset_size 137244.0 / train_return 2.0 / train_length 612.0 / train_episodes 112.0 / train_success 1.0
[138638] dataset_size 138638.0 / train_return 0.0 / train_length 1394.0 / train_episodes 113.0 / train_success 0.0
[139090] dataset_size 139090.0 / train_return 2.0 / train_length 452.0 / train_episodes 114.0 / train_success 1.0
[140090] dataset_size 140090.0 / train_return 0.0 / train_length 1000.0 / train_episodes 115.0 / train_success 0.0
"""

# === 2. 数据解析逻辑 ===
def parse_full_logs(log_text):
    train_data = []
    eval_data = []

    # 逐行扫描
    for line in log_text.strip().split('\n'):
        # 提取中括号内的步数
        step_match = re.match(r"\[(\d+)\]", line)
        if not step_match:
            continue
        step = int(step_match.group(1))

        # 提取 Evaluation 数据
        if "eval_success" in line:
            success_match = re.search(r"eval_success ([\d\.]+)", line)
            if success_match:
                eval_data.append({'step': step, 'success': float(success_match.group(1))})
        
        # 提取 Training 数据
        elif "train_success" in line:
            success_match = re.search(r"train_success ([\d\.]+)", line)
            if success_match:
                train_data.append({'step': step, 'success': float(success_match.group(1))})

    return pd.DataFrame(train_data), pd.DataFrame(eval_data)

# 解析数据
df_train, df_eval = parse_full_logs(raw_log_data)

# === 3. 绘图逻辑 ===
plt.figure(figsize=(12, 6))

# 绘制训练数据（点状图，因为波动大）
# 使用 alpha 透明度来展示密集程度，您会发现在11.4万步后，蓝点(1.0)非常密集
plt.scatter(df_train['step'], df_train['success'], 
            label='训练集采样 (Train Sample)', color='blue', alpha=0.3, s=10)

# 绘制训练数据的移动平均线 (Trend Line)
# 窗口设为10，以平滑显示趋势
if len(df_train) > 0:
    df_train['smooth'] = df_train['success'].rolling(window=10, min_periods=1).mean()
    plt.plot(df_train['step'], df_train['smooth'], 
             label='训练趋势 (Train Trend)', color='blue', linewidth=2, alpha=0.8)

# 绘制评估数据（红线连接方块）
plt.plot(df_eval['step'], df_eval['success'], 
         label='评估集成功率 (Eval Success)', color='red', linewidth=2.5, marker='s', markersize=6)

# 添加标注：标出唯一的评估成功点
high_point = df_eval[df_eval['success'] > 0]
if not high_point.empty:
    plt.annotate(f"Eval Spike: {high_point.iloc[0]['success']}", 
                 xy=(high_point.iloc[0]['step'], high_point.iloc[0]['success']), 
                 xytext=(high_point.iloc[0]['step']-20000, high_point.iloc[0]['success']+0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05))

# 图表装饰
plt.title('LS-Imagine 完整训练过程分析 (0 - 140k Steps)', fontsize=14)
plt.xlabel('环境交互步数 (Environment Steps)', fontsize=12)
plt.ylabel('成功率 (Success Rate)', fontsize=12)
plt.ylim(-0.1, 1.2)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='center right')

plt.tight_layout()
plt.show()