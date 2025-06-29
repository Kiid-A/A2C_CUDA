import matplotlib.pyplot as plt
import numpy as np

# 定义数据
hs_values = ['HS=128', 'HS=1280', 'HS=12800']

# HS=128的数据
spe_128 = [0.07929227352142335, 0.06787137985229492, 0.05401999950408935]
mrt_128 = [0.01582674980163574, 0.0044553279876708984, 0.0039318323135375975]
mut_128 = [0.0025179147720336913, 0.0021799802780151367, 0.0017978429794311523]

# HS=1280的数据
spe_1280 = [0.1681058645248413, 0.13332529067993165, 0.11417088508605958]
mrt_1280 = [0.05512871742248535, 0.022643065452575682, 0.01866621971130371]
mut_1280 = [0.04991762638092041, 0.050721359252929685, 0.041109776496887206]

# HS=12800的数据
spe_12800 = [6.564965724945068, 4.732813167572021, 5.187824535369873]
mrt_12800 = [2.5331607818603517, 0.7712450504302979, 1.1899905443191527]
mut_12800 = [3.971861553192139, 3.8976580619812013, 3.941616082191467]

# 定义颜色
colors = ['#00A1FF', '#5ed935', '#f8ba00']
labels = ['cuda', 'cuda-cpu', 'pytorch-opt-cpu']

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Source Han Sans SC']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 绘制 SPE 图表（对数坐标）
bars1 = axes[0].bar(np.arange(len(labels)) - 0.2, spe_128, width=0.4, label=hs_values[0], color=colors[0], zorder=3)
bars2 = axes[0].bar(np.arange(len(labels)), spe_1280, width=0.4, label=hs_values[1], color=colors[1], zorder=3)
bars3 = axes[0].bar(np.arange(len(labels)) + 0.2, spe_12800, width=0.4, label=hs_values[2], color=colors[2], zorder=3)
axes[0].set_title('不同隐藏层大小下的 SPE 对比（对数坐标）')
axes[0].set_xlabel('测试配置')
axes[0].set_ylabel('Seconds Per Episode')
axes[0].set_xticks(np.arange(len(labels)))
axes[0].set_xticklabels(labels)
axes[0].legend(title='隐藏层大小')
axes[0].grid(axis='y', linestyle='--', zorder=0)
axes[0].set_yscale('log')  # 设置对数坐标
# 优化对数刻度标签显示
axes[0].set_yticks([1e-3, 1e-2, 1e-1, 1, 10])
axes[0].set_yticklabels(['0.001', '0.01', '0.1', '1', '10'])

# 绘制 MRT 图表（对数坐标）
axes[1].bar(np.arange(len(labels)) - 0.2, mrt_128, width=0.4, label=hs_values[0], color=colors[0], zorder=3)
axes[1].bar(np.arange(len(labels)), mrt_1280, width=0.4, label=hs_values[1], color=colors[1], zorder=3)
axes[1].bar(np.arange(len(labels)) + 0.2, mrt_12800, width=0.4, label=hs_values[2], color=colors[2], zorder=3)
axes[1].set_title('不同隐藏层大小下的 MRT 对比（对数坐标）')
axes[1].set_xlabel('测试配置')
axes[1].set_ylabel('Mean Rollout Time')
axes[1].set_xticks(np.arange(len(labels)))
axes[1].set_xticklabels(labels)
axes[1].legend(title='隐藏层大小')
axes[1].grid(axis='y', linestyle='--', zorder=0)
axes[1].set_yscale('log')  # 设置对数坐标
# 优化对数刻度标签显示
axes[1].set_yticks([1e-3, 1e-2, 1e-1, 1, 10])
axes[1].set_yticklabels(['0.001', '0.01', '0.1', '1', '10'])

# 绘制 MUT 图表（对数坐标）
axes[2].bar(np.arange(len(labels)) - 0.2, mut_128, width=0.4, label=hs_values[0], color=colors[0], zorder=3)
axes[2].bar(np.arange(len(labels)), mut_1280, width=0.4, label=hs_values[1], color=colors[1], zorder=3)
axes[2].bar(np.arange(len(labels)) + 0.2, mut_12800, width=0.4, label=hs_values[2], color=colors[2], zorder=3)
axes[2].set_title('不同隐藏层大小下的 MUT 对比（对数坐标）')
axes[2].set_xlabel('测试配置')
axes[2].set_ylabel('Mean Update Time')
axes[2].set_xticks(np.arange(len(labels)))
axes[2].set_xticklabels(labels)
axes[2].legend(title='隐藏层大小')
axes[2].grid(axis='y', linestyle='--', zorder=0)
axes[2].set_yscale('log')  # 设置对数坐标
# 优化对数刻度标签显示
axes[2].set_yticks([1e-3, 1e-2, 1e-1, 1, 10])
axes[2].set_yticklabels(['0.001', '0.01', '0.1', '1', '10'])

# 调整布局
plt.tight_layout()

# 显示图表
plt.savefig('show_log.png')