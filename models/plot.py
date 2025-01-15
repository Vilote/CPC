import matplotlib.pyplot as plt

# Shot 数量
shots = [5, 10, 15, 20, 25, 30]

# 各个方法在不同 shot 数量下的性能数据
CPC = [56.2, 67.3, 40, 78, 82.6, 84.6]
SA2SEI = [49.1, 62.7, 71.6, 78.3, 82.4, 85.6]

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制每种方法的曲线
plt.plot(shots, CPC, marker='o', linestyle='-', label='FineZero', color='red')

plt.plot(shots, SA2SEI, marker='o', linestyle='-', label='SA2SEI (Proposed)', color='blue')

# 设置标题和标签
plt.title('Performance of Different Methods')
plt.xlabel('Shot')
plt.ylabel('P_ec/%')

# 添加图例
plt.legend()

# 显示网格
plt.grid(True)

# 保存图像
plt.savefig('./line_plot.png')

# 显示图形
plt.show()