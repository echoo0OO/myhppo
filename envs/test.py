import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv

# 设置区域大小
area_size = 1000
resolution = 100  # 热力图分辨率
num_sensors = 5
num_tracks_min = 4  # 轨迹点最少数量

# 生成传感器的随机位置
sensors = np.random.rand(num_sensors, 2) * area_size

# 生成轨迹点（测距点），至少4个
num_tracks = np.random.randint(num_tracks_min, num_tracks_min + 3)  # 4~6个轨迹点
tracks = np.random.rand(num_tracks, 2) * area_size


# 计算GDOP的函数
def compute_gdop(points, sensors):
    if len(points) < 4:
        return 0  # 如果测距点少于4个，GDOP为0

    gdop_values = []

    for sensor in sensors:  # 每个传感器独立计算 H 矩阵
        H = []
        for p in points:
            r = np.linalg.norm(p - sensor)
            if r == 0:
                continue
            h = (p - sensor) / r
            H.append(np.append(h, [1]))  # 添加偏移量

        if len(H) < 4:
            continue  # 需要至少4个观测才能计算GDOP

        H = np.array(H)
        Q = inv(H.T @ H)  # 计算GDOP矩阵
        gdop_values.append(np.sqrt(np.trace(Q)))

    if len(gdop_values) == 0:
        return 0

    return np.mean(gdop_values)  # 取各个传感器计算的GDOP值的平均


# 生成热力图网格点
x_range = np.linspace(0, area_size, resolution)
y_range = np.linspace(0, area_size, resolution)
gdop_map = np.zeros((resolution, resolution))

# 遍历网格计算GDOP
for i, x in enumerate(x_range):
    for j, y in enumerate(y_range):
        new_point = np.array([x, y])
        points = np.vstack([tracks, new_point])
        gdop_map[j, i] = compute_gdop(points, sensors)

# 绘制热力图
plt.figure(figsize=(8, 6))
plt.imshow(gdop_map, extent=(0, area_size, 0, area_size), origin='lower', cmap='jet')
plt.colorbar(label='GDOP')
plt.scatter(sensors[:, 0], sensors[:, 1], c='red', marker='^', label='Sensors')
plt.scatter(tracks[:, 0], tracks[:, 1], c='white', marker='o', label='Tracks')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('GDOP Heatmap')
plt.legend()
plt.show()
