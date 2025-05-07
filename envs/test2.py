#file:D:\研二\感知采集\study_point_one\Hybrid_PPO\envs\test2.py
import numpy as np
import matplotlib.pyplot as plt
# from brokenaxes import brokenaxes
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
from scipy.optimize import minimize
from scipy.stats import chi2

# 参数设置
true_pos = np.array([10, 8])  # 真实位置（固定）
min_anchors_per_round = 4  # 每程最少测距点数
max_anchors_per_round = 8  # 每程最多测距点数
num_rounds = 10  # 定位程数
confidence_level = 0.99  # 置信水平
area_size = 50  # 测距点生成区域半径
alpha = 0.03  # 噪声方差系数

# 初始化存储结构
all_anchors = np.empty((0, 2))
all_measurements = np.empty(0)

# 初始估计
x_est = true_pos + 15 * (np.random.rand(2) - 0.5)  # 初始随机估计
r_est = 25  # 初始半径

# 历史记录
hist_error = np.zeros(num_rounds + 1)
hist_error[0] = np.linalg.norm(x_est - true_pos)
hist_pos = np.zeros((num_rounds + 1, 2))
hist_pos[0, :] = x_est
hist_radius = np.zeros(num_rounds + 1)
hist_radius[0] = r_est


# 目标函数（加权残差平方和）
def objective_function(x, anchors, measurements, weights):
    d_est = np.linalg.norm(anchors - x, axis=1)
    residuals = measurements - d_est
    return np.sum(weights * residuals ** 2)


# 计算雅可比矩阵
def compute_jacobian(x, anchors):
    delta = anchors - x
    d = np.linalg.norm(delta, axis=1)
    d[d < 1e-3] = 1e-3  # 正则化处理
    return -delta / d[:, np.newaxis]


# 可视化函数
def visualize_progress(round, anchors, true_pos, hist_pos, hist_radius, hist_error):
    if round % 2 != 0:
        return
    plt.figure(figsize=(15, 5))

    # 位置可视化
    plt.subplot(1, 3, 1)
    plt.scatter(anchors[:, 0], anchors[:, 1], c='g', alpha=0.5, label='测距点')
    plt.plot(true_pos[0], true_pos[1], 'rp', markersize=10, label='真实位置')
    plt.plot(hist_pos[:round + 1, 0], hist_pos[:round + 1, 1], 'b-o', label='估计轨迹')
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.title(f'多程定位 - 第 {round} 程', fontsize=12)
    plt.xlabel('X 坐标', fontsize=10)
    plt.ylabel('Y 坐标', fontsize=10)

    # 半径收敛
    plt.subplot(1, 3, 2)
    plt.plot(range(round + 1), hist_radius[:round + 1], 'b-o')
    plt.grid(True)
    plt.title('估计半径收敛过程', fontsize=12)
    plt.xlabel('程数', fontsize=10)
    plt.ylabel('半径', fontsize=10)

    # 误差收敛
    plt.subplot(1, 3, 3)
    plt.plot(range(round + 1), hist_error[:round + 1], 'm-s')
    plt.grid(True)
    plt.title('位置误差收敛', fontsize=12)
    plt.xlabel('程数', fontsize=10)
    plt.ylabel('误差距离 (m)', fontsize=10)

    plt.tight_layout()
    plt.show()


# 多程定位主循环
for round in range(1, num_rounds + 1):
    num_anchors = np.random.randint(min_anchors_per_round, max_anchors_per_round + 1)
    new_anchors = true_pos + area_size * (np.random.rand(num_anchors, 2) - 0.5)
    true_d = np.linalg.norm(new_anchors - true_pos, axis=1)
    noise = np.sqrt(alpha * true_d ** 2) * np.random.randn(num_anchors)
    new_measurements = true_d + noise

    all_anchors = np.vstack((all_anchors, new_anchors))
    all_measurements = np.hstack((all_measurements, new_measurements))

    # 动态权重最小二乘估计
    current_d_est = np.linalg.norm(all_anchors - x_est, axis=1)
    weights = 1 / (alpha * current_d_est ** 2)

    # 进行优化求解
    result = minimize(objective_function, x_est, args=(all_anchors, all_measurements, weights),
                      bounds=[(x_est[0] - 2 * r_est, x_est[0] + 2 * r_est),
                              (x_est[1] - 2 * r_est, x_est[1] + 2 * r_est)])
    x_new = result.x

    # 不确定性量化
    J = compute_jacobian(x_new, all_anchors)
    info_matrix = J.T @ np.diag(weights) @ J
    cov_matrix = np.linalg.inv(info_matrix)

    eigvals, _ = np.linalg.eig(cov_matrix)
    semi_axes = np.sqrt(eigvals) * np.sqrt(chi2.ppf(confidence_level, 2))
    r_new = np.max(semi_axes)

    # 更新估计
    pos_update_gain = 1 / (1 + 0.1 * round)
    x_est = x_est + pos_update_gain * (x_new - x_est)

    radius_update_gain = 0.3
    r_est = r_est + radius_update_gain * (r_new - r_est)

    # 记录误差
    hist_error[round] = np.linalg.norm(x_est - true_pos)
    hist_pos[round, :] = x_est
    hist_radius[round] = r_est

    # 可视化
    visualize_progress(round, all_anchors, true_pos, hist_pos, hist_radius, hist_error)
