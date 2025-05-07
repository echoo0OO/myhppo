# file:D:\研二\感知采集\study_point_one\Hybrid_PPO\envs\test2.py
import numpy as np
import matplotlib.pyplot as plt

# 保证可以显示中文字符
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
from scipy.optimize import minimize
from scipy.stats import chi2


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
    plt.figure()
    # plt.subplot(1, 3, 1)
    plt.scatter(anchors[:, 0], anchors[:, 1], c='g', alpha=0.5, label='测距点')
    plt.plot(true_pos[0], true_pos[1], 'rp', markersize=10, label='真实位置')
    plt.plot(hist_pos[:round + 1, 0], hist_pos[:round + 1, 1], 'b-o', label='估计轨迹')
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.title(f'多程定位 - 第 {round} 程', fontsize=12)
    plt.xlabel('X 坐标', fontsize=10)
    plt.ylabel('Y 坐标', fontsize=10)
    circle = plt.Circle((hist_pos[round, 0], hist_pos[round, 1]), hist_radius[round], color='r', fill=False,
                        linestyle='--', label='估计半径')
    plt.gca().add_artist(circle)
    plt.show()

    # 半径收敛
    plt.figure()
    # plt.subplot(1, 3, 2)
    plt.plot(range(round + 1), hist_radius[:round + 1], 'b-o')
    plt.grid(True)
    plt.title('估计半径收敛过程', fontsize=12)
    plt.xlabel('程数', fontsize=10)
    plt.ylabel('半径', fontsize=10)
    plt.show()

    # 误差收敛
    plt.figure()
    # plt.subplot(1, 3, 3)
    plt.plot(range(round + 1), hist_error[:round + 1], 'm-s')
    plt.grid(True)
    plt.title('位置误差收敛', fontsize=12)
    plt.xlabel('程数', fontsize=10)
    plt.ylabel('误差距离 (m)', fontsize=10)

    # plt.tight_layout()
    plt.show()


def update_uncertain_model(x_est, r_est, anchors, measurements, g0, num_rounds):
    confidence_level = 0.99  # 置信水平
    # alpha = 1.125*10e-10  # 噪声方差系数
    # num_anchors = np.random.randint(min_anchors_per_round, max_anchors_per_round + 1)
    # new_anchors = true_pos + area_size * (np.random.rand(num_anchors, 2) - 0.5)
    # true_d = np.linalg.norm(new_anchors - true_pos, axis=1)
    # noise = np.sqrt(alpha * true_d ** 2) * np.random.randn(num_anchors)
    # new_measurements = true_d + noise
    #
    # all_anchors = np.vstack((all_anchors, new_anchors))
    # all_measurements = np.hstack((all_measurements, new_measurements))

    # 动态权重最小二乘估计
    current_d_est = np.linalg.norm(anchors - x_est, axis=1)
    weights = 1 / (g0 * current_d_est ** 2)

    # 进行优化求解
    result = minimize(objective_function, x_est, args=(anchors, measurements, weights),
                      bounds=[(x_est[0] - 2 * r_est, x_est[0] + 2 * r_est),
                              (x_est[1] - 2 * r_est, x_est[1] + 2 * r_est)])
    x_new = result.x

    # 不确定性量化
    J = compute_jacobian(x_new, anchors)
    info_matrix = J.T @ np.diag(weights) @ J
    cov_matrix = np.linalg.inv(info_matrix)

    eigvals, _ = np.linalg.eig(cov_matrix)
    semi_axes = np.sqrt(eigvals) * np.sqrt(chi2.ppf(confidence_level, 2))
    r_new = np.max(semi_axes)

    # 更新估计
    pos_update_gain = 1 / (1 + 0.1 * num_rounds)
    x_est_new = x_est + pos_update_gain * (x_new - x_est)
    radius_update_gain = 0.3
    r_est_new = r_est + radius_update_gain * (r_new - r_est)
    return x_est_new, r_est_new


# # 多程定位主循环
#
# # 记录误差
# hist_error[round] = np.linalg.norm(x_est - true_pos)
# hist_pos[round, :] = x_est
# hist_radius[round] = r_est
#
# # 可视化
# visualize_progress(round, all_anchors, true_pos, hist_pos, hist_radius, hist_error)
