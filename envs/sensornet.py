import math

import gym
import numpy as np
import sys
import torch
import random
import copy
# import traci
# import traci.constants as tc
from gym import spaces
# from bisect import bisect_left2
from scipy.spatial import distance
from envs.GDOP import gdop_heatmap, visualize_gdop_map
from envs.uncertain_model import update_uncertain_model, visualize_progress
from envs.CNNTransformerFeatureExtractor import CNNTransformerFeatureExtractor


class sensornet(gym.Env):

    def __init__(self):
        # 修改区域大小和传感器数量等参数
        self.region_size = 1000  # 区域为 1000*1000 米
        self.sensor_num = 5  # sensor 数量：5个
        self.sensor_data_packet_size = 20 * 1024 * 1024  # 传感器数据包大小：20Mbits
        self.sensor_estimated_radii_max = 100  # 传感器估计半径最大值：100m
        self.uav_height = 100  # 无人机固定飞行高度 100m
        self.sensor_height = 0  # sensor 高度：0m
        self.time_slot_length = 1  # 时隙长度：1s
        self.time_slot_num = 0  # 经过时隙数
        self.max_time_slots_per_trip = 100  # 单程飞行最大时隙数：100
        self.bandwidth = 1e6  # 带宽：1MHz
        self.ref_channel_power_gain_at_1m = 60  # Reference channel power gain at 1 m: 60dB
        self.gaussian_white_noise = -110  # 高斯白噪声：-110dBm
        self.sensor_transmit_power = 0.1  # 传感器发射功率：0.1W
        self.uav_max_speed = 30  # 无人机最大飞行速度：30m/s
        self.g0 = 1.125e-5  # 测量噪声方差系数
        self.los_distance = 250  # 视距传输概率为0.5时的水平距离是250m
        self.gdop_heatmap_size = 100  # GDOP热力图分辨率
        self.gdop_feature_dim = 128  # GDOP热力图特征维度
        self.radius_stable = 0
        self.simulation_steps = 2500
        self.action_dis_len = 2
        # action:离散动作：0：单通信；1：单感知；
        #       连续动作：方向角；速度。
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.action_dis_len),
            spaces.Box(low=np.array([0, 0]), high=np.array([360, self.uav_max_speed]), dtype=np.float32)
        ))

        # observation:传感器估计位置；传感器估计半径；传感器剩余数据量；GDOP热力图特征向量
        observation_low = np.concatenate([
            np.array([0., 0.]),  # 无人机位置
            np.zeros(self.sensor_num * 2),  # 传感器估计位置
            np.zeros(self.sensor_num),  # 传感器估计半径
            np.zeros(self.sensor_num),  # 传感器剩余数据量
            np.zeros(self.gdop_feature_dim)  # GDOP 特征向量（提取后的）
        ])
        observation_high = np.concatenate([
            np.array([self.region_size, self.region_size]),  # 无人机位置
            np.array([self.region_size] * (self.sensor_num * 2)),  # 传感器估计位置
            np.array([self.sensor_estimated_radii_max] * self.sensor_num),  # 最大半径
            np.array([self.sensor_data_packet_size] * self.sensor_num),  # 最大数据量
            np.ones(self.gdop_feature_dim) * 50.0  # 给 GDOP 特征一个合理范围上限，比如 50（或者标准化的话用1）
        ])
        self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float32)

    def reset(self):
        """
        Reset the environment to its initial state.
        :return: Initial state of the environment
        """
        # 重置时间步数
        self.episode_steps = 0
        # 重置飞行程数
        self.flights = 0

        # 重置动作历史
        self.action_pre = []
        self.vehicle_pre = []

        # 生成传感器的真实位置，传感器之间的最小距离为250m
        self.sensor_true_positions = self.poisson_disk_sampling(self.region_size, self.sensor_num, 250)

        # 生成传感器的初始估计位置
        self.sensor_estimated_positions = np.array([
            self.sensor_true_positions[i] + np.random.uniform(-self.sensor_estimated_radii_max / 2,
                                                              self.sensor_estimated_radii_max / 2, 2)
            for i in range(self.sensor_num)
        ])
        # 生成初始测距轨迹点
        self.measurement_points = np.empty((0, 2))  # 初始化为空的二维数组
        self.measurements = np.empty((0, self.sensor_num))  # 初始化为二维数组 [0行, 5列]

        # 重置无人机位置
        self.uav_position = np.array([0, 0])
        self.uav_start = np.array([0, 0])
        self.uav_terminal = self.calcute_terminal_position()

        # 重置传感器状态
        self.data_collection_finished = np.zeros(self.sensor_num, dtype=bool)  # 传感器数据传输是否完成
        self.sensor_estimated_radii = np.full(self.sensor_num, self.sensor_estimated_radii_max)  # 传感器估计半径固定为最大值
        self.sensor_remaining_data = np.full(self.sensor_num, self.sensor_data_packet_size)  # 传感器剩余数据量


        # 重置GDOP热力图
        self.gdop_heatmap = gdop_heatmap(self.sensor_estimated_positions, [], self.region_size, self.gdop_heatmap_size)

        # 生成初始状态
        state = self.generate_initial_state()
        info = self.retrieve_more_info()

        # 返回初始状态
        return np.array(state, dtype=np.float32), info

    def calcute_terminal_position(self):
        """
        计算无人机起点 uav_start 根据传感器估计位置，连线距离起点最远的传感器估计位置距离加上 self.los_distance 的点作为这一程的终点。

        :param self.uav_start: 无人机的起点位置 [2,]
        :return: 无人机的终点位置 [2,]
        """
        # 计算每个传感器估计位置到起点 uav_start 的距离
        distances = np.linalg.norm(self.sensor_estimated_positions - self.uav_start, axis=1)

        # 找到距离起点最远的传感器估计位置的索引
        farthest_sensor_index = np.argmax(distances)

        # 获取最远传感器估计位置
        farthest_sensor_position = self.sensor_estimated_positions[farthest_sensor_index]

        # 计算从起点到最远传感器估计位置的方向向量
        direction_vector = farthest_sensor_position - self.uav_start

        # 归一化方向向量
        direction_vector_normalized = direction_vector / np.linalg.norm(direction_vector)

        # 计算终点位置
        terminal_position = self.uav_start + direction_vector_normalized * (
                distances[farthest_sensor_index] + self.los_distance)

        return terminal_position

    def generate_initial_state(self):
        """
        Generate the initial state of the environment.
        :return: Initial state of the environment
        """
        # 传感器真实位置
        # sensor_true_positions = self.sensor_true_positions.flatten()
        # 传感器估计位置
        sensor_est_positions = self.sensor_estimated_positions.flatten()
        # 传感器估计半径
        sensor_radii = self.sensor_estimated_radii
        # 传感器剩余数据量
        sensor_remaining_data = self.sensor_remaining_data
        # GDOP热力图
        gdop_heatmap = self.gdop_heatmap
        # 将它们拼成一个 torch 向量
        sensor_info = np.concatenate([
            self.uav_position,
            sensor_est_positions,
            sensor_radii,
            sensor_remaining_data
        ], axis=0)
        sensor_info_tensor = torch.tensor(sensor_info, dtype=torch.float32)
        # GDOP 热力图特征提取
        gdop_map = torch.tensor(gdop_heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, 100, 100]
        with torch.no_grad():
            feature_extractor = CNNTransformerFeatureExtractor()
            gdop_features = feature_extractor(gdop_map).squeeze(0)  # shape: [128]
        # 拼接成完整状态
        state = torch.cat([sensor_info_tensor, gdop_features], dim=0)  # shape: [N + 128]
        return state

    def poisson_disk_sampling(self, region_size, num_points, r):
        """
        Perform Poisson Disk Sampling to generate points that are uniformly distributed and well-separated.
        :param region_size: Size of the region (square side length)
        :param num_points: Number of points to generate
        :param r: Minimum distance between points
        :return: Array of points (x, y)
        """
        # 传感器离边界有一定距离
        padding = 100

        def get_random_point():
            return np.random.uniform(padding, region_size - padding, 2)

        def in_bounds(point):
            return padding <= point[0] < region_size - padding and padding <= point[1] < region_size - padding

        def valid_point(point, points, r):
            for p in points:
                if distance.euclidean(point, p) < r:
                    return False
            return True

        def generate_points():
            points = []
            active_list = []

            # Start with multiple random points
            num_seed_points = 10  # 增加种子点的数量
            for _ in range(num_seed_points):
                first_point = get_random_point()
                points.append(first_point)
                active_list.append(first_point)

            while len(points) < num_points and active_list:
                idx = np.random.randint(len(active_list))
                point = active_list[idx]

                for _ in range(30):  # 增加迭代次数
                    angle = np.random.uniform(0, 2 * np.pi)
                    radius = np.random.uniform(r, 2 * r)
                    new_point = point + np.array([radius * np.cos(angle), radius * np.sin(angle)])

                    if in_bounds(new_point) and valid_point(new_point, points, r):
                        points.append(new_point)
                        active_list.append(new_point)
                        if len(points) >= num_points:
                            return np.array(points[:num_points])

                active_list.pop(idx)

            return np.array(points[:num_points])
        return generate_points()

    def step(self, action):
        self.episode_steps = self.episode_steps + 1
        # 从动作中提取离散和连续部分
        discrete_action = action[0]  # 转换为 NumPy 标量
        continuous_action = action[1] # 转换为 NumPy 数组

        # 打印调试信息
        print(f"离散动作类型: {type(discrete_action)}, 值: {discrete_action}")
        print(f"连续动作类型: {type(continuous_action)}, 值: {continuous_action}")

        # 将连续动作从 [-1,1] 缩放到实际范围
        direction_angle = (continuous_action[0] + 1) * 180  # [-1,1] -> [0, 360]
        speed = (continuous_action[1] + 1) * self.uav_max_speed / 2  # [-1,1] -> [0, max_speed]


        # 保存前一步的估计半径和剩余数据
        sensor_estimated_radii_prev = self.sensor_estimated_radii.copy()
        sensor_remaining_data_prev = self.sensor_remaining_data.copy()
        self.uav_position = self.uav_position + np.array(
            [speed * np.cos(direction_angle), speed * np.sin(direction_angle)])

        # 计算真实距离
        true_distances = np.linalg.norm(self.uav_position - self.sensor_true_positions, axis=1)
        if discrete_action == 1:  # 进行感知
            if self.measurement_points.size == 0:  # 初始为空数组时直接赋值
                self.measurement_points = self.uav_position
            else:  # 非空时用vstack堆叠
                self.measurement_points = np.vstack([self.measurement_points, self.uav_position])
            # 计算方差
            variances = self.g0 * true_distances ** 2
            # 生成高斯分布的测距值
            measurements = np.random.normal(true_distances, np.sqrt(variances))
            measurements = measurements.reshape(1, -1)  # 转换为二维行向量 [1行, 5列]
            self.measurements = np.vstack([self.measurements, measurements])
            # 计算GDOP热力图
            self.gdop_heatmap = gdop_heatmap(self.sensor_estimated_positions, self.measurement_points, self.region_size,
                                             self.gdop_heatmap_size)
        elif discrete_action == 0:  # 进行通信
            # 计算无人机与每个传感器之间的估计最远距离
            center_distances = np.linalg.norm(
                self.uav_position - self.sensor_estimated_positions,  # 先计算中心距离
                axis=1
            )
            distances = center_distances + self.sensor_estimated_radii  # 叠加估计半径
            # 判断传感器的最远距离是否在通信范围以及在通信范围内的传感器 data_collection_finished 是否为 0
            within_communication_range = distances <= self.los_distance
            can_collect_data = ~self.data_collection_finished
            # 找出可以通信且数据未收集完成的传感器索引
            sensors_to_collect = np.where(within_communication_range & can_collect_data)[0]
            # 处理可以通信且数据未收集完成的传感器
            for sensor_idx in sensors_to_collect:
                # 预计数据量为在不确定性模型内最差的数据量
                pre_data_amount = self.time_slot_length * self.bandwidth * math.log2(1 + 1e4 / distances[sensor_idx])
                if (pre_data_amount >= self.sensor_remaining_data[sensor_idx]):
                    self.data_collection_finished[sensor_idx] = True
                    self.sensor_remaining_data[sensor_idx] = 0
                    continue
                # 真实采集的数据量
                true_data_amount = self.time_slot_length * self.bandwidth * math.log2(
                    1 + 1e4 / true_distances[sensor_idx])
                if (true_data_amount >= self.sensor_remaining_data[sensor_idx]):
                    self.data_collection_finished[sensor_idx] = True
                    self.sensor_remaining_data[sensor_idx] = 0
                    continue
                self.sensor_remaining_data[sensor_idx] = self.sensor_remaining_data[sensor_idx] - true_data_amount
        # 计算无人机与这一程终点的距离
        distance_to_terminal = np.linalg.norm(self.uav_position - self.uav_terminal)
        # 判断无人机是否在终点的1米范围内,
        # 在则表明一程结束，现在的位置作为起点，计算新一程的终点
        if distance_to_terminal <= 1:
            flight_to_end = True
            self.uav_start = self.uav_position
            self.uav_terminal = self.calcute_terminal_position()
            self.flights = self.flights + 1
            # 计算估计位置和半径
            self.sensor_estimated_positions, self.sensor_estimated_radii = update_uncertain_model(
                self.sensor_estimated_positions,
                self.sensor_estimated_radii,
                self.measurement_points,
                self.measurements,
                self.g0,
                self.flights)
            # 判断估计半径是否收敛
            radius_change = np.abs(self.sensor_estimated_radii - sensor_estimated_radii_prev)
            if np.all(radius_change < 0.5):
                self.radius_stable += 1
            else:
                self.radius_stable = 0
        else:
            flight_to_end = False

        # ---- states ----
        state = self.retrieve_state()

        # ---- reward ----
        reward = self.cal_reward(flight_to_end, sensor_estimated_radii_prev, sensor_remaining_data_prev)

        if self.episode_steps > self.simulation_steps or np.all(self.sensor_remaining_data == 0):
            done = 1
        else:
            done = 0
        info = self.retrieve_more_info
        return state, reward, done, info

    def retrieve_state(self):
        # 传感器估计位置
        sensor_est_positions = self.sensor_estimated_positions.flatten()
        # 传感器估计半径
        sensor_radii = self.sensor_estimated_radii
        # 传感器剩余数据量
        sensor_remaining_data = self.sensor_remaining_data
        # GDOP热力图
        gdop_heatmap = self.gdop_heatmap
        # 将它们拼成一个 torch 向量
        sensor_info = np.concatenate([
            sensor_est_positions,
            sensor_radii,
            sensor_remaining_data
        ], axis=0)
        sensor_info_tensor = torch.tensor(sensor_info, dtype=torch.float32)
        # GDOP 热力图特征提取
        gdop_map = torch.tensor(gdop_heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, 100, 100]
        with torch.no_grad():
            feature_extractor = CNNTransformerFeatureExtractor()
            gdop_features = feature_extractor(gdop_map).squeeze(0)  # shape: [128]
        # 拼接成完整状态
        state = torch.cat([sensor_info_tensor, gdop_features], dim=0)  # shape: [N + 128]
        return state

    def cal_reward(self, flight_to_end, sensor_estimated_radii_prev, sensor_remaining_data_prev):
        reward = 0

        # 时延惩罚：每一步都加上时延惩罚，越大代表时间的消耗越严重
        delay_penalty = -1  # 例如每个时隙的惩罚系数
        reward += delay_penalty

        # 终点奖励：到达终点时，给与较大的奖励
        if flight_to_end:
            reward += 100  # 到达终点的奖励
        if np.sum(self.sensor_remaining_data) == 0:
            reward += 500  # 到达终点的奖励
            return reward

        # 感知和通信奖励的计算
        if self.radius_stable < 3:
            # 感知奖励系数较大，通信奖励系数较小
            perception_reward_factor = 10
            communication_reward_factor = 1
        else:
            # 感知奖励系数较小，通信奖励系数较大
            perception_reward_factor = 1
            communication_reward_factor = 10

        # 感知奖励：估计半径的变化量乘以感知奖励系数
        perception_reward = np.sum(perception_reward_factor * (sensor_remaining_data_prev - self.sensor_remaining_data))

        # 通信奖励：数据量的变化乘以通信奖励系数
        communication_reward = 0
        for i in range(self.sensor_num):
            # 如果数据量变化是正的，表示通信成功
            if self.sensor_remaining_data[i] < sensor_remaining_data_prev[i]:
                true_data_amount = self.sensor_remaining_data_prev[i] - self.sensor_remaining_data[i]
                communication_reward += communication_reward_factor * true_data_amount

        reward += perception_reward + communication_reward

        return reward

    def retrieve_more_info(self):
        # 计算估计位置与真实位置的欧氏距离
        distance_errors = np.linalg.norm(
            self.sensor_estimated_positions - self.sensor_true_positions,
            axis=1  # 按行计算每个传感器的误差
        ).tolist()  # 转换为 Python 列表
        info = {
            # "sensor_remaining_data": self.sensor_remaining_data.tolist(),  # 转换为 Python 原生列表
            # "sensor_estimated_positions": self.sensor_estimated_positions.tolist(),
            # "sensor_estimated_radii": self.sensor_estimated_radii.tolist(),
            "distance_errors": distance_errors,
        }
        return info



# if __name__ == "__main__":
#     fw = FreewheelingIntersectionEnv()
#     fw.reset()
#     for i in range(50):
#         fw.sumo_step()
#     raw_info = fw.retrieve_raw_info()
#     print(raw_info)
#     fw.retrieve_state(raw_info)
