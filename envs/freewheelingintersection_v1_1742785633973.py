#file:D:\研二\感知采集\study_point_one\Hybrid_PPO\envs\sensornet.py
import math
import gym
import numpy as np
import sys
import random
import copy
from gym import spaces
from bisect import bisect_left
from scipy.spatial import distance

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
        self.max_time_slots_per_trip = 100  # 单程飞行最大时隙数：100
        self.bandwidth = 1e6  # 带宽：1MHz
        self.ref_channel_power_gain_at_1m = 60  # Reference channel power gain at 1 m: 60dB
        self.gaussian_white_noise = -110  # 高斯白噪声：-110dBm
        self.sensor_transmit_power = 0.1  # 传感器发射功率：0.1W
        self.uav_max_speed = 30  # 无人机最大飞行速度：30m/s

        self.gdop_heatmap_size = 32  # GDOP热力图大小

        # 轨迹点缓冲区大小
        self.trajectory_buffer_size = 10

        # action:离散动作：0：不感知不通信；1：单感知；2：单通信
        #       连续动作：方向角；速度。
        self.action_space = spaces.Tuple((
            spaces.Discrete(3),
            spaces.Box(low=np.array([0, 0]), high=np.array([360, self.uav_max_speed]), dtype=np.float32)
        ))

        # observation:传感器估计位置；传感器估计半径；传感器剩余数据量；GDOP热力图；轨迹点及其测距值
        observation_low = np.concatenate([
            np.zeros(self.sensor_num * 2),  # 传感器估计位置
            np.zeros(self.sensor_num),  # 传感器估计半径
            np.zeros(self.sensor_num),  # 传感器剩余数据量
            np.array([0.] * (self.gdop_heatmap_size * self.gdop_heatmap_size)),  # GDOP热力图
            np.array([0.] * (self.sensor_num * self.trajectory_buffer_size * 3))  # 轨迹点及其测距值
        ])
        observation_high = np.concatenate([
            np.array([self.region_size] * (self.sensor_num * 2)),  # 传感器估计位置 (假设最大范围为1000m)
            np.array([self.sensor_estimated_radii_max] * self.sensor_num),  # 传感器估计半径 (假设最大半径为100m)
            np.array([self.sensor_data_packet_size] * self.sensor_num),  # 传感器剩余数据量 (假设最大值为1)
            np.array([50.] * (self.gdop_heatmap_size * self.gdop_heatmap_size)),  # GDOP热力图
            np.array([self.region_size] * (self.sensor_num * self.trajectory_buffer_size * 3))  # 轨迹点及其测距值
        ])
        self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float32)

    def reset(self):
        """
        Reset the environment to its initial state.
        :return: Initial state of the environment
        """
        # 重置时间步数
        self.episode_steps = 0

        # 重置动作历史
        self.action_pre = []
        self.vehicle_pre = []

        # 重置传感器状态
        self.sensor_estimated_radii = np.full(self.sensor_num, self.sensor_estimated_radii_max)  # 传感器估计半径固定为最大值
        self.sensor_remaining_data = np.full(self.sensor_num, self.sensor_data_packet_size)  # 传感器剩余数据量

        # 生成传感器的真实位置，传感器之间的最小距离为250m
        self.sensor_true_positions = self.poisson_disk_sampling(self.region_size, self.sensor_num, 250)

        # 生成传感器的初始估计位置
        self.sensor_estimated_positions = np.array([
            self.sensor_true_positions[i] + np.random.uniform(-self.sensor_estimated_radii_max/2,
                                                              self.sensor_estimated_radii_max/2, 2)
            for i in range(self.sensor_num)
        ])

        # 初始化轨迹点缓冲区
        self.sensor_trajectory_buffers = [np.zeros((self.trajectory_buffer_size, 3)) for _ in range(self.sensor_num)]

        # 重置GDOP热力图（如果需要）
        self.gdop_heatmap = np.random.uniform(0, 50, (self.gdop_heatmap_size, self.gdop_heatmap_size)).flatten()

        # 生成初始状态
        state = self.generate_initial_state()

        # 返回初始状态
        return np.array(state, dtype=np.float32)

    def generate_initial_state(self):
        """
        Generate the initial state of the environment.
        :return: Initial state of the environment
        """
        # 传感器估计位置
        sensor_positions = self.sensor_estimated_positions.flatten()

        # 传感器估计半径
        sensor_radii = self.sensor_estimated_radii

        # 传感器剩余数据量
        sensor_remaining_data = self.sensor_remaining_data

        # GDOP热力图
        gdop_heatmap = self.gdop_heatmap

        # 轨迹点及其测距值
        trajectory_data = np.concatenate([buffer.flatten() for buffer in self.sensor_trajectory_buffers], axis=0)

        # 拼接所有状态信息
        state = np.concatenate([
            sensor_positions,
            sensor_radii,
            sensor_remaining_data,
            gdop_heatmap,
            trajectory_data
        ], axis=0)

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
            return np.random.uniform(padding, region_size-padding, 2)

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

        # 生成点并验证最小距离约束
        while True:
            points = generate_points()
            if len(points) == num_points:
                distances = []
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        dist = distance.euclidean(points[i], points[j])
                        distances.append(dist)
                min_distance = min(distances)
                if min_distance > r:
                    return points

    def sumo_step(self):
        """
        SUMO steps.
        :return:
        """
        traci.simulationStep()
        self.episode_steps += 1

    def step(self, action):
        """
        Note: the sumo(or traci) doesn't need an action every step until one specific phase is over,
              but the abstract method 'step()' needs as you can see.
              Thus only a new action is input will we change the traffic light state, otherwise just do
              traci.simulationStep() consecutively.
        :param action:list, e.g. [4, [12, 11, 13, 15, 10, 12, 16, 23]],
                         the first element is the phase next period,
                         and the latter ones are duration w.r.t all phases.
        :return: next_state, reward, done, info
        """

        stage_next = action[0]
        stage_duration = action[1]

        # SmartWolfie is a traffic signal control program defined in FW_Inter.add.xml.
        # We achieve hybrid action space control through switch its stage and steps
        # (controlled by YELLOW(3s in default) and GREEN(stage_duration)).
        # There is possibility that stage next period is same with the stage right now
        # Otherwise there is a yellow between two different stages.

        if len(self.action_pre) and action[0] != self.action_pre[0]:
            yellow = self.phase_transformer[self.action_pre[0]][action[0]]
            traci.trafficlight.setPhase('SmartWolfie', yellow)
            for t in range(self.yellow):
                self.sumo_step()

        traci.trafficlight.setPhase('SmartWolfie', stage_next)
        for t in range(stage_duration):
            self.sumo_step()

        raw = self.retrieve_raw_info()
        # ---- states ----
        state = self.retrieve_state(raw)

        # ---- reward ----
        vehicle_now, waiting_time = self.retrieve_reward(raw)
        departed_vehicle = list(set(self.vehicle_pre) - set(vehicle_now))
        reward = self.cal_reward(departed_vehicle, waiting_time)

        self.vehicle_pre = copy.deepcopy(vehicle_now)

        if self.episode_steps > self.simulation_steps:
            done = 1
        else:
            done = 0
        info = self.retrieve_more_info(raw)
        return state, reward, done, info



    def retrieve_state(self, raw):
        """
        :return:
        """

        vehicle_types_so_far = []
        state = np.array([])
        cell_space = np.linspace(0, 240, num=(self.cells + 1))

        for type in raw:
            vehicle_types_so_far.append(type)
        for vehicle_type in self.vehicle_types:
            position = np.zeros(self.cells)
            if vehicle_type in vehicle_types_so_far:
                for vehicle in raw.get(vehicle_type):
                    position[bisect_left(cell_space, vehicle[1]) - 1] = 1.
            state = np.concatenate((state, position), axis=0)

        return state

    def retrieve_reward(self, raw):
        """
        :return:
        """
        vehicle_IDs = []
        accumulated_waiting_time = []

        raw = list(raw.items())
        for vehicles_specific_type in raw:
            # spe: specific
            accumulated_waiting_time_spe_type = []
            for vehicle in vehicles_specific_type[1]:
                vehicle_IDs.append(int(vehicle[0]))
                accumulated_waiting_time_spe_type.append(vehicle[3])
            accumulated_waiting_time.append(accumulated_waiting_time_spe_type)
        accumulated_waiting_time = sum(accumulated_waiting_time, [])

        return vehicle_IDs, accumulated_waiting_time

    def cal_reward(self, departed_vehicle, waiting_time):
        waiting_time_ = np.array([t - self.punish_threshold for t in waiting_time if t > self.punish_threshold])
        # reward = - np.sum(self.punish_scale * np.sqrt(np.divide(waiting_time_, self.latitude))) + len(departed_vehicle) * self.bonus
        # reward = self.punish_scale * (np.exp(len(departed_vehicle) / self.latitude) - 1)
        reward = len(departed_vehicle)

        return reward

    def retrieve_more_info(self, raw):
        """
        Mainly for evaluation

        :param raw:
        :return:
        """
        queue = []
        raw = list(raw.items())
        for vehicles_specific_type in raw:
            for vehicle in vehicles_specific_type[1]:
                if vehicle[2] < self.max_queuing_speed:
                    queue.append(vehicle)

        return len(queue)

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def render(self, mode='human'):
        pass

    def close(self):
        """
        :return:
        """
        traci.close()

if __name__ == "__main__":
    fw = FreewheelingIntersectionEnv()
    fw.reset()
    for i in range(50):
        fw.sumo_step()
    raw_info = fw.retrieve_raw_info()
    print(raw_info)
    fw.retrieve_state(raw_info)
