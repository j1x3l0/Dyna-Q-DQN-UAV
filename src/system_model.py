import logging
import random
import numpy as np
import torch
import os
from numpy.random import SeedSequence
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.propagate = False

log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)

file_handler = RotatingFileHandler(
    os.path.join(log_dir, 'system_model.log'),
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

class Config:
    def __init__(self, seed=42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.N = 3
        self.M = 6
        self.F = 4
        self.v_max = 25.0
        self.d_min = 5.0
        self.tau = 1.0
        self.tau_f = 0.3
        self.tau_s = 0.5
        self.tau_d = 0.2
        self.p_A = 0.1
        self.p_m = 0.01
        self.p_i_r = 0.5
        self.alpha = 2.0
        self.K = 10.0
        self.Gamma_o = 0.5
        self.mu = 0.5
        self.E_max = 10.0
        self.A_min = 5.0
        self.A_max = 15.0
        self.noise_power = 1e-12
        self.gamma_reward = 0.95
        self.gamma_forward = 0.95
        self.eta = 50.0
        self.eta1 = 0.1
        self.reward_scale = 1.0
        self.P_0 = 5.0
        self.denom_epsilon = 1e-6
        self.omega_0 = 1e-1
        self.boundary = 50.0
        self.d_soft = 10.0
        self.eta_soft = 5.0
        self.init_min_separation = 15.0
        self.dyna_k = 1
        self.reward_mode = 'ee_ratio'  # 'ee_ratio' or 'additive'

        seed_sequence = SeedSequence(seed)
        env_seq, action_seq, replay_seq, model_seq, dyna_seq, torch_seq = seed_sequence.spawn(6)
        self.rngs = {
            'env': np.random.default_rng(env_seq),
            'action': np.random.default_rng(action_seq),
            'replay': np.random.default_rng(replay_seq),
            'model': np.random.default_rng(model_seq),
            'dyna': np.random.default_rng(dyna_seq),
            'torch': np.random.default_rng(torch_seq),
        }
        self.torch_seed = int(self.rngs['torch'].integers(0, 2**31 - 1))
        
        logger.info(f"Config initialized: N={self.N}, M={self.M}, F={self.F}")
        logger.info(f"UAV params: v_max={self.v_max}, d_min={self.d_min}")
        logger.info(f"Reward params: gamma_forward={self.gamma_forward}, eta={self.eta}, eta1={self.eta1}, reward_scale={self.reward_scale}, P_0={self.P_0}, denom_epsilon={self.denom_epsilon}")
        logger.info(f"Collision params: d_min={self.d_min}, d_soft={self.d_soft}, eta={self.eta}, eta_soft={self.eta_soft}, init_min_sep={self.init_min_separation}, boundary={self.boundary}")
        logger.info(f"Time slots: tau_f={self.tau_f}, tau_s={self.tau_s}, tau_d={self.tau_d}")

class ChannelModel:
    def __init__(self, config, rng):
        self.config = config
        self.rng = rng
        self.omega_0 = getattr(config, 'omega_0', 1e-3)
        logger.info(f"ChannelModel initialized: omega_0={self.omega_0}, alpha={config.alpha}, K={config.K}")
    
    def large_scale_fading(self, distance):
        fading = self.omega_0 * (distance ** (-self.config.alpha))
        return fading
    
    def rician_fading(self, distance):
        K = self.config.K
        los_component = self.rng.standard_normal((1, self.config.F)) + 1j * self.rng.standard_normal((1, self.config.F))
        los_component = los_component / np.linalg.norm(los_component)
        nlos_component = self.rng.standard_normal((1, self.config.F)) + 1j * self.rng.standard_normal((1, self.config.F))
        nlos_component = nlos_component / np.linalg.norm(nlos_component)
        psi = self.large_scale_fading(distance)
        h = np.sqrt(psi) * (np.sqrt(K / (1 + K)) * los_component + np.sqrt(1 / (1 + K)) * nlos_component)
        return h
    
    def get_channel(self, pos1, pos2):
        distance = np.linalg.norm(pos1 - pos2)
        h = self.rician_fading(distance)
        logger.debug(f"get_channel: distance={distance:.2f}, channel_norm={np.linalg.norm(h):.6f}")
        return h, distance

class GroundUser:
    def __init__(self, config, idx, rng):
        self.config = config
        self.idx = idx
        self.rng = rng
        self.pos = self.rng.uniform(-config.boundary, config.boundary, 2)
        self.energy = self.rng.uniform(0, config.E_max)
        self.buffer = self.rng.uniform(config.A_min, config.A_max)
        self.data_rate_a = 0.0
        self.data_rate_b = 0.0
        self.access = False
        self.mode = 0
        logger.info(f"GroundUser {idx} initialized: pos={self.pos}, energy={self.energy:.2f}, buffer={self.buffer:.2f}")
    
    def update_buffer(self, data_sent, new_data):
        old_buffer = self.buffer
        self.buffer = max(0, self.buffer - data_sent + new_data)
        logger.debug(f"GU {self.idx} buffer update: old={old_buffer:.2f}, sent={data_sent:.2f}, new={new_data:.2f}, now={self.buffer:.2f}")
    
    def update_energy(self, harvested_energy, consumed_energy):
        old_energy = self.energy
        self.energy = max(0, min(self.config.E_max, self.energy + harvested_energy - consumed_energy))
        logger.debug(f"GU {self.idx} energy update: old={old_energy:.4f}, harvested={harvested_energy:.4f}, consumed={consumed_energy:.4f}, now={self.energy:.4f}")

class UAV:
    def __init__(self, config, idx, rng):
        self.config = config
        self.idx = idx
        self.rng = rng
        self.pos = np.array([self.rng.uniform(-config.boundary, config.boundary), 
                            self.rng.uniform(-config.boundary, config.boundary),
                            self.rng.uniform(50, 100)])
        self.buffer = 0.0
        self.energy = 100.0
        self.velocity = np.zeros(3)
        self.scheduled = False
        logger.info(f"UAV {idx} initialized: pos={self.pos}, energy={self.energy}")
    
    def move(self, direction, speed):
        old_pos = self.pos.copy()
        speed = min(speed, self.config.v_max)
        self.velocity = direction * speed
        self.pos = self.pos + self.velocity * self.config.tau_f
        logger.debug(f"UAV {self.idx} move: direction={direction}, speed={speed:.2f}, pos from {old_pos} to {self.pos}")
    
    def update_buffer(self, data_received, data_sent):
        old_buffer = self.buffer
        self.buffer = max(0, self.buffer + data_received - data_sent)
        logger.debug(f"UAV {self.idx} buffer update: old={old_buffer:.2f}, received={data_received:.2f}, sent={data_sent:.2f}, now={self.buffer:.2f}")

class RBStation:
    def __init__(self, config):
        self.config = config
        self.pos = np.array([0, 0, 0])
        logger.info(f"RBStation initialized at pos={self.pos}")

class Environment:
    def __init__(self, config):
        logger.info("=" * 60)
        logger.info("Initializing Environment...")
        logger.info("=" * 60)
        
        self.config = config
        self.rng = config.rngs['env']
        self.channel_model = ChannelModel(config, self.rng)
        self.rbs = RBStation(config)
        self.uavs = [UAV(config, i, self.rng) for i in range(config.N)]
        self.gus = [GroundUser(config, i, self.rng) for i in range(config.M)]
        self.time_slot = 0
        self.last_step_info = None
        
        logger.info(f"Environment initialized: {config.N} UAVs, {config.M} GUs")
        logger.info("=" * 60)
    
    def reset(self, case=1):
        logger.info(f"\n--- Resetting environment (case={case}) ---")
        
        if case == 1:
            logger.info("Case 1: UAVs start at random positions (with minimum separation)")
            max_retries = 100
            for attempt in range(max_retries):
                positions = []
                for i, uav in enumerate(self.uavs):
                    pos = np.array([self.rng.uniform(-self.config.boundary, self.config.boundary),
                                   self.rng.uniform(-self.config.boundary, self.config.boundary),
                                   self.rng.uniform(50, 100)])
                    positions.append(pos)
                distances = []
                for i in range(self.config.N):
                    for j in range(i + 1, self.config.N):
                        distances.append(np.linalg.norm(positions[i] - positions[j]))
                min_dist = min(distances) if distances else float('inf')
                if min_dist >= self.config.init_min_separation:
                    for i, uav in enumerate(self.uavs):
                        uav.pos = positions[i]
                    logger.info(f"UAV initial positions accepted (attempt {attempt+1}, min_dist={min_dist:.1f}m)")
                    break
            else:
                logger.warning(f"Could not find separation after {max_retries} attempts, using last positions")
                for i, uav in enumerate(self.uavs):
                    uav.pos = positions[i]
            for i, uav in enumerate(self.uavs):
                logger.debug(f"UAV {i} pos: {uav.pos}")
        else:
            logger.info("Case 2: All UAVs start at same position (0, 0, 100)")
            for uav in self.uavs:
                uav.pos = np.array([0, 0, 100])
        
        for uav in self.uavs:
            uav.buffer = 0.0
            uav.energy = 100.0
            uav.velocity = np.zeros(3)
            uav.scheduled = False
        
        for gu in self.gus:
            gu.energy = self.rng.uniform(0, self.config.E_max)
            gu.buffer = self.rng.uniform(self.config.A_min, self.config.A_max)

        self.time_slot = 0
        self.last_step_info = None
        logger.info(f"Environment reset completed, time_slot={self.time_slot}")
        
        return self.get_state()
    
    def get_coverage(self, uav):
        coverage = []
        for gu in self.gus:
            gu_pos_3d = np.array([gu.pos[0], gu.pos[1], 0])
            dist = np.linalg.norm(uav.pos - gu_pos_3d)
            if dist < 80:
                coverage.append(gu.idx)
        logger.debug(f"UAV {uav.idx} coverage: {coverage}")
        return coverage
    
    def calculate_rates(self):
        logger.debug("Calculating data rates for all UAVs and GUs...")
        for uav in self.uavs:
            coverage = self.get_coverage(uav)
            for gu_idx in coverage:
                gu = self.gus[gu_idx]
                gu_pos_3d = np.array([gu.pos[0], gu.pos[1], 0])
                h_mi, _ = self.channel_model.get_channel(uav.pos, gu_pos_3d)
                h_norm = np.linalg.norm(h_mi)
                
                tau_z = self.config.tau_s / max(len(coverage), 1)
                gu.data_rate_a = tau_z * np.log2(1 + self.config.p_m * h_norm ** 2 / self.config.noise_power)
                gu.data_rate_b = tau_z * np.log2(1 + self.config.p_A * (self.config.Gamma_o ** 2) * (h_norm ** 4) / self.config.noise_power)
                logger.debug(f"GU {gu_idx} rates: RF={gu.data_rate_a:.4f}, backscatter={gu.data_rate_b:.4f}")
    
    def calculate_harvested_energy(self, uav, gu_idx, access_control, mode_selection):
        gu = self.gus[gu_idx]
        gu_pos_3d = np.array([gu.pos[0], gu.pos[1], 0])
        h_mi, _ = self.channel_model.get_channel(uav.pos, gu_pos_3d)
        harvested = 0.0
        
        coverage = self.get_coverage(uav)
        tau_z = self.config.tau_s / max(len(coverage), 1)
        
        for other_gu_idx in coverage:
            if other_gu_idx != gu_idx and access_control[other_gu_idx] >= 0.5 and mode_selection[other_gu_idx] < 0.5:
                w_mi = h_mi / np.linalg.norm(h_mi)
                h_flat = h_mi.flatten()
                w_flat = w_mi.flatten()
                energy = self.config.mu * self.config.p_A * tau_z * np.abs(np.dot(h_flat.conj(), w_flat)) ** 2
                harvested += energy
                logger.debug(f"GU {gu_idx} harvesting from GU {other_gu_idx}: {energy:.6f}")
        
        logger.debug(f"GU {gu_idx} total harvested energy: {harvested:.6f}")
        return harvested
    
    def get_uav_state(self, uav):
        coverage = self.get_coverage(uav)
        gu_energies = []
        gu_buffers = []
        channels = []
        
        for gu_idx in coverage:
            gu = self.gus[gu_idx]
            gu_energies.append(gu.energy)
            gu_buffers.append(gu.buffer)
            gu_pos_3d = np.array([gu.pos[0], gu.pos[1], 0])
            h_mi, _ = self.channel_model.get_channel(uav.pos, gu_pos_3d)
            channels.append(np.linalg.norm(h_mi))
        
        g_i, d_i0 = self.channel_model.get_channel(uav.pos, self.rbs.pos)
        
        state = np.array([uav.pos[0], uav.pos[1], uav.pos[2], 
                         uav.buffer, uav.energy, 
                         d_i0, np.linalg.norm(g_i)] + 
                         gu_energies + gu_buffers + channels)
        
        padded_state = np.pad(state, (0, max(0, 30 - len(state))))
        logger.debug(f"UAV {uav.idx} state shape: {padded_state.shape}")
        return padded_state
    
    def get_state(self):
        states = []
        for uav in self.uavs:
            states.append(self.get_uav_state(uav))
        states_array = np.array(states)
        logger.debug(f"get_state: shape={states_array.shape}")
        return states_array
    
    def step(self, actions):
        logger.info(f"\n=== Step: time_slot={self.time_slot} ===")
        rewards = np.zeros(self.config.N)
        step_info = {
            'per_agent': [],
            'totals': {
                'collision_events': 0,
                'collision_penalty': 0.0,
                'data_received': 0.0,
                'data_sent_to_rbs': 0.0,
                'energy_consumed': 0.0,
                'flight_energy': 0.0,
                'harvested_energy': 0.0,
                'lower_reward': 0.0,
                'upper_reward': 0.0,
            }
        }
        
        for i, uav in enumerate(self.uavs):
            action = actions[i]
            direction = action[:3]
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            speed = np.abs(action[3])
            
            logger.info(f"UAV {i} action: direction={direction[:2]}, speed={speed:.2f}")
            logger.debug(f"UAV {i} full action: dir={direction}, speed={speed}, access_idx4={action[4]:.2f}")
            
            uav.move(direction, speed)
            self.calculate_rates()

            collision_count = 0
            collision_penalty = 0.0
            for j, other_uav in enumerate(self.uavs):
                if i != j:
                    dist = np.linalg.norm(uav.pos - other_uav.pos)
                    if dist < self.config.d_min:
                        collision_count += 1
                        hard_penalty = self.config.eta * (1 - dist / self.config.d_min)
                        collision_penalty += hard_penalty
                        logger.warning(f"UAV {i} collision with UAV {j}: distance={dist:.2f} < d_min={self.config.d_min}, penalty={hard_penalty:.2f}")
                    elif dist < self.config.d_soft:
                        soft_penalty = self.config.eta_soft * (1 - dist / self.config.d_soft)
                        collision_penalty += soft_penalty
                        logger.debug(f"UAV {i} approaching UAV {j}: distance={dist:.2f} < d_soft={self.config.d_soft}, soft_penalty={soft_penalty:.2f}")
            
            if collision_count > 0:
                logger.info(f"UAV {i} collision penalty: -{self.config.eta * collision_count}")
            
            coverage = self.get_coverage(uav)
            uav.scheduled = bool(action[4])
            access_control = action[5:5+self.config.M]
            mode_selection = action[5+self.config.M:5+2*self.config.M]
            
            logger.info(f"UAV {i} coverage: {coverage}, scheduled={uav.scheduled}")
            
            tau_z = self.config.tau_s / max(len(coverage), 1)
            data_received = 0.0
            sensing_energy_consumed = 0.0
            forward_energy_consumed = 0.0
            harvested_energy_total = 0.0
            data_sent_to_rbs = 0.0
            
            for gu_idx in coverage:
                gu = self.gus[gu_idx]
                if access_control[gu_idx] >= 0.5:
                    mode = 1 if mode_selection[gu_idx] >= 0.5 else 0
                    gu.access = True
                    gu.mode = mode
                    
                    logger.debug(f"GU {gu_idx} access granted, mode={mode} (1=RF, 0=backscatter)")
                    
                    if mode == 1:
                        consumed = self.config.p_m * tau_z
                        if gu.energy >= consumed:
                            data_sent = min(gu.buffer, gu.data_rate_a)
                            harvested = self.calculate_harvested_energy(uav, gu_idx, access_control, mode_selection)
                            gu.update_energy(harvested, consumed)
                            sensing_energy_consumed += consumed
                            harvested_energy_total += harvested
                            logger.info(f"GU {gu_idx} RF mode: sent={data_sent:.4f}, consumed={consumed:.4f}, harvested={harvested:.4f}")
                        else:
                            data_sent = 0.0
                            logger.info(f"GU {gu_idx} RF mode skipped: energy={gu.energy:.4f} < consumed={consumed:.4f}")
                    else:
                        data_sent = min(gu.buffer, gu.data_rate_b)
                        consumed = self.config.p_A * tau_z
                        harvested = self.calculate_harvested_energy(uav, gu_idx, access_control, mode_selection)
                        gu.update_energy(harvested, consumed)
                        sensing_energy_consumed += consumed
                        harvested_energy_total += harvested
                        logger.info(f"GU {gu_idx} backscatter mode: sent={data_sent:.4f}, consumed={consumed:.4f}, harvested={harvested:.4f}")
                    
                    new_data = self.rng.uniform(self.config.A_min, self.config.A_max)
                    gu.update_buffer(data_sent, new_data)
                    data_received += data_sent
                else:
                    logger.debug(f"GU {gu_idx} access denied (access_control={access_control[gu_idx]:.2f})")
            
            uav.update_buffer(data_received, 0.0)
            logger.info(f"UAV {i} data received: {data_received:.4f}, sensing energy consumed: {sensing_energy_consumed:.4f}")

            flight_energy = self.config.P_0 * (1 + 3 * speed**2 / self.config.v_max**2) * self.config.tau_f

            if uav.scheduled:
                g_i, _ = self.channel_model.get_channel(uav.pos, self.rbs.pos)
                o_i = self.config.tau_d * np.log2(1 + self.config.p_i_r * np.linalg.norm(g_i) ** 2 / self.config.noise_power)
                data_sent = min(uav.buffer, o_i)
                uav.update_buffer(0.0, data_sent)
                forward_energy_consumed += self.config.p_i_r * self.config.tau_d
                data_sent_to_rbs = data_sent
                logger.info(f"UAV {i} scheduled to RBS: data_sent={data_sent:.4f}, rate={o_i:.4f}, energy={self.config.p_i_r * self.config.tau_d:.4f}")
            else:
                logger.info(f"UAV {i} not scheduled: no data forwarded to RBS")

            total_energy = flight_energy + sensing_energy_consumed + forward_energy_consumed
            total_energy = max(total_energy, self.config.denom_epsilon)

            ee_numerator = data_received + self.config.gamma_forward * data_sent_to_rbs
            ee_ratio = ee_numerator / total_energy

            energy_consumed = sensing_energy_consumed + forward_energy_consumed

            if self.config.reward_mode == 'additive':
                # Additive reward: data throughput - energy cost - collision penalty
                upper_reward = ee_numerator - self.config.eta * energy_consumed - collision_penalty
            else:
                # EE ratio reward: energy efficiency (bits/Joule) - collision penalty
                upper_reward = ee_ratio * self.config.reward_scale - collision_penalty

            lower_reward = data_received - self.config.eta1 * harvested_energy_total
            rewards[i] = upper_reward
            uav.energy -= forward_energy_consumed
            logger.info(
                f"UAV {i} rewards: upper(EE ratio)={upper_reward:.4f}, lower(sensing)={lower_reward:.4f}, "
                f"ee_ratio={ee_ratio:.4f} bits/J, flight_e={flight_energy:.4f}, "
                f"sensing_e={sensing_energy_consumed:.4f}, forward_e={forward_energy_consumed:.4f}, "
                f"harvested_e={harvested_energy_total:.6f}, total_e={total_energy:.4f}"
            )

            step_info['per_agent'].append({
                'collision_events': collision_count,
                'collision_penalty': collision_penalty,
                'data_received': data_received,
                'data_sent_to_rbs': data_sent_to_rbs,
                'sensing_energy_consumed': sensing_energy_consumed,
                'forward_energy_consumed': forward_energy_consumed,
                'flight_energy': flight_energy,
                'energy_consumed': energy_consumed,
                'harvested_energy': harvested_energy_total,
                'lower_reward': lower_reward,
                'upper_reward': upper_reward,
            })
            step_info['totals']['collision_events'] += collision_count
            step_info['totals']['collision_penalty'] += collision_penalty
            step_info['totals']['data_received'] += data_received
            step_info['totals']['data_sent_to_rbs'] += data_sent_to_rbs
            step_info['totals']['energy_consumed'] += energy_consumed
            step_info['totals']['flight_energy'] += flight_energy
            step_info['totals']['harvested_energy'] += harvested_energy_total
            step_info['totals']['lower_reward'] += lower_reward
            step_info['totals']['upper_reward'] += upper_reward
        
        self.time_slot += 1
        done = self.time_slot >= 100
        logger.info(f"Step completed: time_slot={self.time_slot}, done={done}, rewards={rewards}")
        self.last_step_info = step_info
        
        return self.get_state(), rewards, done
