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
        self.reward_scale = 50.0   # scaled to match P_0=100W, keeping EE ratio ~3-5
        self.P_0 = 100.0        # small research UAV hovering power (W), Zeng 2019
        self.denom_epsilon = 1e-6
        self.omega_0 = 1e-3
        self.boundary = 500.0          # 1km x 1km deployment area
        self.coverage_radius = 150.0   # UAV-GU communication range (m)
        self.d_soft = 10.0
        self.eta_soft = 5.0
        self.init_min_separation = 15.0
        self.dyna_k = 1
        self.dyna_warmup = 32
        self.dyna_warmup_steps = 25_600
        self.dyna_planning_batch_size = 32
        self.dyna_model_fraction = 0.25
        self.dyna_model_error_threshold = 0.20
        self.dyna_counterfactual_fraction = 0.25
        # Planning strategy: ``state_gate`` preserves the repaired Dyna
        # baseline; ``bcad`` uses Bellman-consistency trust and a gradual,
        # probabilistic planning budget.
        self.dyna_planning_strategy = 'state_gate'
        self.dyna_bellman_beta = 2.0
        self.dyna_bellman_denom_floor = 1.0
        self.dyna_plan_probability_max = 0.25
        self.dyna_plan_probability_min = 0.0
        self.dyna_plan_ramp_steps = 8_000
        self.dyna_model_update_lr_scale = 0.10
        self.dyna_robust_scale_floor = 1.0
        self.dyna_plan_keep_fraction = 0.50
        self.dyna_anchor_alpha = 0.25
        self.dyna_anchor_clip = 1.0
        self.dyna_priority_candidate_multiplier = 4
        # ``legacy`` stores the pre-upper state as the lower transition target.
        # New decision-consistent experiments delay insertion until the next
        # post-move lower decision state is available.
        self.lower_transition_mode = 'legacy'
        # ee_ratio: received data + weighted RBS forwarding (historical default)
        # paper_xi: RBS-delivered data / total UAV energy (paper objective)
        # additive: throughput - communication energy (legacy ablation)
        self.reward_mode = 'ee_ratio'

        # M4: state dimension derived from per-RB channel info
        # pos(3) + buffer(1) + energy(1) + d_i0(1) + g_i per RB(F) + M*(energy(1)+buffer(1)+channel per RB(F))
        self.state_dim = 7 + self.F + self.M * (3 + self.F)

        # M7: target network update mode ('soft' or 'hard'); paper uses hard replace every 100 iters
        # 默认soft以保持各算法基准一致；复现论文设为'hard'即可启用周期硬替换
        self.target_update_mode = 'soft'
        self.hard_update_every = 100

        # M8: lower-layer exploration mode ('fixed' or 'decay'); paper uses fixed eps=0.05
        self.epsilon_mode = 'fixed'
        self.epsilon_fixed = 0.05

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
        logger.info(f"Spatial params: boundary={self.boundary}m, coverage_radius={self.coverage_radius}m")
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
        los_component = los_component / np.linalg.norm(los_component)  # LOS: deterministic unit-norm
        nlos_component = self.rng.standard_normal((1, self.config.F)) + 1j * self.rng.standard_normal((1, self.config.F))
        # M6: NLOS分量保持CN(0,1)不再归一化，使||h||真实服从Rician分布
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
        new_pos = old_pos + self.velocity * self.config.tau_f
        # Geofence: clamp to operational area
        b = self.config.boundary
        new_pos[0] = np.clip(new_pos[0], -b, b)
        new_pos[1] = np.clip(new_pos[1], -b, b)
        new_pos[2] = np.clip(new_pos[2], 10.0, 150.0)
        if not np.allclose(old_pos + self.velocity * self.config.tau_f, new_pos):
            logger.debug(f"UAV {self.idx} geofence clamped to: {new_pos}")
        self.pos = new_pos
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
        self._channel_cache = None
        self._rate_a = np.zeros((config.N, config.M), dtype=float)
        self._rate_b = np.zeros((config.N, config.M), dtype=float)
        self._pending_step = None
        
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
        self._pending_step = None
        self._refresh_channel_cache()
        self.calculate_rates()
        logger.info(f"Environment reset completed, time_slot={self.time_slot}")
        
        return self.get_state()

    def _refresh_channel_cache(self):
        """Draw each physical link once and reuse it throughout one time slot."""
        rbs_channels = np.zeros((self.config.N, self.config.F), dtype=complex)
        rbs_distances = np.zeros(self.config.N, dtype=float)
        gu_channels = np.zeros((self.config.N, self.config.M, self.config.F), dtype=complex)
        gu_distances = np.zeros((self.config.N, self.config.M), dtype=float)
        for i, uav in enumerate(self.uavs):
            channel, distance = self.channel_model.get_channel(uav.pos, self.rbs.pos)
            rbs_channels[i] = channel.reshape(-1)
            rbs_distances[i] = distance
            for m, gu in enumerate(self.gus):
                gu_pos = np.array([gu.pos[0], gu.pos[1], 0.0])
                channel, distance = self.channel_model.get_channel(uav.pos, gu_pos)
                gu_channels[i, m] = channel.reshape(-1)
                gu_distances[i, m] = distance
        self._channel_cache = {
            'rbs_channels': rbs_channels,
            'rbs_distances': rbs_distances,
            'gu_channels': gu_channels,
            'gu_distances': gu_distances,
        }

    def _ensure_channel_cache(self):
        if self._channel_cache is None:
            self._refresh_channel_cache()
    
    def get_coverage(self, uav):
        coverage = []
        for gu in self.gus:
            gu_pos_3d = np.array([gu.pos[0], gu.pos[1], 0])
            dist = np.linalg.norm(uav.pos - gu_pos_3d)
            if dist < self.config.coverage_radius:
                coverage.append(gu.idx)
        logger.debug(f"UAV {uav.idx} coverage: {coverage}")
        return coverage
    
    def calculate_rates(self):
        logger.debug("Calculating data rates for all UAVs and GUs...")
        self._ensure_channel_cache()
        self._rate_a.fill(0.0)
        self._rate_b.fill(0.0)
        for i, uav in enumerate(self.uavs):
            coverage = self.get_coverage(uav)
            for gu_idx in coverage:
                h_mi = self._channel_cache['gu_channels'][i, gu_idx]
                h_norm = np.linalg.norm(h_mi)
                tau_z = self.config.tau_s / max(len(coverage), 1)
                self._rate_a[i, gu_idx] = tau_z * np.log2(
                    1 + self.config.p_m * h_norm ** 2 / self.config.noise_power)
                self._rate_b[i, gu_idx] = tau_z * np.log2(
                    1 + self.config.p_A * self.config.Gamma_o ** 2 * h_norm ** 4
                    / (2 * self.config.noise_power))
    
    def calculate_harvested_energy(self, uav, gu_idx, access_control, mode_selection):
        self._ensure_channel_cache()
        harvested = 0.0

        coverage = self.get_coverage(uav)
        tau_z = self.config.tau_s / max(len(coverage), 1)

        for other_gu_idx in coverage:
            if other_gu_idx != gu_idx and access_control[other_gu_idx] >= 0.5 and mode_selection[other_gu_idx] < 0.5:
                h_ni = self._channel_cache['gu_channels'][uav.idx, other_gu_idx]
                w_ni = h_ni / max(np.linalg.norm(h_ni), 1e-12)
                h_flat = h_ni.flatten()
                w_flat = w_ni.flatten()
                energy = self.config.mu * self.config.p_A * tau_z * np.abs(np.dot(h_flat.conj(), w_flat)) ** 2
                harvested += energy
                logger.debug(f"GU {gu_idx} harvesting from GU {other_gu_idx}: {energy:.6f}")

        logger.debug(f"GU {gu_idx} total harvested energy: {harvested:.6f}")
        return harvested
    
    def get_uav_state(self, uav):
        self._ensure_channel_cache()
        coverage = self.get_coverage(uav)
        # M4: 保留每RB信道幅度(实数)，而非压缩为单一范数，以支持频域选择性调度
        state_list = [float(uav.pos[0]), float(uav.pos[1]), float(uav.pos[2]),
                      float(uav.buffer), float(uav.energy), float(uav.scheduled)]

        g_i = self._channel_cache['rbs_channels'][uav.idx]
        d_i0 = self._channel_cache['rbs_distances'][uav.idx]
        state_list.append(float(d_i0))
        state_list.extend(np.abs(g_i).flatten().tolist())

        for gu_idx in range(self.config.M):
            gu = self.gus[gu_idx]
            is_covered = gu_idx in coverage
            state_list.append(float(is_covered))
            state_list.append(float(gu.energy) if is_covered else 0.0)
            state_list.append(float(gu.buffer) if is_covered else 0.0)
            h_mi = self._channel_cache['gu_channels'][uav.idx, gu_idx]
            state_list.extend((np.abs(h_mi) if is_covered else np.zeros(self.config.F)).tolist())

        state = np.array(state_list, dtype=float)
        if state.size != self.config.state_dim:
            raise RuntimeError(f"state_dim mismatch: built {state.size}, configured {self.config.state_dim}")
        logger.debug(f"UAV {uav.idx} state shape: {state.shape}")
        return state
    
    def get_state(self):
        states = []
        for uav in self.uavs:
            states.append(self.get_uav_state(uav))
        states_array = np.array(states)
        logger.debug(f"get_state: shape={states_array.shape}")
        return states_array
    
    def get_lower_action_masks(self):
        """Return valid [no-access, active-RF, backscatter] choices per UAV/GU."""
        masks = np.zeros((self.config.N, self.config.M, 3), dtype=bool)
        masks[:, :, 0] = True
        for i, uav in enumerate(self.uavs):
            coverage = self.get_coverage(uav)
            tau_z = self.config.tau_s / max(len(coverage), 1)
            for m in coverage:
                masks[i, m, 1] = self.gus[m].energy >= self.config.p_m * tau_z
                masks[i, m, 2] = True
        return masks

    def prepare_step(self, upper_actions):
        """Apply all mobility decisions, then expose the post-move lower-layer state."""
        if self._pending_step is not None:
            raise RuntimeError("complete_step must be called before preparing another step")
        upper_actions = np.asarray(upper_actions, dtype=float)
        if upper_actions.shape != (self.config.N, 5):
            raise ValueError(f"upper_actions must have shape {(self.config.N, 5)}, got {upper_actions.shape}")

        speeds = np.zeros(self.config.N, dtype=float)
        for i, uav in enumerate(self.uavs):
            direction = upper_actions[i, :3]
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            speeds[i] = np.clip(abs(upper_actions[i, 3]), 0.0, 1.0) * self.config.v_max
            uav.move(direction, speeds[i])

        schedule_scores = np.clip(upper_actions[:, 4], 0.0, 1.0)
        eligible = np.flatnonzero(schedule_scores >= 0.5)
        scheduled_idx = int(eligible[np.argmax(schedule_scores[eligible])]) if eligible.size else None
        for i, uav in enumerate(self.uavs):
            uav.scheduled = i == scheduled_idx

        collision_counts = np.zeros(self.config.N, dtype=int)
        collision_penalties = np.zeros(self.config.N, dtype=float)
        for i in range(self.config.N):
            for j in range(i + 1, self.config.N):
                distance = np.linalg.norm(self.uavs[i].pos - self.uavs[j].pos)
                if distance < self.config.d_min:
                    penalty = self.config.eta * (1 - distance / self.config.d_min)
                    collision_counts[i] += 1
                    collision_counts[j] += 1
                elif distance < self.config.d_soft:
                    penalty = self.config.eta_soft * (1 - distance / self.config.d_soft)
                else:
                    penalty = 0.0
                collision_penalties[i] += penalty
                collision_penalties[j] += penalty

        self._refresh_channel_cache()
        self.calculate_rates()
        self._pending_step = {
            'speeds': speeds,
            'collision_counts': collision_counts,
            'collision_penalties': collision_penalties,
        }
        return self.get_state()

    def complete_step(self, lower_actions):
        """Resolve joint GU access, sensing, forwarding, rewards, and slot arrivals."""
        if self._pending_step is None:
            raise RuntimeError("prepare_step must be called before complete_step")
        lower_actions = np.asarray(lower_actions, dtype=float)
        expected_shape = (self.config.N, 2 * self.config.M)
        if lower_actions.shape != expected_shape:
            raise ValueError(f"lower_actions must have shape {expected_shape}, got {lower_actions.shape}")

        access = lower_actions[:, :self.config.M]
        modes = lower_actions[:, self.config.M:]
        masks = self.get_lower_action_masks()
        owners = np.full(self.config.M, -1, dtype=int)
        for m in range(self.config.M):
            candidates = [i for i in range(self.config.N)
                          if access[i, m] >= 0.5 and masks[i, m, 2]]
            if candidates:
                owners[m] = max(
                    candidates,
                    key=lambda i: np.linalg.norm(self._channel_cache['gu_channels'][i, m]))

        for gu in self.gus:
            gu.access = False
            gu.mode = 0

        metrics = []
        sent_by_gu = np.zeros(self.config.M, dtype=float)
        for i, uav in enumerate(self.uavs):
            coverage = self.get_coverage(uav)
            tau_z = self.config.tau_s / max(len(coverage), 1)
            data_received = sensing_energy = harvested_total = 0.0
            for m in np.flatnonzero(owners == i):
                gu = self.gus[m]
                mode = 1 if modes[i, m] >= 0.5 and masks[i, m, 1] else 0
                gu.access = True
                gu.mode = mode
                harvested = self.calculate_harvested_energy(uav, m, access[i], modes[i])
                if mode == 1:
                    consumed = self.config.p_m * tau_z
                    data_sent = min(gu.buffer, self._rate_a[i, m])
                else:
                    consumed = self.config.p_A * tau_z
                    data_sent = min(gu.buffer, self._rate_b[i, m])
                gu.update_energy(harvested, consumed)
                sent_by_gu[m] = data_sent
                data_received += data_sent
                sensing_energy += consumed
                harvested_total += harvested

            uav.update_buffer(data_received, 0.0)
            forward_energy = data_to_rbs = 0.0
            if uav.scheduled:
                channel = self._channel_cache['rbs_channels'][i]
                rate = self.config.tau_d * np.log2(
                    1 + self.config.p_i_r * np.linalg.norm(channel) ** 2 / self.config.noise_power)
                data_to_rbs = min(uav.buffer, rate)
                uav.update_buffer(0.0, data_to_rbs)
                forward_energy = self.config.p_i_r * self.config.tau_d
                uav.energy -= forward_energy

            speed = self._pending_step['speeds'][i]
            flight_energy = self.config.P_0 * (
                1 + 3 * speed ** 2 / self.config.v_max ** 2) * self.config.tau_f
            total_energy = max(flight_energy + sensing_energy + forward_energy,
                               self.config.denom_epsilon)
            numerator = data_received + self.config.gamma_forward * data_to_rbs
            ee_ratio = numerator / total_energy
            paper_xi = data_to_rbs / total_energy
            collision_penalty = self._pending_step['collision_penalties'][i]
            if self.config.reward_mode == 'additive':
                upper_reward = numerator - self.config.eta * (sensing_energy + forward_energy) - collision_penalty
            elif self.config.reward_mode == 'paper_xi':
                upper_reward = paper_xi * self.config.reward_scale - collision_penalty
            elif self.config.reward_mode == 'ee_ratio':
                upper_reward = ee_ratio * self.config.reward_scale - collision_penalty
            else:
                raise ValueError(f"Unsupported reward_mode: {self.config.reward_mode}")
            lower_reward = data_received - self.config.eta1 * harvested_total
            metrics.append({
                'collision_events': int(self._pending_step['collision_counts'][i]),
                'collision_penalty': float(collision_penalty),
                'data_received': float(data_received),
                'data_sent_to_rbs': float(data_to_rbs),
                'sensing_energy_consumed': float(sensing_energy),
                'forward_energy_consumed': float(forward_energy),
                'flight_energy': float(flight_energy),
                'energy_consumed': float(sensing_energy + forward_energy),
                'harvested_energy': float(harvested_total),
                'paper_xi': float(paper_xi),
                'lower_reward': float(lower_reward),
                'upper_reward': float(upper_reward),
            })

        # Every GU receives a stochastic arrival once per slot, whether scheduled or not.
        for m, gu in enumerate(self.gus):
            gu.update_buffer(sent_by_gu[m], self.rng.uniform(self.config.A_min, self.config.A_max))

        rewards = np.asarray([item['upper_reward'] for item in metrics], dtype=float)
        total_keys = ('collision_events', 'collision_penalty', 'data_received',
                      'data_sent_to_rbs', 'energy_consumed', 'flight_energy',
                      'harvested_energy', 'lower_reward', 'upper_reward')
        self.last_step_info = {
            'per_agent': metrics,
            'totals': {key: sum(item[key] for item in metrics) for key in total_keys},
        }
        self.time_slot += 1
        done = self.time_slot >= 200
        self._pending_step = None
        return self.get_state(), rewards, done

    def step(self, actions):
        """Compatibility entry point for flat agents using the concatenated action."""
        actions = np.asarray(actions, dtype=float)
        self.prepare_step(actions[:, :5])
        return self.complete_step(actions[:, 5:5 + 2 * self.config.M])

    def _legacy_step(self, actions):
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

        # M5: 每时隙每个GU最多被一个UAV接入，避免多UAV重复计入同一份数据
        served_gus = set()

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
            uav.scheduled = action[4] >= 0.5  # M2: 阈值判定替代bool()，避免action[4]近0时误判
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
                if gu_idx in served_gus:
                    logger.debug(f"GU {gu_idx} already served by another UAV this slot, skip")
                    continue
                if access_control[gu_idx] >= 0.5:
                    served_gus.add(gu_idx)  # M5: 锁定该GU，后续UAV不再重复接入
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
            paper_xi = data_sent_to_rbs / total_energy

            energy_consumed = sensing_energy_consumed + forward_energy_consumed

            if self.config.reward_mode == 'additive':
                # Additive reward: data throughput - energy cost - collision penalty
                upper_reward = ee_numerator - self.config.eta * energy_consumed - collision_penalty
            elif self.config.reward_mode == 'paper_xi':
                # Paper objective: data delivered to RBS per total UAV energy.
                upper_reward = paper_xi * self.config.reward_scale - collision_penalty
            elif self.config.reward_mode == 'ee_ratio':
                # EE ratio reward: energy efficiency (bits/Joule) - collision penalty
                upper_reward = ee_ratio * self.config.reward_scale - collision_penalty
            else:
                raise ValueError(f"Unsupported reward_mode: {self.config.reward_mode}")

            lower_reward = data_received - self.config.eta1 * harvested_energy_total
            rewards[i] = upper_reward
            uav.energy -= forward_energy_consumed
            logger.info(
                f"UAV {i} rewards: mode={self.config.reward_mode}, upper={upper_reward:.4f}, "
                f"lower(sensing)={lower_reward:.4f}, ee_ratio={ee_ratio:.4f}, paper_xi={paper_xi:.4f}, "
                f"flight_e={flight_energy:.4f}, "
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
                'paper_xi': paper_xi,
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
        done = self.time_slot >= 200  # 200 steps × 7.5m = 1500m > 1414m diagonal at boundary=500
        logger.info(f"Step completed: time_slot={self.time_slot}, done={done}, rewards={rewards}")
        self.last_step_info = step_info
        
        return self.get_state(), rewards, done
