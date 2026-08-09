import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from collections import deque
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.propagate = False

log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)

file_handler = RotatingFileHandler(
    os.path.join(log_dir, 'hierarchical_agent.log'),
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


def _lower_action_masks_from_states(states, config):
    """Build [no-access, RF, backscatter] masks from fixed-slot state tensors."""
    base, stride = 7 + config.F, 3 + config.F
    covered = torch.stack(
        [states[:, base + j * stride] >= 0.5 for j in range(config.M)], dim=1)
    energies = torch.stack(
        [states[:, base + j * stride + 1] for j in range(config.M)], dim=1)
    covered_count = covered.sum(dim=1, keepdim=True).clamp(min=1)
    rf_required = config.p_m * config.tau_s / covered_count
    return torch.stack([
        torch.ones_like(covered),
        covered & (energies >= rf_required),
        covered,
    ], dim=2)


def _build_state_scale(config, state_dim):
    scale = [config.boundary, config.boundary, 100.0,
             5000.0, 100.0, 1.0, 1500.0]
    scale.extend([1e-3] * config.F)
    for _ in range(config.M):
        scale.extend([1.0, config.E_max, 5000.0])
        scale.extend([1e-3] * config.F)
    if len(scale) != state_dim:
        raise ValueError(f"state scale has {len(scale)} values, expected {state_dim}")
    return np.asarray(scale, dtype=np.float32)


def _linear_noise_std(initial, final, episode, decay_episodes):
    decay_episodes = max(1, int(decay_episodes))
    progress = np.clip(float(episode) / decay_episodes, 0.0, 1.0)
    return float(initial + (final - initial) * progress)


def _upper_noise_stds(config, episode):
    decay = getattr(config, 'upper_noise_decay_episodes', 150)
    return {
        'direction': _linear_noise_std(
            getattr(config, 'upper_direction_noise_initial', 0.15),
            getattr(config, 'upper_direction_noise_final', 0.02), episode, decay),
        'speed': _linear_noise_std(
            getattr(config, 'upper_speed_noise_initial', 0.10),
            getattr(config, 'upper_speed_noise_final', 0.02), episode, decay),
        'schedule': _linear_noise_std(
            getattr(config, 'upper_schedule_noise_initial', 0.05),
            getattr(config, 'upper_schedule_noise_final', 0.01), episode, decay),
    }


def _state_scale_tensor(state_dim, state_scale):
    if state_scale is None:
        return torch.ones(state_dim, dtype=torch.float32)
    scale = torch.as_tensor(state_scale, dtype=torch.float32)
    if scale.numel() != state_dim or torch.any(scale <= 0):
        raise ValueError(f"state_scale must contain {state_dim} positive values")
    return scale


def _unpack_upper_transition(transition):
    if len(transition) == 5:
        state, commanded, reward, next_state, done = transition
        return state, commanded, commanded, reward, next_state, done
    if len(transition) == 6:
        return transition
    raise ValueError(f"Unsupported upper replay transition length: {len(transition)}")


def _projection_residual_loss(
        current_actions, commanded_actions, executed_actions, tolerance):
    mobility_delta = executed_actions[:, :4] - commanded_actions[:, :4]
    intervention_mask = torch.linalg.vector_norm(
        mobility_delta, dim=1) > tolerance
    if not torch.any(intervention_mask):
        return current_actions[:, :4].sum() * 0.0

    current_mobility = current_actions[intervention_mask, :4]
    correction = mobility_delta[intervention_mask]
    target = current_mobility.detach() + correction
    target[:, :3] = target[:, :3].clamp(-1.0, 1.0)
    target[:, 3] = target[:, 3].clamp(-1.0, 1.0)
    return F.mse_loss(current_mobility, target)

class UpperActor(nn.Module):
    def __init__(self, state_dim, continuous_dim, discrete_dim, hidden_dim=64,
                 state_scale=None):
        super(UpperActor, self).__init__()
        logger.info(f"Creating UpperActor: state_dim={state_dim}, continuous_dim={continuous_dim}, discrete_dim={discrete_dim}, hidden_dim={hidden_dim}")
        self.continuous_dim = continuous_dim
        self.discrete_dim = discrete_dim
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_cont = nn.Linear(hidden_dim, continuous_dim)
        self.fc3_disc = nn.Linear(hidden_dim, discrete_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.register_buffer(
            'state_scale', _state_scale_tensor(state_dim, state_scale),
            persistent=False)

    def forward(self, x):
        x = x / self.state_scale
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        cont = self.tanh(self.fc3_cont(x))
        disc = self.sigmoid(self.fc3_disc(x))
        return torch.cat([cont, disc], dim=-1)

class UpperCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, state_scale=None):
        super(UpperCritic, self).__init__()
        logger.info(f"Creating UpperCritic: state_dim={state_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}")
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.register_buffer(
            'state_scale', _state_scale_tensor(state_dim, state_scale),
            persistent=False)
    
    def forward(self, x, a):
        x = x / self.state_scale
        x = torch.cat([x, a], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LowerDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, state_scale=None):
        super(LowerDQN, self).__init__()
        logger.info(f"Creating LowerDQN: state_dim={state_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}")
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.register_buffer(
            'state_scale', _state_scale_tensor(state_dim, state_scale),
            persistent=False)
    
    def forward(self, x):
        x = x / self.state_scale
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Model(nn.Module):
    def __init__(self, state_dim, action_dim, state_scale, hidden_dim=64):
        super(Model, self).__init__()
        logger.info(f"Creating Model: state_dim={state_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}")
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_reward = nn.Linear(hidden_dim, 1)
        self.fc_next_state = nn.Linear(hidden_dim, state_dim)
        self.relu = nn.ReLU()
        self.register_buffer('state_scale', torch.as_tensor(state_scale, dtype=torch.float32))
    
    def forward(self, state, action):
        x = torch.cat([state / self.state_scale, action], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        reward = self.fc_reward(x)
        next_state = state + self.fc_next_state(x) * self.state_scale
        return reward, next_state

class HierarchicalAgent:
    def __init__(self, state_dim, action_dim, num_agents, config, dyna_k=None,
                 planning_mode=None):
        logger.info("=" * 60)
        logger.info("Initializing HierarchicalAgent...")
        logger.info("=" * 60)

        self.config = config
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_scale = _build_state_scale(config, state_dim)
        self.global_state_scale = np.tile(self.state_scale, num_agents)
        self.upper_projection_loss_weight = max(
            0.0, float(getattr(config, 'upper_projection_loss_weight', 1.0)))
        self.projection_tolerance = max(
            0.0, float(getattr(config, 'safety_projection_tolerance', 1e-6)))
        self.last_upper_update_info = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(config.torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.torch_seed)
        self.action_rng = config.rngs['action']
        self.replay_rng = config.rngs['replay']
        self.upper_noise_episode = 0
        self.model_rng = config.rngs['model']
        self.dyna_rng = config.rngs['dyna']
        self.dyna_k = config.dyna_k if dyna_k is None else dyna_k
        self.dyna_warmup = max(0, int(getattr(config, 'dyna_warmup', 32)))
        self.dyna_warmup_steps = max(0, int(getattr(config, 'dyna_warmup_steps', 0)))
        self.dyna_planning_batch_size = max(1, int(getattr(config, 'dyna_planning_batch_size', 32)))
        self.dyna_model_fraction = float(getattr(config, 'dyna_model_fraction', 0.25))
        self.dyna_model_error_threshold = float(getattr(config, 'dyna_model_error_threshold', 0.20))
        self.dyna_counterfactual_fraction = float(getattr(config, 'dyna_counterfactual_fraction', 0.25))
        self.dyna_planning_strategy = getattr(config, 'dyna_planning_strategy', 'state_gate')
        if self.dyna_planning_strategy not in (
                'state_gate', 'bcad', 'robust_budget', 'real_anchor',
                'utility_priority'):
            raise ValueError(f"Unsupported dyna_planning_strategy: {self.dyna_planning_strategy}")
        self.dyna_bellman_beta = max(0.0, float(getattr(config, 'dyna_bellman_beta', 2.0)))
        self.dyna_bellman_denom_floor = max(
            1e-6, float(getattr(config, 'dyna_bellman_denom_floor', 1.0)))
        self.dyna_plan_probability_max = float(np.clip(
            getattr(config, 'dyna_plan_probability_max', 0.25), 0.0, 1.0))
        self.dyna_plan_probability_min = float(np.clip(
            getattr(config, 'dyna_plan_probability_min', 0.0), 0.0,
            self.dyna_plan_probability_max))
        self.dyna_plan_ramp_steps = max(1, int(getattr(config, 'dyna_plan_ramp_steps', 8_000)))
        self.dyna_model_update_lr_scale = float(np.clip(
            getattr(config, 'dyna_model_update_lr_scale', 0.10), 0.0, 1.0))
        self.dyna_robust_scale_floor = max(
            1e-6, float(getattr(config, 'dyna_robust_scale_floor', 1.0)))
        self.dyna_plan_keep_fraction = float(np.clip(
            getattr(config, 'dyna_plan_keep_fraction', 0.50), 1e-3, 1.0))
        self.dyna_anchor_alpha = float(np.clip(
            getattr(config, 'dyna_anchor_alpha', 0.25), 0.0, 1.0))
        self.dyna_anchor_clip = max(
            0.0, float(getattr(config, 'dyna_anchor_clip', 1.0)))
        self.dyna_priority_candidate_multiplier = max(
            1, int(getattr(config, 'dyna_priority_candidate_multiplier', 4)))
        self.real_steps = np.zeros(num_agents, dtype=np.int64)
        self.model_error_ema = np.full(num_agents, np.inf, dtype=float)
        self.bellman_error_ema = np.full(num_agents, np.inf, dtype=float)
        self.last_plan_info = [None for _ in range(num_agents)]
        # Replay-1 (attribution) planning mode:
        #   'model'  — use the learned world model for the extra single-sample
        #              Q-update (DynaQ-1, historical default).
        #   'replay' — use the true (r, s') stored in the replay tuple (perfect
        #              oracle control). Sampling, gating and update form are
        #              identical; only the target source differs.
        mode = getattr(config, 'planning_mode', 'model') if planning_mode is None else planning_mode
        if mode not in ('model', 'replay'):
            raise ValueError(f"Unsupported planning_mode: {mode}")
        self.planning_mode = mode
        # Zero-cost online model-error instrumentation ('model' mode only):
        # per planning sample, record |r_hat - r| and ||s_hat' - s'||
        # (the true values live in the same replay tuple).
        self.planning_errors = {
            i: {'count': 0, 'reward_err': 0.0, 'state_err': 0.0}
            for i in range(num_agents)
        }

        logger.info(f"Hierarchical params: num_agents={num_agents}, state_dim={state_dim}, action_dim={action_dim}, device={self.device}")

        logger.info("Creating upper-layer actor networks...")
        # M9: 前4维(direction+speed)tanh，第5维(scheduled)sigmoid
        self.upper_actors = [
            UpperActor(state_dim, 4, 1, state_scale=self.state_scale).to(self.device)
            for _ in range(num_agents)]

        logger.info("Creating upper-layer critic networks...")
        self.upper_critics = [
            UpperCritic(state_dim * num_agents, 5 * num_agents,
                        state_scale=self.global_state_scale).to(self.device)
            for _ in range(num_agents)]

        logger.info("Creating upper-layer target actor networks...")
        self.target_upper_actors = [
            UpperActor(state_dim, 4, 1, state_scale=self.state_scale).to(self.device)
            for _ in range(num_agents)]

        logger.info("Creating upper-layer target critic networks...")
        self.target_upper_critics = [
            UpperCritic(state_dim * num_agents, 5 * num_agents,
                        state_scale=self.global_state_scale).to(self.device)
            for _ in range(num_agents)]

        logger.info("Copying upper-layer weights to target networks...")
        for i in range(num_agents):
            self.target_upper_actors[i].load_state_dict(self.upper_actors[i].state_dict())
            self.target_upper_critics[i].load_state_dict(self.upper_critics[i].state_dict())

        logger.info("Creating upper-layer optimizers...")
        self.upper_actor_optimizers = [optim.Adam(self.upper_actors[i].parameters(), lr=1e-4) for i in range(num_agents)]
        self.upper_critic_optimizers = [optim.Adam(self.upper_critics[i].parameters(), lr=1e-3) for i in range(num_agents)]

        logger.info("Creating lower-layer DQN networks...")
        self.lower_dqns = [
            LowerDQN(state_dim, 3 * config.M, state_scale=self.state_scale).to(self.device)
            for _ in range(num_agents)]

        logger.info("Creating lower-layer target DQN networks...")
        self.target_lower_dqns = [
            LowerDQN(state_dim, 3 * config.M, state_scale=self.state_scale).to(self.device)
            for _ in range(num_agents)]
        
        logger.info("Copying lower-layer weights to target networks...")
        for i in range(num_agents):
            self.target_lower_dqns[i].load_state_dict(self.lower_dqns[i].state_dict())
        
        logger.info("Creating lower-layer optimizers...")
        self.lower_optimizers = [optim.Adam(self.lower_dqns[i].parameters(), lr=1e-4) for i in range(num_agents)]
        
        if self.dyna_k > 0:
            logger.info("Creating Dyna-Q model networks...")
            self.models = [Model(state_dim, 2 * config.M, self.state_scale).to(self.device)
                           for _ in range(num_agents)]
            
            logger.info("Creating model optimizers...")
            self.model_optimizers = [optim.Adam(self.models[i].parameters(), lr=1e-4) for i in range(num_agents)]
        else:
            self.models = []
            self.model_optimizers = []

        logger.info("Creating learning rate schedulers...")
        self.upper_actor_schedulers = [optim.lr_scheduler.StepLR(self.upper_actor_optimizers[i], step_size=500, gamma=0.9) for i in range(num_agents)]
        self.upper_critic_schedulers = [optim.lr_scheduler.StepLR(self.upper_critic_optimizers[i], step_size=500, gamma=0.9) for i in range(num_agents)]
        self.lower_schedulers = [optim.lr_scheduler.StepLR(self.lower_optimizers[i], step_size=500, gamma=0.9) for i in range(num_agents)]
        if self.dyna_k > 0:
            self.model_schedulers = [optim.lr_scheduler.StepLR(self.model_optimizers[i], step_size=500, gamma=0.9) for i in range(num_agents)]
        else:
            self.model_schedulers = []
        
        logger.info("Initializing upper-layer replay memory...")
        self.upper_memory = deque(maxlen=10000)
        
        logger.info("Initializing lower-layer replay memories...")
        self.lower_memory = [deque(maxlen=10000) for _ in range(num_agents)]
        
        self.gamma = 0.95
        self.tau = 0.01
        self.batch_size = 32
        # M8: 下层探索模式('fixed'固定/'decay'衰减)，论文用固定eps=0.05
        self.epsilon_mode = getattr(config, 'epsilon_mode', 'fixed')
        self.epsilon_fixed = getattr(config, 'epsilon_fixed', 0.05)
        self.epsilon = self.epsilon_fixed if self.epsilon_mode == 'fixed' else 0.1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.clip_norm = 1.0
        # M7: 目标网络更新模式开关
        self.target_update_mode = getattr(config, 'target_update_mode', 'soft')
        self.hard_update_every = getattr(config, 'hard_update_every', 100)
        self._hard_counter = 0

        logger.info(f"Upper memory capacity: {self.upper_memory.maxlen}")
        logger.info(f"Lower memory capacity: {self.lower_memory[0].maxlen} per agent")
        logger.info(f"gamma={self.gamma}, tau={self.tau}, batch_size={self.batch_size}, epsilon={self.epsilon}, epsilon_mode={self.epsilon_mode}")
        logger.info(f"epsilon_min={self.epsilon_min}, epsilon_decay={self.epsilon_decay}, clip_norm={self.clip_norm}")
        logger.info(
            f"dyna_k={self.dyna_k}, dyna_warmup={self.dyna_warmup}, "
            f"planning_strategy={self.dyna_planning_strategy}")
        logger.info("=" * 60)
        logger.info("HierarchicalAgent initialization complete!")
        logger.info("=" * 60)

    def _build_state_scale(self):
        """Feature scales shared by policy, value, DQN, and world-model networks."""
        return _build_state_scale(self.config, self.state_dim)
    
    def upper_act(self, states, noise=True):
        logger.debug(f"upper_act() called: states shape={states.shape}, noise={noise}")
        actions = []
        
        for i in range(self.num_agents):
            state = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
            action = self.upper_actors[i](state).detach().cpu().numpy()[0]
            
            if noise:
                stds = _upper_noise_stds(
                    self.config, self.upper_noise_episode)
                noise_val = np.empty_like(action)
                noise_val[:3] = self.action_rng.normal(
                    0, stds['direction'], size=3)
                noise_val[3] = self.action_rng.normal(0, stds['speed'])
                noise_val[4] = self.action_rng.normal(0, stds['schedule'])
                action += noise_val
                logger.debug(f"Upper Agent {i} action with noise: noise_norm={np.linalg.norm(noise_val):.4f}")
            
            action[:4] = np.clip(action[:4], -1, 1)
            action[4] = np.clip(action[4], 0, 1)
            
            logger.debug(f"Upper Agent {i} action: direction={action[:3]}, speed={action[3]:.4f}, scheduled={bool(action[4])}")
            actions.append(action)
        
        actions_array = np.array(actions)
        logger.debug(f"upper_act() completed: actions shape={actions_array.shape}")
        return actions_array
    
    def lower_act(self, states, action_masks=None, explore=True):
        logger.debug(f"lower_act() called: states shape={states.shape}, epsilon={self.epsilon}")
        actions = []
        M = self.config.M

        for i in range(self.num_agents):
            state = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)

            if explore and self.action_rng.random() < self.epsilon:
                # Random: pick one of {0,1,2} per GU, encode to 2M
                if action_masks is None:
                    action_indices = self.action_rng.integers(0, 3, size=M)
                else:
                    action_indices = np.array([
                        self.action_rng.choice(np.flatnonzero(action_masks[i, j]))
                        for j in range(M)
                    ])
                action = np.zeros(2 * M)
                for j, idx in enumerate(action_indices):
                    if idx == 0:      # no access
                        action[j] = 0; action[M + j] = 0
                    elif idx == 1:    # RF
                        action[j] = 1; action[M + j] = 1
                    else:             # Backscatter
                        action[j] = 1; action[M + j] = 0
                logger.debug(f"Lower Agent {i} exploration action: random choice")
            else:
                q_values = self.lower_dqns[i](state).detach().cpu().numpy()[0]  # (3M,)
                action = np.zeros(2 * M)
                for j in range(M):
                    q_slice = q_values[j * 3 : j * 3 + 3]  # Q for {no_access, RF, BS}
                    if action_masks is not None:
                        q_slice = np.where(action_masks[i, j], q_slice, -np.inf)
                    idx = int(np.argmax(q_slice))
                    if idx == 0:      # no access
                        action[j] = 0; action[M + j] = 0
                    elif idx == 1:    # RF
                        action[j] = 1; action[M + j] = 1
                    else:             # Backscatter
                        action[j] = 1; action[M + j] = 0
                logger.debug(f"Lower Agent {i} exploitation action: Q-values computed")

            logger.debug(f"Lower Agent {i} action: access={action[:M]}, mode={action[M:]}")
            actions.append(action)

        actions_array = np.array(actions)
        logger.debug(f"lower_act() completed: actions shape={actions_array.shape}")
        return actions_array
    
    def add_upper_memory(self, states, actions, rewards, next_states, dones,
                         executed_actions=None):
        logger.debug(f"add_upper_memory() called: states shape={states.shape}, rewards={rewards}, dones={dones}")
        commanded = np.asarray(actions, dtype=float).copy()
        executed = (commanded.copy() if executed_actions is None else
                    np.asarray(executed_actions, dtype=float).copy())
        if executed.shape != commanded.shape:
            raise ValueError(
                f"executed_actions shape {executed.shape} != commanded shape {commanded.shape}")
        self.upper_memory.append(
            (states, commanded, executed, rewards, next_states, dones))
        logger.debug(f"Upper memory size: {len(self.upper_memory)}/{self.upper_memory.maxlen}")
    
    def add_lower_memory(self, agent_idx, state, action, reward, next_state, done):
        logger.debug(f"add_lower_memory() called: agent_idx={agent_idx}, reward={reward:.4f}, done={done}")
        self.lower_memory[agent_idx].append((state, action, reward, next_state, done))
        self.real_steps[agent_idx] += 1
        logger.debug(f"Lower memory[{agent_idx}] size: {len(self.lower_memory[agent_idx])}/{self.lower_memory[agent_idx].maxlen}")
    
    def update_upper(self):
        if len(self.upper_memory) < self.batch_size:
            logger.debug(f"update_upper() skipped: memory size {len(self.upper_memory)} < batch size {self.batch_size}")
            return
        
        logger.debug(f"update_upper() called: memory size={len(self.upper_memory)}, batch_size={self.batch_size}")
        
        batch = self.replay_rng.choice(len(self.upper_memory), self.batch_size, replace=False)
        states_batch = []
        commanded_actions_batch = []
        executed_actions_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []
        
        for idx in batch:
            s, commanded, executed, r, ns, d = _unpack_upper_transition(
                self.upper_memory[idx])
            states_batch.append(s)
            commanded_actions_batch.append(commanded)
            executed_actions_batch.append(executed)
            rewards_batch.append(r)
            next_states_batch.append(ns)
            dones_batch.append(d)
        
        states_batch = torch.FloatTensor(np.array(states_batch)).to(self.device)
        commanded_actions_batch = torch.FloatTensor(
            np.array(commanded_actions_batch)).to(self.device)
        executed_actions_batch = torch.FloatTensor(
            np.array(executed_actions_batch)).to(self.device)
        rewards_batch = torch.FloatTensor(np.array(rewards_batch)).to(self.device)
        next_states_batch = torch.FloatTensor(np.array(next_states_batch)).to(self.device)
        dones_batch = torch.FloatTensor(np.array(dones_batch)).to(self.device)
        
        logger.debug(
            f"Upper batch tensors created: states={states_batch.shape}, "
            f"executed_actions={executed_actions_batch.shape}")
        update_info = []
        
        for i in range(self.num_agents):
            logger.debug(f"Updating upper-layer agent {i} networks...")
            
            target_next_actions = []
            for j in range(self.num_agents):
                target_next_actions.append(self.target_upper_actors[j](next_states_batch[:, j]))
            target_next_actions = torch.cat(target_next_actions, dim=1)
            
            next_states_cat = next_states_batch.view(self.batch_size, -1)
            target_q = self.target_upper_critics[i](next_states_cat, target_next_actions)
            
            y_i = rewards_batch[:, i].unsqueeze(1) + self.gamma * target_q * (1 - dones_batch.unsqueeze(1))
            
            states_cat = states_batch.view(self.batch_size, -1)
            actions_cat = executed_actions_batch.view(self.batch_size, -1)
            q_i = self.upper_critics[i](states_cat, actions_cat)
            
            critic_loss = nn.MSELoss()(q_i, y_i.detach())
            logger.debug(f"Upper Agent {i} critic loss: {critic_loss.item():.6f}")
            
            self.upper_critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.upper_critics[i].parameters(), self.clip_norm)
            self.upper_critic_optimizers[i].step()
            
            self.upper_actor_optimizers[i].zero_grad()
            current_actions = []
            for j in range(self.num_agents):
                if j == i:
                    current_actions.append(self.upper_actors[j](states_batch[:, j]))
                else:
                    current_actions.append(executed_actions_batch[:, j].detach())
            current_agent_actions = current_actions[i]
            current_actions = torch.cat(current_actions, dim=1)

            policy_loss = -self.upper_critics[i](states_cat, current_actions).mean()
            projection_loss = _projection_residual_loss(
                current_agent_actions,
                commanded_actions_batch[:, i],
                executed_actions_batch[:, i],
                self.projection_tolerance,
            )
            actor_loss = (
                policy_loss
                + self.upper_projection_loss_weight * projection_loss)
            logger.debug(f"Upper Agent {i} actor loss: {actor_loss.item():.6f}")
            
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.upper_actors[i].parameters(), self.clip_norm)
            self.upper_actor_optimizers[i].step()
            
            self.soft_update(self.upper_actors[i], self.target_upper_actors[i])
            self.soft_update(self.upper_critics[i], self.target_upper_critics[i])
            update_info.append({
                'critic_loss': float(critic_loss.item()),
                'policy_loss': float(policy_loss.item()),
                'projection_loss': float(projection_loss.item()),
            })

        self.last_upper_update_info = update_info
        logger.debug("update_upper() completed successfully!")
    
    def update_lower(self, agent_idx):
        if len(self.lower_memory[agent_idx]) < self.batch_size:
            logger.debug(f"update_lower({agent_idx}) skipped: memory size {len(self.lower_memory[agent_idx])} < batch size {self.batch_size}")
            return

        logger.debug(f"update_lower({agent_idx}) called: memory size={len(self.lower_memory[agent_idx])}, batch_size={self.batch_size}")

        batch = self.replay_rng.choice(len(self.lower_memory[agent_idx]), self.batch_size, replace=False)
        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []

        for idx in batch:
            s, a, r, ns, d = self.lower_memory[agent_idx][idx]
            states_batch.append(s)
            actions_batch.append(a)
            rewards_batch.append(r)
            next_states_batch.append(ns)
            dones_batch.append(d)

        states_batch = torch.FloatTensor(np.array(states_batch)).to(self.device)
        actions_batch = torch.FloatTensor(np.array(actions_batch)).to(self.device)
        rewards_batch = torch.FloatTensor(np.array(rewards_batch)).to(self.device)
        next_states_batch = torch.FloatTensor(np.array(next_states_batch)).to(self.device)
        dones_batch = torch.FloatTensor(np.array(dones_batch)).to(self.device)

        logger.debug(f"Lower batch tensors created: states={states_batch.shape}, actions={actions_batch.shape}")

        q_values = self.lower_dqns[agent_idx](states_batch)  # (batch, 3M)

        with torch.no_grad():
            next_q_all = self.target_lower_dqns[agent_idx](next_states_batch)  # (batch, 3M)
            next_masks = _lower_action_masks_from_states(next_states_batch, self.config)

        M = self.config.M
        total_loss = 0.0

        for j in range(M):
            # Decode discrete action index {0,1,2} for GU j from stored 2M action
            access_np = actions_batch[:, j].detach().cpu().numpy()
            mode_np = actions_batch[:, M + j].detach().cpu().numpy()
            idx_np = np.where(access_np < 0.5, 0, np.where(mode_np >= 0.5, 1, 2))
            action_idx = torch.LongTensor(idx_np).to(self.device)  # (batch,)

            # Q-value for chosen action
            q_slice = q_values[:, j * 3 : j * 3 + 3]  # (batch, 3)
            q_j = q_slice.gather(1, action_idx.unsqueeze(1)).squeeze(1)  # (batch,)

            # Target: r + gamma * max_a' Q'(s', a')
            next_q_slice = next_q_all[:, j * 3 : j * 3 + 3]  # (batch, 3)
            next_q_slice = next_q_slice.masked_fill(~next_masks[:, j], -torch.inf)
            max_next_q_j = next_q_slice.max(dim=1)[0]  # (batch,)
            target_j = rewards_batch + self.gamma * max_next_q_j * (1 - dones_batch)

            total_loss += nn.MSELoss()(q_j, target_j.detach())

        logger.debug(f"Lower Agent {agent_idx} DQN loss: {total_loss.item():.6f}")

        self.lower_optimizers[agent_idx].zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.lower_dqns[agent_idx].parameters(), self.clip_norm)
        self.lower_optimizers[agent_idx].step()
        self.soft_update(self.lower_dqns[agent_idx], self.target_lower_dqns[agent_idx])

        logger.debug(f"update_lower({agent_idx}) completed successfully!")
    
    def model_predict(self, agent_idx, state, action):
        if self.dyna_k <= 0 or not self.models:
            raise RuntimeError("Dyna-Q model is disabled when dyna_k <= 0")
        logger.debug(f"model_predict({agent_idx}) called: state shape={state.shape}, action shape={action.shape}")
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        self.models[agent_idx].eval()
        with torch.no_grad():
            reward_pred, next_state_pred = self.models[agent_idx](state_tensor, action_tensor)
        reward_pred = reward_pred.detach().cpu().numpy()[0, 0]
        next_state_pred = next_state_pred.detach().cpu().numpy()[0]
        logger.debug(f"model_predict({agent_idx}) completed: reward_pred={reward_pred:.4f}, next_state_pred_norm={np.linalg.norm(next_state_pred):.4f}")
        return reward_pred, next_state_pred
    
    def update_model(self, agent_idx):
        if self.dyna_k <= 0 or not self.models:
            return None
        if len(self.lower_memory[agent_idx]) < self.batch_size:
            logger.debug(f"update_model({agent_idx}) skipped: memory size {len(self.lower_memory[agent_idx])} < batch size {self.batch_size}")
            return None
        
        logger.debug(f"update_model({agent_idx}) called: memory size={len(self.lower_memory[agent_idx])}, batch_size={self.batch_size}")
        
        batch = self.model_rng.choice(len(self.lower_memory[agent_idx]), self.batch_size, replace=False)
        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []
        
        for idx in batch:
            s, a, r, ns, d = self.lower_memory[agent_idx][idx]
            states_batch.append(s)
            actions_batch.append(a)
            rewards_batch.append(r)
            next_states_batch.append(ns)
            dones_batch.append(d)
        
        states_batch = torch.FloatTensor(np.array(states_batch)).to(self.device)
        actions_batch = torch.FloatTensor(np.array(actions_batch)).to(self.device)
        rewards_batch = torch.FloatTensor(np.array(rewards_batch)).unsqueeze(1).to(self.device)
        next_states_batch = torch.FloatTensor(np.array(next_states_batch)).to(self.device)
        dones_batch = torch.FloatTensor(np.array(dones_batch)).to(self.device)
        split = max(1, int(0.8 * states_batch.shape[0]))
        train_states, val_states = states_batch[:split], states_batch[split:]
        train_actions, val_actions = actions_batch[:split], actions_batch[split:]
        train_rewards, val_rewards = rewards_batch[:split], rewards_batch[split:]
        train_next_states, val_next_states = next_states_batch[:split], next_states_batch[split:]
        val_dones = dones_batch[split:]

        self.models[agent_idx].train()
        reward_pred, next_state_pred = self.models[agent_idx](train_states, train_actions)
        
        reward_loss = nn.SmoothL1Loss()(reward_pred, train_rewards)
        state_error = (next_state_pred - train_next_states) / self.models[agent_idx].state_scale
        state_loss = nn.SmoothL1Loss()(state_error, torch.zeros_like(state_error))
        total_loss = reward_loss + state_loss
        self.models[agent_idx].eval()
        with torch.no_grad():
            val_reward_pred, val_next_pred = self.models[agent_idx](val_states, val_actions)
            validation_error = (
                (val_next_pred - val_next_states) / self.models[agent_idx].state_scale
            ).abs().mean()
            true_next_q = self.target_lower_dqns[agent_idx](val_next_states).view(
                val_next_states.shape[0], self.config.M, 3)
            model_next_q = self.target_lower_dqns[agent_idx](val_next_pred).view(
                val_next_pred.shape[0], self.config.M, 3)
            true_masks = _lower_action_masks_from_states(val_next_states, self.config)
            model_masks = _lower_action_masks_from_states(val_next_pred, self.config)
            true_next_q = true_next_q.masked_fill(~true_masks, -torch.inf).max(dim=2).values
            model_next_q = model_next_q.masked_fill(~model_masks, -torch.inf).max(dim=2).values
            true_targets = val_rewards + self.gamma * true_next_q * (1 - val_dones.unsqueeze(1))
            model_targets = val_reward_pred + self.gamma * model_next_q * (1 - val_dones.unsqueeze(1))
            bellman_error = (
                (model_targets - true_targets).abs()
                / (true_targets.abs() + self.dyna_bellman_denom_floor)
            ).mean()
        normalized_mae = float(validation_error.item())
        normalized_bellman_error = float(bellman_error.item())
        previous = self.model_error_ema[agent_idx]
        self.model_error_ema[agent_idx] = (
            normalized_mae if not np.isfinite(previous)
            else 0.95 * previous + 0.05 * normalized_mae)
        previous_bellman = self.bellman_error_ema[agent_idx]
        self.bellman_error_ema[agent_idx] = (
            normalized_bellman_error if not np.isfinite(previous_bellman)
            else 0.95 * previous_bellman + 0.05 * normalized_bellman_error)
        
        logger.debug(f"Model {agent_idx} loss: reward_loss={reward_loss.item():.6f}, state_loss={state_loss.item():.6f}, total_loss={total_loss.item():.6f}")
        
        self.model_optimizers[agent_idx].zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.models[agent_idx].parameters(), self.clip_norm)
        self.model_optimizers[agent_idx].step()
        return {
            'reward_loss': float(reward_loss.item()),
            'state_loss': float(state_loss.item()),
            'total_loss': float(total_loss.item()),
            'normalized_state_mae': normalized_mae,
            'normalized_state_mae_ema': float(self.model_error_ema[agent_idx]),
            'bellman_error': normalized_bellman_error,
            'bellman_error_ema': float(self.bellman_error_ema[agent_idx]),
        }
    
    def _dyna_plan_batch(self, agent_idx, k):
        base_batch_size = max(k, int(np.ceil(
            self.dyna_planning_batch_size * self.dyna_model_fraction)))
        batch_size = base_batch_size
        if self.dyna_planning_strategy == 'utility_priority':
            batch_size *= self.dyna_priority_candidate_multiplier
        batch_size = min(batch_size, len(self.lower_memory[agent_idx]))
        indices = self.dyna_rng.choice(
            len(self.lower_memory[agent_idx]), batch_size, replace=False)
        samples = [self.lower_memory[agent_idx][idx] for idx in indices]
        states = np.asarray([x[0] for x in samples], dtype=np.float32)
        actions = np.asarray([x[1] for x in samples], dtype=np.float32)
        true_rewards = np.asarray([x[2] for x in samples], dtype=np.float32)
        true_next = np.asarray([x[3] for x in samples], dtype=np.float32)
        dones = np.asarray([x[4] for x in samples], dtype=np.float32)
        counterfactual = np.zeros(batch_size, dtype=bool)

        if (self.planning_mode == 'model'
                and self.dyna_planning_strategy == 'state_gate'
                and self.dyna_counterfactual_fraction > 0):
            with torch.no_grad():
                q_np = self.lower_dqns[agent_idx](
                    torch.as_tensor(states, device=self.device)).cpu().numpy()
            base, stride = 7 + self.config.F, 3 + self.config.F
            for b in range(batch_size):
                if self.dyna_rng.random() >= self.dyna_counterfactual_fraction:
                    continue
                covered = [j for j in range(self.config.M)
                           if states[b, base + j * stride] >= 0.5]
                if not covered:
                    continue
                j = int(self.dyna_rng.choice(covered))
                choice = int(np.argmax(q_np[b, j * 3:j * 3 + 3]))
                actions[b, j] = float(choice != 0)
                actions[b, self.config.M + j] = float(choice == 1)
                counterfactual[b] = True

        state_t = torch.as_tensor(states, device=self.device)
        action_t = torch.as_tensor(actions, device=self.device)
        done_t = torch.as_tensor(dones, device=self.device)
        if self.planning_mode == 'replay':
            reward_t = torch.as_tensor(true_rewards, device=self.device)
            next_t = torch.as_tensor(true_next, device=self.device)
        else:
            self.models[agent_idx].eval()
            with torch.no_grad():
                reward_pred, next_t = self.models[agent_idx](state_t, action_t)
            reward_t = reward_pred.squeeze(1)
            scale = self.models[agent_idx].state_scale.cpu().numpy()
            for b in np.flatnonzero(~counterfactual):
                stats = self.planning_errors[agent_idx]
                stats['count'] += 1
                stats['reward_err'] += abs(float(reward_t[b].item()) - float(true_rewards[b]))
                stats['state_err'] += float(np.linalg.norm(
                    (next_t[b].cpu().numpy() - true_next[b]) / scale))

        q_all = self.lower_dqns[agent_idx](state_t).view(batch_size, self.config.M, 3)
        access_t, mode_t = action_t[:, :self.config.M], action_t[:, self.config.M:]
        action_idx = torch.where(
            access_t < 0.5, torch.zeros_like(access_t, dtype=torch.long),
            torch.where(mode_t >= 0.5, torch.ones_like(access_t, dtype=torch.long),
                        torch.full_like(access_t, 2, dtype=torch.long)))
        chosen_q = q_all.gather(2, action_idx.unsqueeze(2)).squeeze(2)
        with torch.no_grad():
            next_q_all = self.target_lower_dqns[agent_idx](next_t).view(
                batch_size, self.config.M, 3)
            next_masks = _lower_action_masks_from_states(next_t, self.config)
            next_q = next_q_all.masked_fill(~next_masks, -torch.inf).max(dim=2).values
            targets = reward_t.unsqueeze(1) + self.gamma * next_q * (1 - done_t.unsqueeze(1))

        plan_bellman_error = None
        plan_sample_weight = 1.0
        robust_scale_value = None
        selected_fraction = 1.0
        mean_utility = None
        guarded_strategies = {
            'bcad', 'robust_budget', 'real_anchor', 'utility_priority'}
        if (self.dyna_planning_strategy in guarded_strategies
                and self.planning_mode == 'model'):
            true_next_t = torch.as_tensor(true_next, device=self.device)
            true_reward_t = torch.as_tensor(true_rewards, device=self.device)
            with torch.no_grad():
                true_next_q_all = self.target_lower_dqns[agent_idx](true_next_t).view(
                    batch_size, self.config.M, 3)
                true_masks = _lower_action_masks_from_states(true_next_t, self.config)
                true_next_q = true_next_q_all.masked_fill(
                    ~true_masks, -torch.inf).max(dim=2).values
                true_targets = true_reward_t.unsqueeze(1) + self.gamma * true_next_q * (
                    1 - done_t.unsqueeze(1))
                if self.dyna_planning_strategy == 'bcad':
                    target_error = (
                        (targets - true_targets).abs()
                        / (true_targets.abs() + self.dyna_bellman_denom_floor))
                    sample_weights = torch.exp(
                        -self.dyna_bellman_beta * target_error).clamp_min(1e-3)
                    guarded_targets = targets
                else:
                    # A batch-level robust scale avoids exploding relative
                    # errors when an individual real Bellman target is near 0.
                    flat_targets = true_targets.reshape(-1)
                    target_median = flat_targets.median()
                    mad = (flat_targets - target_median).abs().median()
                    robust_scale = torch.clamp(
                        1.4826 * mad, min=self.dyna_robust_scale_floor)
                    target_error = (targets - true_targets).abs() / robust_scale
                    sample_weights = 1.0 / (1.0 + target_error.square())
                    robust_scale_value = float(robust_scale.item())
                    if self.dyna_planning_strategy in ('real_anchor', 'utility_priority'):
                        max_delta = self.dyna_anchor_clip * robust_scale
                        model_delta = (targets - true_targets).clamp(
                            min=-max_delta, max=max_delta)
                        guarded_targets = (
                            true_targets + self.dyna_anchor_alpha * model_delta)
                    else:
                        guarded_targets = targets

            selected = torch.arange(batch_size, device=self.device)
            sample_error = target_error.mean(dim=1)
            keep_count = max(1, int(np.ceil(
                batch_size * self.dyna_plan_keep_fraction)))
            if self.dyna_planning_strategy == 'robust_budget':
                selected = torch.argsort(sample_error)[:keep_count]
            elif self.dyna_planning_strategy == 'utility_priority':
                with torch.no_grad():
                    real_td_error = (chosen_q - true_targets).abs().mean(dim=1)
                    utility = real_td_error / (1.0 + sample_error)
                    selected = torch.argsort(utility, descending=True)[:keep_count]
                    mean_utility = float(utility[selected].mean().item())

            selected_fraction = float(selected.numel() / batch_size)
            selected_q = chosen_q[selected]
            selected_targets = guarded_targets[selected]
            selected_weights = sample_weights[selected]
            element_loss = F.smooth_l1_loss(
                selected_q, selected_targets, reduction='none')
            loss = ((element_loss * selected_weights).sum()
                    / selected_weights.sum().clamp_min(1e-6))
            plan_bellman_error = float(target_error.mean().item())
            plan_sample_weight = float(selected_weights.mean().item())
        else:
            loss = nn.SmoothL1Loss()(chosen_q, targets)

        self.lower_optimizers[agent_idx].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.lower_dqns[agent_idx].parameters(), self.clip_norm)
        optimizer = self.lower_optimizers[agent_idx]
        saved_lrs = [group['lr'] for group in optimizer.param_groups]
        if self.dyna_planning_strategy in (
                'bcad', 'robust_budget', 'real_anchor', 'utility_priority'):
            for group, lr in zip(optimizer.param_groups, saved_lrs):
                group['lr'] = lr * self.dyna_model_update_lr_scale
        optimizer.step()
        for group, lr in zip(optimizer.param_groups, saved_lrs):
            group['lr'] = lr
        self.soft_update(self.lower_dqns[agent_idx], self.target_lower_dqns[agent_idx])
        if self.last_plan_info[agent_idx] is not None:
            self.last_plan_info[agent_idx].update({
                'planned': 1.0,
                'sample_weight': plan_sample_weight,
                'sample_bellman_error': plan_bellman_error,
                'robust_scale': robust_scale_value,
                'selected_fraction': selected_fraction,
                'mean_utility': mean_utility,
                'loss': float(loss.item()),
            })
        return float(loss.item())

    def dyna_plan(self, agent_idx, k=None):
        self.last_plan_info[agent_idx] = {
            'strategy': self.dyna_planning_strategy,
            'eligible': 0.0,
            'planned': 0.0,
            'trust': 0.0,
            'plan_probability': 0.0,
            'sample_weight': None,
            'sample_bellman_error': None,
            'robust_scale': None,
            'selected_fraction': None,
            'mean_utility': None,
            'loss': None,
        }
        if self.dyna_k <= 0 or not self.models:
            return
        if k is None:
            k = self.dyna_k

        if k <= 0:
            return
        required_samples = max(k, self.dyna_warmup)
        if len(self.lower_memory[agent_idx]) < required_samples:
            logger.debug(
                f"dyna_plan({agent_idx}) skipped: memory size {len(self.lower_memory[agent_idx])} "
                f"< required={required_samples} (k={k}, warmup={self.dyna_warmup})"
            )
            return

        if self.real_steps[agent_idx] < self.dyna_warmup_steps:
            return
        self.last_plan_info[agent_idx]['eligible'] = 1.0
        if self.planning_mode == 'model':
            if self.dyna_planning_strategy == 'bcad':
                error = self.bellman_error_ema[agent_idx]
                if not np.isfinite(error):
                    return
                trust = float(np.exp(-self.dyna_bellman_beta * error))
                ramp_progress = max(
                    0, int(self.real_steps[agent_idx]) - self.dyna_warmup_steps + 1)
                ramp = min(1.0, ramp_progress / self.dyna_plan_ramp_steps)
                probability = self.dyna_plan_probability_max * ramp * trust
                self.last_plan_info[agent_idx].update({
                    'trust': trust,
                    'plan_probability': probability,
                })
                if self.dyna_rng.random() >= probability:
                    return
            elif self.dyna_planning_strategy in (
                    'robust_budget', 'real_anchor', 'utility_priority'):
                ramp_progress = max(
                    0, int(self.real_steps[agent_idx]) - self.dyna_warmup_steps + 1)
                ramp = min(1.0, ramp_progress / self.dyna_plan_ramp_steps)
                probability = (
                    self.dyna_plan_probability_min
                    + (self.dyna_plan_probability_max
                       - self.dyna_plan_probability_min) * ramp)
                self.last_plan_info[agent_idx].update({
                    'trust': 1.0,
                    'plan_probability': probability,
                })
                if self.dyna_rng.random() >= probability:
                    return
            else:
                if self.model_error_ema[agent_idx] > self.dyna_model_error_threshold:
                    return
                self.last_plan_info[agent_idx].update({
                    'trust': 1.0,
                    'plan_probability': 1.0,
                })
        return self._dyna_plan_batch(agent_idx, k)

        logger.debug(f"dyna_plan({agent_idx}) called: k={k}, memory size={len(self.lower_memory[agent_idx])}")

        batch = self.dyna_rng.choice(len(self.lower_memory[agent_idx]), k, replace=False)

        self.lower_optimizers[agent_idx].zero_grad()

        total_loss = 0.0
        M = self.config.M
        for idx in batch:
            s, a, r, ns, d = self.lower_memory[agent_idx][idx]

            if self.planning_mode == 'replay':
                # Perfect-oracle control: use the true transition stored in the
                # replay tuple instead of the learned world model. Sampling,
                # gating and update form are identical to 'model' mode — only
                # the (r, s') target source differs.
                r_pred, s_pred = float(r), ns
            else:
                r_pred, s_pred = self.model_predict(agent_idx, s, a)
                # Zero-cost online model-error instrumentation (truth is in the
                # same tuple): |r_hat - r| and ||s_hat' - s'|| per plan sample.
                self.planning_errors[agent_idx]['count'] += 1
                self.planning_errors[agent_idx]['reward_err'] += abs(float(r_pred) - float(r))
                self.planning_errors[agent_idx]['state_err'] += float(
                    np.linalg.norm(np.asarray(s_pred, dtype=float) - np.asarray(ns, dtype=float))
                )

            state_tensor = torch.FloatTensor(s).unsqueeze(0).to(self.device)
            action_np = np.array(a)  # (2M,) encoded action
            next_state_pred_tensor = torch.FloatTensor(s_pred).unsqueeze(0).to(self.device)
            reward_pred_tensor = torch.FloatTensor([[r_pred]]).to(self.device)
            done_tensor = torch.FloatTensor([[d]]).to(self.device)

            self.lower_dqns[agent_idx].train()

            q_values = self.lower_dqns[agent_idx](state_tensor)  # (1, 3M)

            with torch.no_grad():
                next_q_all = self.target_lower_dqns[agent_idx](next_state_pred_tensor)  # (1, 3M)

            sample_loss = 0.0
            for j in range(M):
                access = action_np[j]
                mode = action_np[M + j]
                if access < 0.5:
                    act_idx = 0      # no_access
                elif mode >= 0.5:
                    act_idx = 1      # RF
                else:
                    act_idx = 2      # Backscatter

                q_j = q_values[0, j * 3 + act_idx]
                max_next_q_j = next_q_all[0, j * 3 : j * 3 + 3].max()
                target_j = reward_pred_tensor.squeeze() + self.gamma * max_next_q_j * (1 - done_tensor.squeeze())
                sample_loss += nn.MSELoss()(q_j, target_j.detach())

            total_loss += sample_loss.item()
            sample_loss.backward()

            del state_tensor, next_state_pred_tensor, reward_pred_tensor, done_tensor
            del q_values, next_q_all, sample_loss

        self.lower_optimizers[agent_idx].step()

        logger.debug(f"dyna_plan({agent_idx}) completed: avg_loss={total_loss/k:.6f}, k={k}")

    def planning_error_summary(self):
        """Per-agent mean online model error over the current accumulation window
        ('model' planning mode only). Returns {agent_idx: {count, mean_reward_err,
        mean_state_err}}; mean is None when the agent recorded no planning samples."""
        summary = {}
        for i, stats in self.planning_errors.items():
            n = stats['count']
            summary[i] = {
                'count': n,
                'mean_reward_err': float(stats['reward_err'] / n) if n else None,
                'mean_state_err': float(stats['state_err'] / n) if n else None,
            }
        return summary

    def reset_planning_errors(self):
        for i in self.planning_errors:
            self.planning_errors[i]['count'] = 0
            self.planning_errors[i]['reward_err'] = 0.0
            self.planning_errors[i]['state_err'] = 0.0

    def step_episode_schedulers(self):
        self.upper_noise_episode += 1
        for i in range(self.num_agents):
            self.upper_actor_schedulers[i].step()
            self.upper_critic_schedulers[i].step()
            self.lower_schedulers[i].step()
            if self.dyna_k > 0 and self.model_schedulers:
                self.model_schedulers[i].step()
        if self.target_update_mode == 'hard':
            self._hard_counter += 1
            if self._hard_counter >= self.hard_update_every:
                self._hard_counter = 0
                for i in range(self.num_agents):
                    self.target_upper_actors[i].load_state_dict(self.upper_actors[i].state_dict())
                    self.target_upper_critics[i].load_state_dict(self.upper_critics[i].state_dict())
                    self.target_lower_dqns[i].load_state_dict(self.lower_dqns[i].state_dict())
                logger.debug(f"Hierarchical hard target sync at episode boundary (every {self.hard_update_every})")

    def soft_update(self, source, target):
        if getattr(self, 'target_update_mode', 'soft') == 'hard':
            return
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)

    def save_checkpoint(self, filepath, episode):
        checkpoint = {
            'episode': episode,
            'epsilon': self.epsilon,
            'upper_noise_episode': self.upper_noise_episode,
            'real_steps': self.real_steps.copy(),
            'model_error_ema': self.model_error_ema.copy(),
            'bellman_error_ema': self.bellman_error_ema.copy(),
            'dyna_planning_strategy': self.dyna_planning_strategy,
            'upper_actors': {i: self.upper_actors[i].state_dict() for i in range(self.num_agents)},
            'target_upper_actors': {i: self.target_upper_actors[i].state_dict() for i in range(self.num_agents)},
            'upper_critics': {i: self.upper_critics[i].state_dict() for i in range(self.num_agents)},
            'target_upper_critics': {i: self.target_upper_critics[i].state_dict() for i in range(self.num_agents)},
            'lower_dqns': {i: self.lower_dqns[i].state_dict() for i in range(self.num_agents)},
            'target_lower_dqns': {i: self.target_lower_dqns[i].state_dict() for i in range(self.num_agents)},
            'upper_actor_optimizers': {i: self.upper_actor_optimizers[i].state_dict() for i in range(self.num_agents)},
            'upper_critic_optimizers': {i: self.upper_critic_optimizers[i].state_dict() for i in range(self.num_agents)},
            'lower_optimizers': {i: self.lower_optimizers[i].state_dict() for i in range(self.num_agents)},
            'upper_actor_schedulers': {i: self.upper_actor_schedulers[i].state_dict() for i in range(self.num_agents)},
            'upper_critic_schedulers': {i: self.upper_critic_schedulers[i].state_dict() for i in range(self.num_agents)},
            'lower_schedulers': {i: self.lower_schedulers[i].state_dict() for i in range(self.num_agents)},
        }
        if self.dyna_k > 0 and self.models:
            checkpoint['models'] = {i: self.models[i].state_dict() for i in range(self.num_agents)}
            checkpoint['model_optimizers'] = {i: self.model_optimizers[i].state_dict() for i in range(self.num_agents)}
            checkpoint['model_schedulers'] = {i: self.model_schedulers[i].state_dict() for i in range(self.num_agents)}
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath} (episode {episode}, epsilon={self.epsilon:.4f}, dyna_k={self.dyna_k})")

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        for i in range(self.num_agents):
            self.upper_actors[i].load_state_dict(checkpoint['upper_actors'][i])
            self.target_upper_actors[i].load_state_dict(checkpoint['target_upper_actors'][i])
            self.upper_critics[i].load_state_dict(checkpoint['upper_critics'][i])
            self.target_upper_critics[i].load_state_dict(checkpoint['target_upper_critics'][i])
            self.lower_dqns[i].load_state_dict(checkpoint['lower_dqns'][i])
            self.target_lower_dqns[i].load_state_dict(checkpoint['target_lower_dqns'][i])
            self.upper_actor_optimizers[i].load_state_dict(checkpoint['upper_actor_optimizers'][i])
            self.upper_critic_optimizers[i].load_state_dict(checkpoint['upper_critic_optimizers'][i])
            self.lower_optimizers[i].load_state_dict(checkpoint['lower_optimizers'][i])
            self.upper_actor_schedulers[i].load_state_dict(checkpoint['upper_actor_schedulers'][i])
            self.upper_critic_schedulers[i].load_state_dict(checkpoint['upper_critic_schedulers'][i])
            self.lower_schedulers[i].load_state_dict(checkpoint['lower_schedulers'][i])
        if self.dyna_k > 0 and self.models and 'models' in checkpoint:
            for i in range(self.num_agents):
                self.models[i].load_state_dict(checkpoint['models'][i])
                self.model_optimizers[i].load_state_dict(checkpoint['model_optimizers'][i])
                self.model_schedulers[i].load_state_dict(checkpoint['model_schedulers'][i])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.upper_noise_episode = checkpoint.get(
            'upper_noise_episode', self.upper_noise_episode)
        self.real_steps = np.asarray(checkpoint.get('real_steps', self.real_steps), dtype=np.int64)
        self.model_error_ema = np.asarray(
            checkpoint.get('model_error_ema', self.model_error_ema), dtype=float)
        self.bellman_error_ema = np.asarray(
            checkpoint.get('bellman_error_ema', self.bellman_error_ema), dtype=float)
        logger.info(f"Checkpoint loaded from {filepath} (episode {checkpoint['episode']}, epsilon={self.epsilon:.4f})")
        return checkpoint['episode']


class HierarchicalNoDynaAgent:
    def __init__(self, state_dim, action_dim, num_agents, config):
        logger.info("=" * 60)
        logger.info("Initializing HierarchicalNoDynaAgent (without Dyna-Q)...")
        logger.info("=" * 60)
        
        self.config = config
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_scale = _build_state_scale(config, state_dim)
        self.global_state_scale = np.tile(self.state_scale, num_agents)
        self.upper_projection_loss_weight = max(
            0.0, float(getattr(config, 'upper_projection_loss_weight', 1.0)))
        self.projection_tolerance = max(
            0.0, float(getattr(config, 'safety_projection_tolerance', 1e-6)))
        self.last_upper_update_info = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(config.torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.torch_seed)
        self.action_rng = config.rngs['action']
        self.replay_rng = config.rngs['replay']
        self.upper_noise_episode = 0
        
        logger.info(f"HierarchicalNoDyna params: num_agents={num_agents}, state_dim={state_dim}, action_dim={action_dim}, device={self.device}")

        logger.info("Creating upper-layer actor networks...")
        # M9: 前4维(direction+speed)tanh，第5维(scheduled)sigmoid
        self.upper_actors = [
            UpperActor(state_dim, 4, 1, state_scale=self.state_scale).to(self.device)
            for _ in range(num_agents)]

        logger.info("Creating upper-layer critic networks...")
        self.upper_critics = [
            UpperCritic(state_dim * num_agents, 5 * num_agents,
                        state_scale=self.global_state_scale).to(self.device)
            for _ in range(num_agents)]

        logger.info("Creating upper-layer target actor networks...")
        self.target_upper_actors = [
            UpperActor(state_dim, 4, 1, state_scale=self.state_scale).to(self.device)
            for _ in range(num_agents)]
        
        logger.info("Creating upper-layer target critic networks...")
        self.target_upper_critics = [
            UpperCritic(state_dim * num_agents, 5 * num_agents,
                        state_scale=self.global_state_scale).to(self.device)
            for _ in range(num_agents)]
        
        logger.info("Copying upper-layer weights to target networks...")
        for i in range(num_agents):
            self.target_upper_actors[i].load_state_dict(self.upper_actors[i].state_dict())
            self.target_upper_critics[i].load_state_dict(self.upper_critics[i].state_dict())
        
        logger.info("Creating upper-layer optimizers...")
        self.upper_actor_optimizers = [optim.Adam(self.upper_actors[i].parameters(), lr=1e-4) for i in range(num_agents)]
        self.upper_critic_optimizers = [optim.Adam(self.upper_critics[i].parameters(), lr=1e-3) for i in range(num_agents)]
        
        logger.info("Creating lower-layer DQN networks...")
        self.lower_dqns = [
            LowerDQN(state_dim, 3 * config.M, state_scale=self.state_scale).to(self.device)
            for _ in range(num_agents)]
        
        logger.info("Creating lower-layer target DQN networks...")
        self.target_lower_dqns = [
            LowerDQN(state_dim, 3 * config.M, state_scale=self.state_scale).to(self.device)
            for _ in range(num_agents)]
        
        logger.info("Copying lower-layer weights to target networks...")
        for i in range(num_agents):
            self.target_lower_dqns[i].load_state_dict(self.lower_dqns[i].state_dict())
        
        logger.info("Creating lower-layer optimizers...")
        self.lower_optimizers = [optim.Adam(self.lower_dqns[i].parameters(), lr=1e-4) for i in range(num_agents)]
        
        logger.info("Creating learning rate schedulers...")
        self.upper_actor_schedulers = [optim.lr_scheduler.StepLR(self.upper_actor_optimizers[i], step_size=500, gamma=0.9) for i in range(num_agents)]
        self.upper_critic_schedulers = [optim.lr_scheduler.StepLR(self.upper_critic_optimizers[i], step_size=500, gamma=0.9) for i in range(num_agents)]
        self.lower_schedulers = [optim.lr_scheduler.StepLR(self.lower_optimizers[i], step_size=500, gamma=0.9) for i in range(num_agents)]
        
        logger.info("Initializing upper-layer replay memory...")
        self.upper_memory = deque(maxlen=10000)
        
        logger.info("Initializing lower-layer replay memories...")
        self.lower_memory = [deque(maxlen=10000) for _ in range(num_agents)]
        
        self.gamma = 0.95
        self.tau = 0.01
        self.batch_size = 32
        # M8: 下层探索模式('fixed'固定/'decay'衰减)，论文用固定eps=0.05
        self.epsilon_mode = getattr(config, 'epsilon_mode', 'fixed')
        self.epsilon_fixed = getattr(config, 'epsilon_fixed', 0.05)
        self.epsilon = self.epsilon_fixed if self.epsilon_mode == 'fixed' else 0.1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.clip_norm = 1.0
        # M7: 目标网络更新模式开关
        self.target_update_mode = getattr(config, 'target_update_mode', 'soft')
        self.hard_update_every = getattr(config, 'hard_update_every', 100)
        self._hard_counter = 0

        logger.info(f"Upper memory capacity: {self.upper_memory.maxlen}")
        logger.info(f"Lower memory capacity: {self.lower_memory[0].maxlen} per agent")
        logger.info(f"gamma={self.gamma}, tau={self.tau}, batch_size={self.batch_size}, epsilon={self.epsilon}, epsilon_mode={self.epsilon_mode}")
        logger.info(f"epsilon_min={self.epsilon_min}, epsilon_decay={self.epsilon_decay}, clip_norm={self.clip_norm}")
        logger.info("=" * 60)
        logger.info("HierarchicalNoDynaAgent initialization complete!")
        logger.info("=" * 60)
    
    def upper_act(self, states, noise=True):
        logger.debug(f"upper_act() called: states shape={states.shape}, noise={noise}")
        actions = []
        
        for i in range(self.num_agents):
            state = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
            action = self.upper_actors[i](state).detach().cpu().numpy()[0]
            
            if noise:
                stds = _upper_noise_stds(
                    self.config, self.upper_noise_episode)
                noise_val = np.empty_like(action)
                noise_val[:3] = self.action_rng.normal(
                    0, stds['direction'], size=3)
                noise_val[3] = self.action_rng.normal(0, stds['speed'])
                noise_val[4] = self.action_rng.normal(0, stds['schedule'])
                action += noise_val
            
            action[:4] = np.clip(action[:4], -1, 1)
            action[4] = np.clip(action[4], 0, 1)
            actions.append(action)
        
        actions_array = np.array(actions)
        return actions_array
    
    def lower_act(self, states, action_masks=None, explore=True):
        logger.debug(f"lower_act() called: states shape={states.shape}, epsilon={self.epsilon}")
        actions = []
        M = self.config.M

        for i in range(self.num_agents):
            state = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)

            if explore and self.action_rng.random() < self.epsilon:
                # Random: pick one of {0,1,2} per GU, encode to 2M
                if action_masks is None:
                    action_indices = self.action_rng.integers(0, 3, size=M)
                else:
                    action_indices = np.array([
                        self.action_rng.choice(np.flatnonzero(action_masks[i, j]))
                        for j in range(M)
                    ])
                action = np.zeros(2 * M)
                for j, idx in enumerate(action_indices):
                    if idx == 0: action[j] = 0; action[M + j] = 0
                    elif idx == 1: action[j] = 1; action[M + j] = 1
                    else: action[j] = 1; action[M + j] = 0
            else:
                q_values = self.lower_dqns[i](state).detach().cpu().numpy()[0]  # (3M,)
                action = np.zeros(2 * M)
                for j in range(M):
                    q_slice = q_values[j * 3 : j * 3 + 3]
                    if action_masks is not None:
                        q_slice = np.where(action_masks[i, j], q_slice, -np.inf)
                    idx = int(np.argmax(q_slice))
                    if idx == 0: action[j] = 0; action[M + j] = 0
                    elif idx == 1: action[j] = 1; action[M + j] = 1
                    else: action[j] = 1; action[M + j] = 0

            actions.append(action)

        actions_array = np.array(actions)
        return actions_array
    
    def add_upper_memory(self, states, actions, rewards, next_states, dones,
                         executed_actions=None):
        commanded = np.asarray(actions, dtype=float).copy()
        executed = (commanded.copy() if executed_actions is None else
                    np.asarray(executed_actions, dtype=float).copy())
        if executed.shape != commanded.shape:
            raise ValueError(
                f"executed_actions shape {executed.shape} != commanded shape {commanded.shape}")
        self.upper_memory.append(
            (states, commanded, executed, rewards, next_states, dones))
    
    def add_lower_memory(self, agent_idx, state, action, reward, next_state, done):
        self.lower_memory[agent_idx].append((state, action, reward, next_state, done))
    
    def update_upper(self):
        if len(self.upper_memory) < self.batch_size:
            return
        
        batch = self.replay_rng.choice(len(self.upper_memory), self.batch_size, replace=False)
        states_batch = []
        commanded_actions_batch = []
        executed_actions_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []
        
        for idx in batch:
            s, commanded, executed, r, ns, d = _unpack_upper_transition(
                self.upper_memory[idx])
            states_batch.append(s)
            commanded_actions_batch.append(commanded)
            executed_actions_batch.append(executed)
            rewards_batch.append(r)
            next_states_batch.append(ns)
            dones_batch.append(d)
        
        states_batch = torch.FloatTensor(np.array(states_batch)).to(self.device)
        commanded_actions_batch = torch.FloatTensor(
            np.array(commanded_actions_batch)).to(self.device)
        executed_actions_batch = torch.FloatTensor(
            np.array(executed_actions_batch)).to(self.device)
        rewards_batch = torch.FloatTensor(np.array(rewards_batch)).to(self.device)
        next_states_batch = torch.FloatTensor(np.array(next_states_batch)).to(self.device)
        dones_batch = torch.FloatTensor(np.array(dones_batch)).to(self.device)
        
        update_info = []
        for i in range(self.num_agents):
            target_next_actions = []
            for j in range(self.num_agents):
                target_next_actions.append(self.target_upper_actors[j](next_states_batch[:, j]))
            target_next_actions = torch.cat(target_next_actions, dim=1)
            
            next_states_cat = next_states_batch.view(self.batch_size, -1)
            target_q = self.target_upper_critics[i](next_states_cat, target_next_actions)
            
            y_i = rewards_batch[:, i].unsqueeze(1) + self.gamma * target_q * (1 - dones_batch.unsqueeze(1))
            
            states_cat = states_batch.view(self.batch_size, -1)
            actions_cat = executed_actions_batch.view(self.batch_size, -1)
            q_i = self.upper_critics[i](states_cat, actions_cat)
            
            critic_loss = nn.MSELoss()(q_i, y_i.detach())
            
            self.upper_critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.upper_critics[i].parameters(), self.clip_norm)
            self.upper_critic_optimizers[i].step()
            
            self.upper_actor_optimizers[i].zero_grad()
            current_actions = []
            for j in range(self.num_agents):
                if j == i:
                    current_actions.append(self.upper_actors[j](states_batch[:, j]))
                else:
                    current_actions.append(executed_actions_batch[:, j].detach())
            current_agent_actions = current_actions[i]
            current_actions = torch.cat(current_actions, dim=1)

            policy_loss = -self.upper_critics[i](states_cat, current_actions).mean()
            projection_loss = _projection_residual_loss(
                current_agent_actions,
                commanded_actions_batch[:, i],
                executed_actions_batch[:, i],
                self.projection_tolerance,
            )
            actor_loss = (
                policy_loss
                + self.upper_projection_loss_weight * projection_loss)
            
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.upper_actors[i].parameters(), self.clip_norm)
            self.upper_actor_optimizers[i].step()
            
            self.soft_update(self.upper_actors[i], self.target_upper_actors[i])
            self.soft_update(self.upper_critics[i], self.target_upper_critics[i])
            update_info.append({
                'critic_loss': float(critic_loss.item()),
                'policy_loss': float(policy_loss.item()),
                'projection_loss': float(projection_loss.item()),
            })
        self.last_upper_update_info = update_info
    
    def update_lower(self, agent_idx):
        if len(self.lower_memory[agent_idx]) < self.batch_size:
            return

        batch = self.replay_rng.choice(len(self.lower_memory[agent_idx]), self.batch_size, replace=False)
        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = [], [], [], [], []

        for idx in batch:
            s, a, r, ns, d = self.lower_memory[agent_idx][idx]
            states_batch.append(s); actions_batch.append(a); rewards_batch.append(r)
            next_states_batch.append(ns); dones_batch.append(d)

        states_batch = torch.FloatTensor(np.array(states_batch)).to(self.device)
        actions_batch = torch.FloatTensor(np.array(actions_batch)).to(self.device)
        rewards_batch = torch.FloatTensor(np.array(rewards_batch)).to(self.device)
        next_states_batch = torch.FloatTensor(np.array(next_states_batch)).to(self.device)
        dones_batch = torch.FloatTensor(np.array(dones_batch)).to(self.device)

        q_values = self.lower_dqns[agent_idx](states_batch)  # (batch, 3M)

        with torch.no_grad():
            next_q_all = self.target_lower_dqns[agent_idx](next_states_batch)  # (batch, 3M)
            next_masks = _lower_action_masks_from_states(next_states_batch, self.config)

        M = self.config.M
        total_loss = 0.0

        for j in range(M):
            access_np = actions_batch[:, j].detach().cpu().numpy()
            mode_np = actions_batch[:, M + j].detach().cpu().numpy()
            idx_np = np.where(access_np < 0.5, 0, np.where(mode_np >= 0.5, 1, 2))
            action_idx = torch.LongTensor(idx_np).to(self.device)

            q_slice = q_values[:, j * 3 : j * 3 + 3]
            q_j = q_slice.gather(1, action_idx.unsqueeze(1)).squeeze(1)

            next_q_slice = next_q_all[:, j * 3 : j * 3 + 3]
            next_q_slice = next_q_slice.masked_fill(~next_masks[:, j], -torch.inf)
            max_next_q_j = next_q_slice.max(dim=1)[0]
            target_j = rewards_batch + self.gamma * max_next_q_j * (1 - dones_batch)

            total_loss += nn.MSELoss()(q_j, target_j.detach())

        self.lower_optimizers[agent_idx].zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.lower_dqns[agent_idx].parameters(), self.clip_norm)
        self.lower_optimizers[agent_idx].step()
        self.soft_update(self.lower_dqns[agent_idx], self.target_lower_dqns[agent_idx])

    def step_episode_schedulers(self):
        self.upper_noise_episode += 1
        for i in range(self.num_agents):
            self.upper_actor_schedulers[i].step()
            self.upper_critic_schedulers[i].step()
            self.lower_schedulers[i].step()
        if self.target_update_mode == 'hard':
            self._hard_counter += 1
            if self._hard_counter >= self.hard_update_every:
                self._hard_counter = 0
                for i in range(self.num_agents):
                    self.target_upper_actors[i].load_state_dict(self.upper_actors[i].state_dict())
                    self.target_upper_critics[i].load_state_dict(self.upper_critics[i].state_dict())
                    self.target_lower_dqns[i].load_state_dict(self.lower_dqns[i].state_dict())
                logger.debug(f"NoDyna hard target sync at episode boundary (every {self.hard_update_every})")

    def save_checkpoint(self, filepath, episode):
        checkpoint = {
            'episode': episode,
            'epsilon': self.epsilon,
            'upper_noise_episode': self.upper_noise_episode,
            'upper_actors': {i: self.upper_actors[i].state_dict() for i in range(self.num_agents)},
            'target_upper_actors': {i: self.target_upper_actors[i].state_dict() for i in range(self.num_agents)},
            'upper_critics': {i: self.upper_critics[i].state_dict() for i in range(self.num_agents)},
            'target_upper_critics': {i: self.target_upper_critics[i].state_dict() for i in range(self.num_agents)},
            'lower_dqns': {i: self.lower_dqns[i].state_dict() for i in range(self.num_agents)},
            'target_lower_dqns': {i: self.target_lower_dqns[i].state_dict() for i in range(self.num_agents)},
            'upper_actor_optimizers': {i: self.upper_actor_optimizers[i].state_dict() for i in range(self.num_agents)},
            'upper_critic_optimizers': {i: self.upper_critic_optimizers[i].state_dict() for i in range(self.num_agents)},
            'lower_optimizers': {i: self.lower_optimizers[i].state_dict() for i in range(self.num_agents)},
            'upper_actor_schedulers': {i: self.upper_actor_schedulers[i].state_dict() for i in range(self.num_agents)},
            'upper_critic_schedulers': {i: self.upper_critic_schedulers[i].state_dict() for i in range(self.num_agents)},
            'lower_schedulers': {i: self.lower_schedulers[i].state_dict() for i in range(self.num_agents)},
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath} (episode {episode}, epsilon={self.epsilon:.4f})")

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        for i in range(self.num_agents):
            self.upper_actors[i].load_state_dict(checkpoint['upper_actors'][i])
            self.target_upper_actors[i].load_state_dict(checkpoint['target_upper_actors'][i])
            self.upper_critics[i].load_state_dict(checkpoint['upper_critics'][i])
            self.target_upper_critics[i].load_state_dict(checkpoint['target_upper_critics'][i])
            self.lower_dqns[i].load_state_dict(checkpoint['lower_dqns'][i])
            self.target_lower_dqns[i].load_state_dict(checkpoint['target_lower_dqns'][i])
            self.upper_actor_optimizers[i].load_state_dict(checkpoint['upper_actor_optimizers'][i])
            self.upper_critic_optimizers[i].load_state_dict(checkpoint['upper_critic_optimizers'][i])
            self.lower_optimizers[i].load_state_dict(checkpoint['lower_optimizers'][i])
            self.upper_actor_schedulers[i].load_state_dict(checkpoint['upper_actor_schedulers'][i])
            self.upper_critic_schedulers[i].load_state_dict(checkpoint['upper_critic_schedulers'][i])
            self.lower_schedulers[i].load_state_dict(checkpoint['lower_schedulers'][i])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.upper_noise_episode = checkpoint.get(
            'upper_noise_episode', self.upper_noise_episode)
        logger.info(f"Checkpoint loaded from {filepath} (episode {checkpoint['episode']}, epsilon={self.epsilon:.4f})")
        return checkpoint['episode']

    def soft_update(self, source, target):
        if getattr(self, 'target_update_mode', 'soft') == 'hard':
            return
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)
