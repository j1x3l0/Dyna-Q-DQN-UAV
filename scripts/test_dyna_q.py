import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from system_model import Config, Environment
from hierarchical_agent import HierarchicalAgent

def test_dyna_q():
    config = Config()
    env = Environment(config)
    
    state_dim = 30
    action_dim = 4 + 2 * config.M + 1
    
    agent = HierarchicalAgent(state_dim, action_dim, config.N, config)
    
    print("Testing Dyna-Q functionality...")
    print(f"Number of agents: {agent.num_agents}")
    print(f"Dyna-K value: {agent.dyna_k}")
    
    states = env.reset(1)
    
    for episode in range(10):
        states = env.reset(1)
        total_reward = 0.0
        
        for t in range(5):
            upper_actions = agent.upper_act(states)
            lower_actions = agent.lower_act(states)
            
            full_actions = []
            for i in range(config.N):
                full_action = np.concatenate([upper_actions[i], lower_actions[i]])
                full_actions.append(full_action)
            
            full_actions = np.array(full_actions)
            next_states, rewards, done = env.step(full_actions)
            
            for i in range(config.N):
                agent.add_lower_memory(i, states[i], lower_actions[i], rewards[i], next_states[i], done)
                agent.update_lower(i)
                agent.update_model(i)
                agent.dyna_plan(i, k=3)
            
            total_reward += np.sum(rewards)
            states = next_states
        
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}")
    
    print("\nDyna-Q test completed successfully!")
    
    for i in range(config.N):
        print(f"\nAgent {i}:")
        print(f"  Lower memory size: {len(agent.lower_memory[i])}")
        print(f"  Model network: {type(agent.models[i])}")
        print(f"  Lower DQN: {type(agent.lower_dqns[i])}")

if __name__ == '__main__':
    test_dyna_q()