# Create file: control_latency_wrapper.py
import torch
from collections import defaultdict, deque
from isaaclab.envs import ManagerBasedRLEnv
import gym

class ControlLatencyWrapper:
    """Wrapper to add control latency to Isaac Lab environments."""
    
    def __init__(self, env: ManagerBasedRLEnv):
        self.env = env
        self.action_delay_buffers = defaultdict(lambda: deque())
        self.action_delays = {}
        
    def step(self, actions: torch.Tensor):
        """Apply actions with simulated latency."""
        if hasattr(self.env, 'action_delays'):
            delayed_actions = self._apply_latency(actions)
        else:
            delayed_actions = actions
            
        return self.env.step(delayed_actions)
    
    def _apply_latency(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply control latency to actions."""
        delayed_actions = actions.clone()
        
        for env_idx in range(self.env.num_envs):
            if env_idx in self.env.action_delays:
                delay = self.env.action_delays[env_idx]
                buffer = self.action_delay_buffers[env_idx]
                
                # Add current action to buffer
                buffer.append(actions[env_idx].clone())
                
                # Apply delayed action if buffer has enough elements
                if len(buffer) > delay:
                    delayed_actions[env_idx] = buffer.popleft()
        
        return delayed_actions
    
    def __getattr__(self, name):
        """Delegate all other attributes to wrapped environment."""
        return getattr(self.env, name)

# Usage in your training/play scripts:
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
env = ControlLatencyWrapper(env)  # Wrap the environment