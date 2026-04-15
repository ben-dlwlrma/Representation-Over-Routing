import os
import time
import random
from typing import List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int):
    """Seed all sources of randomness for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def make_env(env_id, idx, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env
    return thunk


# baseline model
class BaselineActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(x)


# target decoupling model
class DecouplingActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, num_gammas):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, num_gammas), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(x)


# training loop
def train_agent(agent_type: str, seed: int, num_episodes: int = 1000):
    set_global_seed(seed)
    
    env_id = "LunarLander-v2"
    num_envs = 8
    num_steps = 256
    learning_rate = 2.5e-4
    gae_lambda = 0.95
    norm_adv = True
    clip_coef = 0.2
    clip_vloss = True
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    update_epochs = 10
    num_minibatches = 8
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches

    baseline_gamma = 0.99
    decoupling_gammas = [0.5, 0.9, 0.99, 0.999]
    num_gammas = len(decoupling_gammas)

    envs = gym.vector.SyncVectorEnv([make_env(env_id, i, seed) for i in range(num_envs)])
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    act_dim = envs.single_action_space.n

    if agent_type == "baseline":
        agent = BaselineActorCritic(obs_dim, act_dim).to(device)
    elif agent_type == "decoupling":
        agent = DecouplingActorCritic(obs_dim, act_dim, num_gammas).to(device)
        gamma_tensor = torch.tensor(decoupling_gammas, dtype=torch.float32, device=device).view(1, num_gammas)
    else:
        raise ValueError("Invalid agent_type")

    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    obs_buf = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape, device=device)
    act_buf = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
    logp_buf = torch.zeros((num_steps, num_envs), device=device)
    reward_buf = torch.zeros((num_steps, num_envs), device=device)
    done_buf = torch.zeros((num_steps, num_envs), device=device)

    if agent_type == "baseline":
        val_buf = torch.zeros((num_steps, num_envs), device=device)
    else:
        val_buf = torch.zeros((num_steps, num_envs, num_gammas), device=device)

    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(num_envs, device=device)

    episodic_rewards = []
    
    # Run indefinitely until num_episodes is reached
    while len(episodic_rewards) < num_episodes:
        for step in range(num_steps):
            obs_buf[step] = next_obs
            done_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            
            act_buf[step] = action
            logp_buf[step] = logprob
            if agent_type == "baseline":
                val_buf[step] = value.flatten()
            else:
                val_buf[step] = value

            next_obs_np, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(terminated | truncated, dtype=torch.float32, device=device)
            reward_buf[step] = torch.tensor(reward, dtype=torch.float32, device=device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None and "episode" in info:
                        ep_return = info["episode"]["r"]
                        if isinstance(ep_return, np.ndarray):
                            ep_return = ep_return.item()
                        episodic_rewards.append(ep_return)
                        
                        if len(episodic_rewards) >= num_episodes:
                            break
            if len(episodic_rewards) >= num_episodes:
                break

        if len(episodic_rewards) >= num_episodes:
            break

        # GAE / PPO Update
        with torch.no_grad():
            if agent_type == "baseline":
                next_value = agent.get_value(next_obs).view(-1)
                advantages = torch.zeros_like(reward_buf, device=device)
                last_gae = torch.zeros_like(next_done, device=device)
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        next_non_terminal = 1.0 - next_done
                        next_values = next_value
                    else:
                        next_non_terminal = 1.0 - done_buf[t + 1]
                        next_values = val_buf[t + 1]
                    delta = reward_buf[t] + baseline_gamma * next_values * next_non_terminal - val_buf[t]
                    last_gae = delta + baseline_gamma * gae_lambda * next_non_terminal * last_gae
                    advantages[t] = last_gae
                returns = advantages + val_buf
                
            else:  # Decoupling
                next_value = agent.get_value(next_obs).view(num_envs, num_gammas)
                advantages = torch.zeros((num_steps, num_envs, num_gammas), dtype=torch.float32, device=device)
                last_gae = torch.zeros((num_envs, num_gammas), dtype=torch.float32, device=device)

                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        next_non_terminal = (1.0 - next_done).view(num_envs, 1)
                        next_values = next_value
                    else:
                        next_non_terminal = (1.0 - done_buf[t + 1]).view(num_envs, 1)
                        next_values = val_buf[t + 1]

                    reward_t = reward_buf[t].view(num_envs, 1)
                    delta = reward_t + gamma_tensor * next_values * next_non_terminal - val_buf[t]
                    last_gae = delta + gamma_tensor * gae_lambda * next_non_terminal * last_gae
                    advantages[t] = last_gae

                returns = advantages + val_buf

        # Flat batch
        b_obs = obs_buf.reshape((-1,) + envs.single_observation_space.shape)
        b_logp = logp_buf.reshape(-1)
        b_act = act_buf.reshape(-1)
        
        if agent_type == "baseline":
            b_adv = advantages.reshape(-1)
            b_ret = returns.reshape(-1)
            b_val = val_buf.reshape(-1)
        else:
            b_adv = advantages.reshape(-1, num_gammas)
            b_ret = returns.reshape(-1, num_gammas)
            b_val = val_buf.reshape(-1, num_gammas)
            target_gamma_idx = 3
            b_adv_aggregated = b_adv[:, target_gamma_idx]

        b_inds = np.arange(batch_size)

        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb = b_inds[start:end]

                _, newlogp, entropy, newval = agent.get_action_and_value(b_obs[mb], b_act[mb])
                logratio = newlogp - b_logp[mb]
                ratio = logratio.exp()
                
                if agent_type == "baseline":
                    mb_adv = b_adv[mb]
                    if norm_adv:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                        
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newval = newval.view(-1)
                    if clip_vloss:
                        v_loss_unclipped = (newval - b_ret[mb]) ** 2
                        v_clipped = b_val[mb] + torch.clamp(newval - b_val[mb], -clip_coef, clip_coef)
                        v_loss_clipped = (v_clipped - b_ret[mb]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((newval - b_ret[mb]) ** 2).mean()
                else:
                    mb_adv_dynamic = b_adv_aggregated[mb]
                    if norm_adv:
                        mb_adv_dynamic = (mb_adv_dynamic - mb_adv_dynamic.mean()) / (mb_adv_dynamic.std() + 1e-8)
                        
                    newval = newval.view(-1, num_gammas)
                    v_target = b_ret[mb]

                    pg_loss1 = -mb_adv_dynamic * ratio
                    pg_loss2 = -mb_adv_dynamic * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    if clip_vloss:
                        v_loss_unclipped = (newval - v_target) ** 2
                        v_clipped = b_val[mb] + torch.clamp(newval - b_val[mb], -clip_coef, clip_coef)
                        v_loss_clipped = (v_clipped - v_target) ** 2
                        v_loss_per_head = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean(dim=0)
                    else:
                        v_loss_per_head = 0.5 * ((newval - v_target) ** 2).mean(dim=0)
                    v_loss = v_loss_per_head.mean()

                ent_loss = entropy.mean()
                loss = pg_loss - ent_coef * ent_loss + vf_coef * v_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

    envs.close()
    return episodic_rewards[:num_episodes]


def moving_average(data: List[float], window: int) -> np.ndarray:
    """Computes a simple moving average of a list."""
    if len(data) < window:
        window = len(data)
    padded = np.pad(data, (window - 1, 0), mode='edge')
    return np.convolve(padded, np.ones(window)/window, mode='valid')

if __name__ == "__main__":
    seeds = [42, 100, 123, 1024, 2026]
    num_episodes = 3000
    moving_avg_window = 50

    results_baseline = []
    results_decoupling = []
    
    print(f"Starting Evaluation for {num_episodes} episodes across {len(seeds)} seeds: {seeds}")

    for s in seeds:
        print(f"--- Training Baseline (Seed: {s}) ---")
        base_ep_rewards = train_agent("baseline", s, num_episodes)
        results_baseline.append(base_ep_rewards)

        print(f"--- Training Target Decoupling (Seed: {s}) ---")
        decoup_ep_rewards = train_agent("decoupling", s, num_episodes)
        results_decoupling.append(decoup_ep_rewards)

    results_baseline = np.array(results_baseline)
    results_decoupling = np.array(results_decoupling)

    # Calculate statistics
    mean_baseline = np.mean(results_baseline, axis=0)
    std_baseline = np.std(results_baseline, axis=0)

    mean_decoupling = np.mean(results_decoupling, axis=0)
    std_decoupling = np.std(results_decoupling, axis=0)

    # Smooth the resulting statistics
    smooth_mean_base = moving_average(mean_baseline, moving_avg_window)
    smooth_std_base = moving_average(std_baseline, moving_avg_window)
    
    smooth_mean_decoup = moving_average(mean_decoupling, moving_avg_window)
    smooth_std_decoup = moving_average(std_decoupling, moving_avg_window)

    episodes_x = np.arange(1, num_episodes + 1)

    # plotting
    plt.figure(figsize=(10, 6))

    # baseline plot
    plt.plot(episodes_x, smooth_mean_base, label="Baseline", color="red", linewidth=2)
    plt.fill_between(
        episodes_x, 
        smooth_mean_base - smooth_std_base, 
        smooth_mean_base + smooth_std_base, 
        color="red", 
        alpha=0.2
    )

    # target decoupling plot
    plt.plot(episodes_x, smooth_mean_decoup, label="Target Decoupling", color="blue", linewidth=2)
    plt.fill_between(
        episodes_x, 
        smooth_mean_decoup - smooth_std_decoup, 
        smooth_mean_decoup + smooth_std_decoup, 
        color="blue", 
        alpha=0.2
    )

    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Episodic Reward", fontsize=12)
    plt.title("Baseline vs Target Decoupling (5 Seeds)", fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    os.makedirs("docs", exist_ok=True)
    plot_path = os.path.join("docs", "seed_comparison_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot successfully saved to: {plot_path}")
