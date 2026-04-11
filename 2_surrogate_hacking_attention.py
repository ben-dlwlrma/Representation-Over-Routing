"""
2_surrogate_hacking_attention.py: Multi-timescale PPO.
Demonstrates Surrogate Objective Hacking via a state-dependent attention mechanism 
exposed to the policy gradients.
"""

import time
import random
from dataclasses import dataclass, field
from typing import List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = "ppo_multi_gamma_attention"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    env_id: str = "LunarLander-v2"
    num_envs: int = 8

    total_timesteps: int = 1000000
    num_steps: int = 256

    learning_rate: float = 2.5e-4
    anneal_lr: bool = True

    gammas: List[float] = field(default_factory=lambda: [0.5, 0.9, 0.99, 0.999])
    num_gammas: int = 4
    gae_lambda: float = 0.95

    num_minibatches: int = 8
    update_epochs: int = 10
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    clip_vloss: bool = True

    batch_size: int = 0
    minibatch_size: int = 0
    num_updates: int = 0


def make_env(env_id, idx, seed):
    """Return a thunk that creates a single gymnasium environment."""

    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialisation for a linear layer."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
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
        # [EVOLUTION: SURROGATE HACKING] Differentiable attention mechanism added to route multi-timescale signals.
        self.attention = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, num_gammas), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_attention_weights(self, x):
        return torch.softmax(self.attention(x), dim=-1)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(x)


if __name__ == "__main__":
    args = Args()
    assert len(args.gammas) == args.num_gammas, "len(args.gammas) must equal args.num_gammas"
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_updates = args.total_timesteps // args.batch_size

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.seed) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)

    obs_dim = np.array(envs.single_observation_space.shape).prod()
    act_dim = envs.single_action_space.n
    gamma_tensor = torch.tensor(args.gammas, dtype=torch.float32, device=device).view(1, args.num_gammas)

    agent = ActorCritic(obs_dim, act_dim, args.num_gammas).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs_buf = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
        dtype=torch.float32,
        device=device,
    )
    act_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=device)
    logp_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    reward_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    done_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    val_buf = torch.zeros(
        (args.num_steps, args.num_envs, args.num_gammas),
        dtype=torch.float32,
        device=device,
    )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)

    for update in range(1, args.num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / args.num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs
            done_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)

            act_buf[step] = action
            logp_buf[step] = logprob
            val_buf[step] = value

            next_obs_np, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(terminated | truncated, dtype=torch.float32, device=device)
            reward_buf[step] = torch.tensor(reward, dtype=torch.float32, device=device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None and "episode" in info:
                        ep_return = info["episode"]["r"]
                        ep_length = info["episode"]["l"]
                        if isinstance(ep_return, np.ndarray):
                            ep_return = ep_return.item()
                        if isinstance(ep_length, np.ndarray):
                            ep_length = ep_length.item()
                        print(f"global_step={global_step}, episodic_return={ep_return:.1f}")
                        writer.add_scalar("charts/episodic_return", ep_return, global_step)
                        writer.add_scalar("charts/episodic_length", ep_length, global_step)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).view(args.num_envs, args.num_gammas)
            advantages = torch.zeros(
                (args.num_steps, args.num_envs, args.num_gammas),
                dtype=torch.float32,
                device=device,
            )
            last_gae = torch.zeros((args.num_envs, args.num_gammas), dtype=torch.float32, device=device)

            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_non_terminal = (1.0 - next_done).view(args.num_envs, 1)
                    next_values = next_value
                else:
                    next_non_terminal = (1.0 - done_buf[t + 1]).view(args.num_envs, 1)
                    next_values = val_buf[t + 1]

                reward_t = reward_buf[t].view(args.num_envs, 1)
                delta = reward_t + gamma_tensor * next_values * next_non_terminal - val_buf[t]
                last_gae = delta + gamma_tensor * args.gae_lambda * next_non_terminal * last_gae
                advantages[t] = last_gae

            returns = advantages + val_buf

        b_obs = obs_buf.reshape((-1,) + envs.single_observation_space.shape)
        b_logp = logp_buf.reshape(-1)
        b_act = act_buf.reshape(-1)
        b_adv = advantages.reshape(-1, args.num_gammas)
        b_ret = returns.reshape(-1, args.num_gammas)
        b_val = val_buf.reshape(-1, args.num_gammas)

        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb = b_inds[start:end]

                _, newlogp, entropy, newval = agent.get_action_and_value(b_obs[mb], b_act[mb])
                attention_weights = agent.get_attention_weights(b_obs[mb])
                logratio = newlogp - b_logp[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                # [EVOLUTION: SURROGATE HACKING] Attention weights are differentiable and multiply 
                # the advantages, exposing the routing mechanism to policy gradients and leading to objective hacking.
                mb_adv = b_adv[mb]
                mb_adv_dynamic = (mb_adv * attention_weights).sum(dim=1)
                if args.norm_adv:
                    mb_adv_dynamic = (mb_adv_dynamic - mb_adv_dynamic.mean()) / (mb_adv_dynamic.std() + 1e-8)

                pg_loss1 = -mb_adv_dynamic * ratio
                pg_loss2 = -mb_adv_dynamic * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newval = newval.view(-1, args.num_gammas)
                if args.clip_vloss:
                    v_loss_unclipped = (newval - b_ret[mb]) ** 2
                    v_clipped = b_val[mb] + torch.clamp(
                        newval - b_val[mb], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_ret[mb]) ** 2
                    v_loss_per_head = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean(dim=0)
                else:
                    v_loss_per_head = 0.5 * ((newval - b_ret[mb]) ** 2).mean(dim=0)
                v_loss = v_loss_per_head.mean()

                ent_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * ent_loss + args.vf_coef * v_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", ent_loss.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        if update % 10 == 0:
            print(f"update {update}/{args.num_updates}  SPS={sps}")

    envs.close()
    writer.close()
    print("training complete")
