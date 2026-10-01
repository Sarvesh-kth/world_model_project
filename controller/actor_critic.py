"""Task 8 (Grade C prep): an actor-critic trained on imagined rollouts. Skeleton only: the networks
and one training step are here and tested on the toy problem; it isn't trained on the task yet.

Same plug-in points as the planner: dynamics_fn(s, a) -> s_next (M2's D, or the state MLP) and
reward_fn(s, a) -> r (at Grade C: M2's reward model R). One update, Dreamer-style:

  1. start from a batch of real states s0
  2. imagine H steps: the actor picks actions, dynamics_fn predicts the next states
  3. reward_fn scores every imagined step; the critic values every imagined state
  4. lambda-returns mix the rewards with a slow-moving copy of the critic (the target critic;
     bootstrapping from the critic being trained diverged on the toy problem)
  5. critic: regress V(s_t) onto the (detached) lambda-returns
  6. actor: maximize the lambda-returns by backpropagating through the learned dynamics
     (needs a differentiable dynamics_fn, so not the oracle), plus a small entropy bonus

Grade C still has to decide: train only on imagined rollouts or mix in real data, the reward
model, and an exploration / novelty bonus (see the trajectory-diversity issue in CLAUDE.md).
"""

import copy

import torch
from torch import nn
from torch.distributions import Normal


class Actor(nn.Module):
    """Gaussian policy, squashed by tanh into M1's action range [-1, 1]."""

    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(),
                                 nn.Linear(hidden, 2 * action_dim))

    def forward(self, s, deterministic=False):
        """Returns (action [B, A] in [-1, 1], entropy of the unsquashed Gaussian [B])."""
        mean, log_std = self.net(s).chunk(2, dim=-1)
        dist = Normal(mean, log_std.clamp(-5, 2).exp())
        raw = mean if deterministic else dist.rsample()  # rsample: gradients flow through the sample
        return torch.tanh(raw), dist.entropy().sum(dim=-1)


class Critic(nn.Module):
    """State value V(s)."""

    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, s):
        return self.net(s).squeeze(-1)


def lambda_returns(rewards, values, gamma=0.99, lam=0.95):
    """TD(lambda) returns. rewards [H, B] for steps 0..H-1, values [H+1, B] for states 0..H.

    G_t = r_t + gamma * ((1 - lam) * V(s_{t+1}) + lam * G_{t+1}), with G_H = V(s_H).
    """
    returns, g = [], values[-1]
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * ((1 - lam) * values[t + 1] + lam * g)
        returns.append(g)
    return torch.stack(returns[::-1])


class ActorCritic:
    def __init__(self, state_dim, action_dim, dynamics_fn, reward_fn, horizon=10, gamma=0.99, lam=0.95,
                 actor_lr=3e-4, critic_lr=3e-4, entropy_weight=1e-3, target_tau=0.02, grad_clip=100.0):
        self.actor, self.critic = Actor(state_dim, action_dim), Critic(state_dim)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.target_tau, self.grad_clip = target_tau, grad_clip
        self.dynamics_fn, self.reward_fn = dynamics_fn, reward_fn
        self.horizon, self.gamma, self.lam, self.entropy_weight = horizon, gamma, lam, entropy_weight
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def imagine(self, s0):
        """Roll the actor through the model: states [H+1, B, S], rewards [H, B], entropies [H, B]."""
        states, rewards, entropies, s = [s0], [], [], s0
        for _ in range(self.horizon):
            a, entropy = self.actor(s)
            rewards.append(self.reward_fn(s, a))
            s = self.dynamics_fn(s, a)
            states.append(s)
            entropies.append(entropy)
        return torch.stack(states), torch.stack(rewards), torch.stack(entropies)

    def update(self, s0):
        """One actor and one critic gradient step from a batch of start states [B, S]."""
        states, rewards, entropies = self.imagine(s0)
        values = self.critic_target(states)  # gradients still flow through the states to the actor
        returns = lambda_returns(rewards, values, self.gamma, self.lam)

        actor_loss = -(returns.mean() + self.entropy_weight * entropies.mean())
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_opt.step()

        critic_loss = (self.critic(states[:-1].detach()) - returns.detach()).square().mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_opt.step()

        with torch.no_grad():  # the target critic slowly follows the critic
            for target, source in zip(self.critic_target.parameters(), self.critic.parameters()):
                target.lerp_(source, self.target_tau)
        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(),
                "imagined_reward": rewards.mean().item()}

    @torch.no_grad()
    def act(self, s):
        """Deterministic action [A] for one state [S] (for running the trained policy)."""
        return self.actor(s[None], deterministic=True)[0][0]
