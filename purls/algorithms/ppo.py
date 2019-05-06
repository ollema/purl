import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces.discrete import Discrete
from tensorboardX import SummaryWriter
from torch.distributions.categorical import Categorical

from purls.algorithms.base import ReinforcementLearningAlgorithm
from purls.utils import seed_all
from purls.utils.envs import SubprocVecEnv
from purls.utils.logs import debug
from purls.utils.plotters import MultiprocessPlotter
from purls.utils.trackers import TensorboardTracker

# TODO: fix device mismatch
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


def preprocess_observations(observations):
    observations = np.array([obs["image"] for obs in observations])
    return torch.tensor(observations, device=device, dtype=torch.float)


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64
        self.actor = nn.Sequential(nn.Linear(self.embedding_size, 64), nn.Tanh(), nn.Linear(64, 3))
        self.critic = nn.Sequential(nn.Linear(self.embedding_size, 64), nn.Tanh(), nn.Linear(64, 1))
        self.apply(initialize_parameters)

    def forward(self, obs):
        x = torch.transpose(torch.transpose(obs, 1, 3), 2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        embedding = x

        x = self.actor(embedding)
        policy = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return policy, value


class ppo(ReinforcementLearningAlgorithm):
    def __init__(self, env, args):
        super().__init__(
            env,
            args,
            # default values for this algorithm
            default_learning_rate=7e-4,
            default_discount_factor=0.99,
            default_num_updates=10 ** 7,
        )
        # default values which are currently unavailable from the cmdline. maybe add these as flags
        self.processes = 16
        self.frames_per_process = 128
        self.lam = 0.95
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        self.optim_eps = 1e-5
        self.clip_eps = 0.2
        self.epochs = 4
        self.batch_size = 256
        self.seed = 666 if not self.seed else self.seed

        self.frames_per_update = self.frames_per_process * self.processes
        self.update_shape = (self.frames_per_process, self.processes)
        seed_all(self.seed)

        obs_space = {"image": self.env.observation_space.spaces["image"].shape}
        self.model = {"acmodel": ACModel(obs_space, Discrete(3)).to(device)}

    def train(self):
        writer = SummaryWriter(comment=f"-{self.model_name}") if self.tensorboard else None

        self.acmodel = self.model["acmodel"]
        self.acmodel.train()
        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), self.lr, eps=1e-5)

        self.envs = []
        for i in range(self.processes):
            env = gym.make(self.env.spec.id)
            env.seed(self.seed + i * 100)
            self.envs.append(env)
        self.env = SubprocVecEnv(self.envs)
        self.last_observation = self.env.reset()
        self.last_done = torch.ones(self.update_shape[1], device=device)

        # TODO: these three need to be removed!
        self.log_episode_num_frames = torch.zeros(self.processes, device=device)
        self.log_reward = [0] * self.processes
        self.log_num_frames_per_episode = [0] * self.processes

        with TensorboardTracker(writer) as tracker:
            if self.render_interval != 0:
                self.renderer = MultiprocessPlotter(self.processes, self.env)

            for i in range(1, self.max_num_updates + 1):
                if hasattr(self, "renderer") and self.renderer.closed:
                    self.renderer.create_figure()

                render = hasattr(self, "renderer") and i % self.render_interval == 0
                update_start_time = time.time()

                # for N=`self.processes` actors, run policy π_old in environment for T=`self.frames_per_process` timesteps
                observations, rewards, dones, actions, log_probs, values, reward_per_episode, frames_per_episode = self.run_policy(
                    render
                )

                # compute advantage estimates A1, ..., AT
                advantages = self.compute_advantage_estimates(
                    observations, rewards, dones, actions, values
                )

                # optimize surrogate L wrt θ, with K=`self.epochs` epochs and minibatch size M ≤ NT
                # and then update θ_old <- θ
                entropy, policy_loss, value_loss, grad_norm = self.optimize_parameters(
                    observations, rewards, dones, actions, log_probs, values, advantages
                )

                update_end_time = time.time()
                fps = self.frames_per_update / (update_end_time - update_start_time)
                mean_reward = np.mean(reward_per_episode)
                mean_frames_per_episode = np.mean(frames_per_episode)

                tracker.push(
                    update=i,
                    debug_message=f"update {i:6d} - avg. reward: {mean_reward:2f}",
                    fps=fps,
                    reward=mean_reward,
                    frames_per_episode=mean_frames_per_episode,
                    entropy=entropy,
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                    grad_norm=grad_norm,
                )

                if self.save_interval != 0 and i % self.save_interval == 0:
                    self.save()

    def select_action(self, obs):
        preprocessed_obs = preprocess_observations(obs)
        with torch.no_grad():
            policy, value = self.acmodel(preprocessed_obs)
        return policy.sample(), policy, value

    def run_policy(self, render):
        self.done_counter = 0

        observations = [None] * (self.update_shape[0])
        rewards = torch.zeros(*self.update_shape, device=device)
        dones = torch.zeros(*self.update_shape, device=device)
        actions = torch.zeros(*self.update_shape, device=device, dtype=torch.int)
        log_probs = torch.zeros(*self.update_shape, device=device)
        values = torch.zeros(*self.update_shape, device=device)

        for i in range(self.frames_per_process):
            # do one agent-environment interaction
            action, policy, value = self.select_action(self.last_observation)
            observation, reward, done, _ = self.env.step(action.cpu().numpy())
            observations[i] = self.last_observation
            rewards[i] = torch.tensor(reward, device=device)
            dones[i] = self.last_done
            actions[i] = action
            log_probs[i] = policy.log_prob(action)  # this is just log(probabilities)!
            values[i] = value
            self.last_observation = observation
            self.last_done = torch.tensor(done, device=device, dtype=torch.float)

            # update log values TODO: I hate this...
            self.log_episode_num_frames += torch.ones(self.processes, device=device)
            for j, done_ in enumerate(done):
                if done_:
                    self.done_counter += 1
                    self.log_reward.append(rewards[i][j].item())
                    self.log_num_frames_per_episode.append(self.log_episode_num_frames[j].item())
            self.log_episode_num_frames *= 1 - self.last_done

            if render and not self.renderer.closed:
                self.renderer.render(i)

        observations = [
            observations[i][j]
            for j in range(self.processes)
            for i in range(self.frames_per_process)
        ]
        observations = preprocess_observations(observations)

        keep = max(self.done_counter, self.processes)
        self.log_reward = self.log_reward[-self.processes :]
        self.log_num_frames_per_episode = self.log_num_frames_per_episode[-self.processes :]

        return (
            observations,
            rewards,
            dones,
            actions,
            log_probs,
            values,
            self.log_reward[-keep:],
            self.log_num_frames_per_episode[-keep:],
        )

    def compute_advantage_estimates(self, observations, rewards, dones, actions, values):
        advantages = torch.zeros(*self.update_shape, device=device)

        preprocessed_last_obs = preprocess_observations(self.last_observation)
        with torch.no_grad():
            _, last_value = self.acmodel(preprocessed_last_obs)

        last_adv = 0
        # compute advantage estimates A1, ..., AT
        for i in reversed(range(self.frames_per_process)):
            # special case for the last_observation (which we handle first since "reversed")
            if i == self.frames_per_process - 1:
                next_nonterminal = 1.0 - self.last_done
                next_value = last_value
            else:
                next_nonterminal = 1.0 - dones[i + 1]
                next_value = values[i + 1]

            # δ_t = r_t + γV(s_{t+1}) − V(s_t)
            delta = rewards[i] + self.y * next_value * next_nonterminal - values[i]
            # truncated version of generalized advantage estimation
            advantages[i] = last_adv = delta + self.y * self.lam * next_nonterminal * last_adv

        return advantages

    def optimize_parameters(
        self, observations, rewards, dones, actions, log_probs, values, advantages
    ):
        # reshape experiences
        actions = actions.transpose(0, 1).reshape(-1)
        log_probs = log_probs.transpose(0, 1).reshape(-1)
        values = values.transpose(0, 1).reshape(-1)
        advantages = advantages.transpose(0, 1).reshape(-1)
        returns = values + advantages

        entropies = []
        policy_losses = []
        value_losses = []
        grad_norms = []

        for _ in range(self.epochs):

            # randomize order of mini-batches
            indexes = np.arange(0, self.frames_per_update)
            indexes = np.random.permutation(indexes)
            num_indexes = self.batch_size
            for inds in [indexes[i : i + num_indexes] for i in range(0, len(indexes), num_indexes)]:

                # mb_ stands for mini-batch
                mb_observations = observations[inds]
                mb_actions = actions[inds]
                mb_log_probs = log_probs[inds]
                mb_values = values[inds]
                mb_advantages = advantages[inds]
                mb_returns = returns[inds]

                # forward pass with mini-batch of observations
                policy, value = self.acmodel(mb_observations)
                # entropy is used as a bonus to ensure sufficient exploration
                entropy = policy.entropy().mean()

                # calculate probability ratio r_t(θ)
                ratio = torch.exp(policy.log_prob(mb_actions) - mb_log_probs)
                # calculate the first surrogate objective: L^{CLI}
                surr1 = ratio * mb_advantages
                # calculate the second surrogate objective, a clipped version of L^{CLI}
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                # policy loss is the mean of the -minimum of the two surrogate objectives: (L^{CLIP})
                policy_loss = -torch.min(surr1, surr2).mean()

                # clip the value diff
                value_clipped = mb_values + torch.clamp(
                    value - mb_values, -self.clip_eps, self.clip_eps
                )
                # calculate the first surrogate objective: squared error
                surr1 = (value - mb_returns).pow(2)
                # calculate the second surrogate objective: squared error of clipped value
                surr2 = (value_clipped - mb_returns).pow(2)
                # value loss is the mean of the maximum of the two surrogate objectives
                value_loss = torch.max(surr1, surr2).mean()

                # finally, our loss/objective function is defined as the policy loss + c1 * value_loss - c2 * entropy
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                # optimize and update params
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = (
                    sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                )
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                entropies.append(entropy.item())
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                grad_norms.append(grad_norm)

        return (
            np.mean(entropies),
            np.mean(policy_losses),
            np.mean(value_losses),
            np.mean(grad_norms),
        )

    def visualize(self):
        self.model = self.load()
        self.acmodel = self.model["acmodel"]
        self.acmodel.eval()

        env = gym.make(self.env.spec.id)
        env.seed(self.seed)
        self.env = SubprocVecEnv([env])

        self.obs = self.env.reset()
        done = False

        while True:
            action, _, _ = self.select_action(self.obs)
            obs, reward, done, _ = self.env.step(action.cpu().numpy())
            self.obs = obs
            self.env.render()
            time.sleep(1 / self.fps)

            if done[0]:
                debug(f"reward: {reward[0]}")
