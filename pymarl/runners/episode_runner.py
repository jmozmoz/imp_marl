import time

from pymarl.envs import REGISTRY as env_REGISTRY
from functools import partial
from pymarl.components.episode_buffer import EpisodeBatch
import numpy as np
from pymarl.modules.bandits.const_lr import Constant_Lr
from pymarl.modules.bandits.uniform import Uniform
from pymarl.modules.bandits.reinforce_hierarchial import EZ_agent as enza
from pymarl.modules.bandits.returns_bandit import ReturnsBandit as RBandit


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1
        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size,
                                 self.episode_limit + 1,
                                 preprocess=preprocess,
                                 device=self.args.device)
        self.mac = mac

        # Setup the noise distribution sampler
        if self.args.mac == "maven_mac":
            if self.args.noise_bandit:
                if self.args.bandit_policy:
                    self.noise_distrib = enza(self.args, logger=self.logger)
                else:
                    self.noise_distrib = RBandit(self.args, logger=self.logger)
            else:
                self.noise_distrib = Uniform(self.args)

        self.noise_returns = {}
        self.noise_test_won = {}
        self.noise_train_won = {}

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        if self.args.mac == "maven_mac":
            # Sample the noise at the beginning of the episode
            self.noise = self.noise_distrib.sample(self.batch['state'][:, 0],
                                                   test_mode)

            self.batch.update({"noise": self.noise}, ts=0)
        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1

            if self.args.mac == "is_mac":
                # Inserting policy experiences to the buffer
                actions, behavior = self.mac.select_actions(self.batch,
                                                            t_ep=self.t,
                                                            t_env=self.t_env,
                                                            test_mode=test_mode)
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t,
                                                  t_env=self.t_env,
                                                  test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [
                    (terminated != env_info.get("episode_limit", False),)],
            }
            if self.args.mac == "is_mac":
                post_transition_data["behavior"] = behavior

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state

        if self.args.mac == "is_mac":
            # Inserting policy experiences to the buffer
            actions, behavior = self.mac.select_actions(self.batch,
                                                        t_ep=self.t,
                                                        t_env=self.t_env,
                                                        test_mode=test_mode)
            self.batch.update({"actions": actions, "behavior":behavior},
                              ts=self.t)
        else:
            actions = self.mac.select_actions(self.batch, t_ep=self.t,
                                              t_env=self.t_env,
                                              test_mode=test_mode)
            self.batch.update({"actions": actions, }, ts=self.t)



        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns

        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in
                          set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon",
                                     self.mac.action_selector.epsilon,
                                     self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def save_models(self, path):
        if self.args.noise_bandit:
            self.noise_distrib.save_model(path)

    def load_models(self, path):
        if self.args.mac == "maven_mac" and self.args.noise_bandit:
            self.noise_distrib.load_model(path)

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns),
                             self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns),
                             self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean",
                                     v / stats["n_episodes"], self.t_env)
        stats.clear()
