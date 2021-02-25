import argparse
from gym.spaces import Dict, Tuple, Box, Discrete

import ray
import ray.tune as tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.tests.test_rollout_worker import MockPolicy
from ray.rllib.utils.spaces.repeated import Repeated
from ray.tune.registry import register_env
from ray.rllib.examples.env.nested_space_repeat_after_me_env import \
    NestedSpaceRepeatAfterMeEnv
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.agents import pg

from ray_scratch.my_random_policy import MyRandomPolicy

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--torch", action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-reward", type=float, default=0.0)
parser.add_argument("--stop-iters", type=int, default=100)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--num-cpus", type=int, default=0)



if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(num_cpus=args.num_cpus or None)
    register_env("NestedSpaceRepeatAfterMeEnv",
                 lambda c: NestedSpaceRepeatAfterMeEnv(c))

    config = {
        "env": "NestedSpaceRepeatAfterMeEnv",
        "env_config": {
            "space": Dict({
                "a": Tuple(
                    [Dict({
                        "d": Box(-10.0, 10.0, ()),
                        "e": Discrete(2)
                    })]),
                "b": Box(-10.0, 10.0, (2, )),
                # "c": Repeated(Discrete(4), 7)
            }),
        },
        # "entropy_coeff": 0.00005,  # We don't want high entropy in this Env.
        "gamma": 0.0,  # No history in Env (bandit problem).
        "lr": 0.0005,
        "num_envs_per_worker": 1,
        # "num_sgd_iter": 1,
        "num_workers": 0,
        # "vf_loss_coeff": 0.01,
        "framework": "torch" if args.torch else "tfe",
    }

    stop = {
        "training_iteration": args.stop_iters,
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
    }

    # results = tune.run(args.run, config=config, stop=stop)

    agent_cls = build_trainer(name='MyPG', default_config=pg.DEFAULT_CONFIG,
                          default_policy=MyRandomPolicy, get_policy_class=None)
    trainer_obj = agent_cls(config=config)
    env = trainer_obj.workers.local_worker().env
    for _ in range(args.stop_iters):
        results = trainer_obj.train()


    ray.shutdown()
