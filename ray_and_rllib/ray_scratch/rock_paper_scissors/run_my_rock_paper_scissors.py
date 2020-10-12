"""A simple multi-agent env with two agents playing rock paper scissors.

This demonstrates running the following policies in competition:
    (1) heuristic policy of repeating the same move
    (2) heuristic policy of beating the last opponent move
    (3) LSTM/feedforward PG policies
    (4) LSTM policy with custom entropy loss
"""

import argparse
from gym.spaces import Discrete

from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.examples.env.rock_paper_scissors import RockPaperScissors
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved

from ray_scratch.rock_paper_scissors.policies import MyAlwaysSameHeuristic, MyBeatLastHeuristic
from ray_scratch.rock_paper_scissors.tf_model import MyFastModel, MyFastEagerModel

tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument("--torch", action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=60)
parser.add_argument("--stop-reward", type=float, default=1000.0)
parser.add_argument("--stop-timesteps", type=int, default=100000)

import random


def run_same_policy(args, stop):
    """Use the same policy for both agents (trivial case)."""
    config = {
        "env": RockPaperScissors,
        "framework": "torch" if args.torch else "tf",
    }

    results = tune.run("PG", config=config, stop=stop)

    if args.as_test:
        # Check vs 0.0 as we are playing a zero-sum game.
        check_learning_achieved(results, 0.0)


def run_heuristic_vs_learned(args, use_lstm=False, trainer="PG"):
    """Run heuristic policies vs a learned agent.

    The learned agent should eventually reach a reward of ~5 with
    use_lstm=False, and ~7 with use_lstm=True. The reason the LSTM policy
    can perform better is since it can distinguish between the always_same vs
    beat_last heuristics.
    """

    def select_policy(agent_id):
        # return {
        #     'player1': 'always_same',
        #     'player2': 'beat_last'
        # }[agent_id]
        if agent_id == "player1":
            return "learned"
        else:
            return random.choice(["always_same", "beat_last"])

    rps_space = Discrete(3)

    ModelCatalog.register_custom_model("my_fast_model", MyFastModel)
    ModelCatalog.register_custom_model("my_fast_eager_model", MyFastEagerModel)

    config = {
        "env": RockPaperScissors,
        "gamma": 0.9,
        "num_workers": 0,
        "num_envs_per_worker": 4,
        "rollout_fragment_length": 10,
        "train_batch_size": 200,
        "multiagent": {
            # "policies_to_train": ["learned"],
            "policies": {
                "always_same": (MyAlwaysSameHeuristic, rps_space, rps_space,
                                {}),
                "beat_last": (MyBeatLastHeuristic, rps_space, rps_space, {}),
                "learned": (None, Discrete(3), Discrete(3), {
                    "model": {
                        "custom_model": 'my_fast_model'
                    },
                    "framework": "torch" if args.torch else "tf",
                }),
            },
            "policy_mapping_fn": select_policy,
        },
        "framework": "torch" if args.torch else "tf",
    }
    cls = get_agent_class(trainer) if isinstance(trainer, str) else trainer
    trainer_obj = cls(config=config)
    env = trainer_obj.workers.local_worker().env
    for _ in range(args.stop_iters):
        results = trainer_obj.train()
        print(results['policy_reward_mean'])
        # Timesteps reached.
        if results["timesteps_total"] > args.stop_timesteps:
            break
        # Reward (difference) reached -> all good, return.
        elif env.player1_score - env.player2_score > args.stop_reward:
            return

    # Reward (difference) not reached: Error if `as_test`.
    if args.as_test:
        raise ValueError(
            "Desired reward difference ({}) not reached! Only got to {}.".
            format(args.stop_reward, env.player1_score - env.player2_score))


if __name__ == "__main__":
    args = parser.parse_args()

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    # run_same_policy(args, stop=stop)
    # print("run_same_policy: ok.")

    run_heuristic_vs_learned(args)
    print("run_heuristic_vs_learned(w/o lstm): ok.")

    # run_heuristic_vs_learned(args, use_lstm=True)
    # print("run_heuristic_vs_learned (w/ lstm): ok.")
    #
    # run_with_custom_entropy_loss(args, stop=stop)
    # print("run_with_custom_entropy_loss: ok.")
