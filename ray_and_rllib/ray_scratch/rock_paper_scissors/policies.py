import random
import tensorflow as tf
from ray.rllib import Policy

from ray_scratch.rock_paper_scissors.my_rock_paper_scissors import MyRockPaperScissors


class MyAlwaysSameHeuristic(Policy):
    """Pick a random move and stick with it for the entire episode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration = self._create_exploration()

    def get_initial_state(self):
        return [
            random.choice([
                MyRockPaperScissors.ROCK, MyRockPaperScissors.PAPER,
                MyRockPaperScissors.SCISSORS
            ])
        ]

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        return state_batches[0], state_batches, {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass


class MyBeatLastHeuristic(Policy):
    """Play the move that would beat the last move of the opponent."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration = self._create_exploration()

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        def successor(x):
            if x[MyRockPaperScissors.ROCK] == 1:
                return MyRockPaperScissors.PAPER
            elif x[MyRockPaperScissors.PAPER] == 1:
                return MyRockPaperScissors.SCISSORS
            elif x[MyRockPaperScissors.SCISSORS] == 1:
                return MyRockPaperScissors.ROCK

        return [successor(x) for x in obs_batch], [], {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass

class MixedRPS(Policy):
    """Draw at random with fixed weights"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        def successor(x):
            if x[MyRockPaperScissors.ROCK] == 1:
                return MyRockPaperScissors.PAPER
            elif x[MyRockPaperScissors.PAPER] == 1:
                return MyRockPaperScissors.SCISSORS
            elif x[MyRockPaperScissors.SCISSORS] == 1:
                return MyRockPaperScissors.ROCK

        return [successor(x) for x in obs_batch], [], {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass