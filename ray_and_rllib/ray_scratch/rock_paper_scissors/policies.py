import random
import tensorflow as tf
from ray.rllib import Policy
import numpy as np
from ray_scratch.rock_paper_scissors.my_rock_paper_scissors import MyRockPaperScissors


class OnlyRock(Policy):
    """Pick a random move and stick with it for the entire episode."""

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
        return [MyRockPaperScissors.ROCK for x in obs_batch], [], {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass


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
    """Draw at random with fixed weights

    We update the weights using a heuristic.
    We look at the observed marginal probabilities of our opponent's moves, and then step our own probabilities
    in the direction that would counter these moves. So, if we observe 60% rock, 30% paper, and 10% scissors, we
    will take a step toward 10% rock, 60% paper, and 30% scissors, i.e. the marginal probabilities we would have
    observed had we won every match.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._probs = np.array([0, 0, 1], dtype=np.float32)
        self._choices = np.array([0, 1, 2])
        self._learning_rate = .1

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):

        def choose(x):
            try:
                return np.random.choice(self._choices, p=self._probs)
            except ValueError:
                print(self._probs)
                raise

        return [choose(x) for x in obs_batch], [], {}

    def learn_on_batch(self, samples):
        played_probs = np.mean(samples.data[samples.OBS], axis=0)
        win_probs = np.roll(played_probs, shift=1)
        update_direction = win_probs - self._probs
        self._probs = np.clip(self._probs + self._learning_rate * update_direction, 0, None)
        self._probs = self._probs / np.sum(self._probs)

        print(self._probs)
        print(win_probs)
        print('    ', self._learning_rate * update_direction)

    def get_weights(self):
        return self._probs

    def set_weights(self, weights):
        self._probs = weights
