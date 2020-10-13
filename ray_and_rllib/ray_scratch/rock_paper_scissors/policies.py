import random
import tensorflow as tf
from ray.rllib import Policy
import numpy as np
from ray_scratch.rock_paper_scissors.my_rock_paper_scissors import MyRockPaperScissors
from scipy.special import softmax, logit

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
        self._logits = np.array([-10, -10, 10], dtype=np.float32)
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
                return np.random.choice(self._choices, p=softmax(self._logits))
            except ValueError:
                print(self._logits)
                raise

        return [choose(x) for x in obs_batch], [], {}

    def learn_on_batch(self, samples):
        obs_probs = np.mean(samples.data[samples.OBS], axis=0)
        # print('obs:', obs_probs)
        # print('bef:', softmax(self._logits))
        win_probs = np.roll(obs_probs, shift=1)
        win_logits = np.clip(logit(win_probs), -20, 20)
        update_direction = win_logits - self._logits
        self._logits = self._logits + self._learning_rate*update_direction

        # print(obs_probs)
        print('aft:',softmax(self._logits),'\n')
        # print('log:',self._logits)
        # print('dir:', update_direction)
        # print(self._probs)
        # print(win_probs)

        # print('    ', self._learning_rate * update_direction)

    def get_weights(self):
        return self._logits

    def set_weights(self, weights):
        self._logits = weights
