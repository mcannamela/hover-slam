from gym.spaces import Discrete

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class MyRockPaperScissors(MultiAgentEnv):
    """Two-player environment for the famous rock paper scissors game.

    The observation is simply the last opponent action."""

    ROCK = 0
    PAPER = 1
    SCISSORS = 2

    def __init__(self, config):
        self.action_space = None
        self.observation_space = None
        self.player1 = "player1"
        self.player2 = "player2"
        self.last_move = None
        self.num_moves = 0
        self.n_matches = config.get('n_matches', 10)

        # For test-case inspections (compare both players' scores).
        self.player1_score = self.player2_score = 0

    def reset(self):
        self.last_move = (0, 0)
        self.num_moves = 0
        return {
            self.player1: self.last_move[1],
            self.player2: self.last_move[0],
        }

    def step(self, action_dict):
        move1 = action_dict[self.player1]
        move2 = action_dict[self.player2]

        self.last_move = (move1, move2)
        obs = {
            self.player1: self.last_move[1],
            self.player2: self.last_move[0],
        }
        r1, r2 = {
            (self.ROCK, self.ROCK): (0, 0),
            (self.ROCK, self.PAPER): (-1, 1),
            (self.ROCK, self.SCISSORS): (1, -1),
            (self.PAPER, self.ROCK): (1, -1),
            (self.PAPER, self.PAPER): (0, 0),
            (self.PAPER, self.SCISSORS): (-1, 1),
            (self.SCISSORS, self.ROCK): (-1, 1),
            (self.SCISSORS, self.PAPER): (1, -1),
            (self.SCISSORS, self.SCISSORS): (0, 0),
        }[move1, move2]
        rew = {
            self.player1: r1,
            self.player2: r2,
        }
        self.num_moves += 1
        done = {
            "__all__": self.num_moves >= self.n_matches,
        }

        if rew["player1"] > rew["player2"]:
            self.player1_score += 1
        elif rew["player2"] > rew["player1"]:
            self.player2_score += 1

        return obs, rew, done, {}
