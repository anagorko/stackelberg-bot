import logging
import random
import time

import numpy as np

import bsgmc
import security_games


class Awesome(bsgmc.Algorithm):
    tag = "awesome"
    series = ("lb",)

    def solve(self, game: security_games.BayesianStackelbergGame, time_limit: int) -> dict[str, bsgmc.TimeSeries]:
        random.seed(self.seed)
        rng = np.random.default_rng(self.seed)

        x = []
        y = []
        max_y = float('-inf')

        start_time = time.process_time()
        while time.process_time() - start_time < time_limit:
            mv = rng.dirichlet(np.ones(len(game.X)), size=1)[0]
            strat = {
                s: mv[i] for i, s in enumerate(game.X)
            }
            payoff = game.expected_reward(strat)
            if payoff > max_y:
                x.append(time.process_time() - start_time)
                y.append(payoff)
                max_y = payoff
                print(max_y)

        x.append(time.process_time() - start_time)
        y.append(max_y)

        return {"lb": bsgmc.TimeSeries(x=np.array(x), y=np.array(y))}

    def run(self, gm: bsgmc.GameMetadata, time_limit: int) -> bsgmc.ExperimentResult:
        logging.info("Running Awesome with %ss time limit.", time_limit)

        game = gm.get()
        start_time = time.process_time()
        result = self.solve(game, time_limit)
        elapsed_time = time.process_time() - start_time

        if elapsed_time < time_limit:
            elapsed_time = float('inf')

        return bsgmc.ExperimentResult(
            elapsed_time=elapsed_time,
            series=result,
        )
