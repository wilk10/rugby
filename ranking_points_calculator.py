import numpy
import pandas
import argparse
import scipy.stats


class RankingPoints:
    def __init__(
            self, home_name: str, away_name: str, home_points: float, away_points: float, no_home_factor: bool=False,
            world_cup: bool=False):
        self.home_name = home_name
        self.away_name = away_name
        self.home_points = home_points
        self.away_points = away_points
        self.home_factor = True if not no_home_factor else False
        self.home_factor_points = 3 if self.home_factor else 0
        self.world_cup = world_cup
        self.columns = ["large_win", "narrow_win", "draw", "narrow_loss", "large_loss"]
        self.df = pandas.DataFrame(index=[self.home_name, self.away_name], columns=self.columns)
        self.regression_results = self.init_regression_line()
        self.slope = self.regression_results.slope
        self.intercept = self.regression_results.intercept

    @staticmethod
    def init_regression_line():
        x = [-10, 10]
        y = [-1, 1]
        return scipy.stats.linregress(x, y)

    def calculate_and_display(self):
        raw_difference = self.home_points - self.away_points + self.home_factor_points
        difference = max(min(raw_difference, 10), -10)
        home_row = pandas.Series(index=self.columns)
        home_row.draw = self.intercept - self.slope * difference
        home_row.narrow_win = home_row.draw + 1
        home_row.narrow_loss = home_row.draw - 1
        home_row.large_win = home_row.narrow_win * 1.5
        home_row.large_loss = home_row.narrow_loss * 1.5
        if self.world_cup:
            home_row = numpy.array(home_row) * 2
        reversed_home_row = home_row.iloc[::-1]
        away_row = numpy.array(reversed_home_row) * -1
        self.df.loc[self.home_name, :] = home_row
        self.df.loc[self.away_name, :] = away_row
        print(f"points at stake:\n{self.df}")
        self.df.loc[self.home_name, :] = numpy.array(home_row) + self.home_points
        self.df.loc[self.away_name, :] = numpy.array(away_row) + self.away_points
        print(f"\nfinal ranking points:\n{self.df}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("home_name", type=str)
    parser.add_argument("away_name", type=str)
    parser.add_argument("home_points", type=float)
    parser.add_argument("away_points", type=float)
    parser.add_argument("--no_home_factor", action="store_true")
    parser.add_argument("--world_cup", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    ranking_points = RankingPoints(**vars(arguments))
    ranking_points.calculate_and_display()
