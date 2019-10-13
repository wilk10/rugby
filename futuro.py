import os
import pandas
import pathlib
import datetime
import numpy as np


class Futuro:
    ORDER_NUM_BY_POSITION = {'PR': 1, 'HK': 2, 'SR': 4, 'F8': 6, 'SH': 9, 'FH': 10, 'CE': 12, 'WI / FB': 14}
    SEASONS = ['19/20', '20/21', '21/22', '22/23', '23/24']
    NUM_PLAYERS_BY_POSITION = {'PR': 4, 'HK': 2, 'SR': 3, 'F8': 4, 'SH': 2, 'FH': 2, 'CE': 3, 'WI / FB': 3}
    CAPS_BY_NUM_PLAYERS = {2: [9, 8, 2, 1], 3: [9, 8, 7, 2, 1], 4: [10, 9, 8, 7, 2, 1]}

    def __init__(self):
        self.data_dir = pathlib.Path.cwd() / 'data_futuro'
        self.df = self.load_df()

    def load_df(self):
        files = os.listdir(str(self.data_dir))
        date_strs = [file.split('.')[0].split('_')[-1] for file in files if '.DS_Store' not in file]
        dates = [datetime.datetime.strptime(date_str, '%Y-%m-%d') for date_str in date_strs]
        most_recent_date = sorted(dates)[-1]
        most_recent_date_str = most_recent_date.strftime('%Y-%m-%d')
        most_recent_file = f'priority_{most_recent_date_str}.csv'
        filepath = self.data_dir / most_recent_file
        df = pandas.read_csv(filepath, header=0)
        df.fillna(0, inplace=True)
        return df

    @staticmethod
    def find_new_rank(j, positions_to_drop, season_df, season, position):
        unassigned_df = season_df.loc[season_df[f'player_{season}'].isnull()]
        if positions_to_drop == 0:
            new_rank = min(unassigned_df.index.values)
        else:
            new_rank = j + positions_to_drop
            while True:
                if new_rank >= season_df.index.values[-1]:
                    new_rank = max(unassigned_df.index.values)
                    break
                else:
                    target_spot = season_df.loc[new_rank, :]
                    if pandas.isnull(target_spot[f'player_{season}']):
                        break
                    else:
                        new_rank += 1
        return new_rank

    def foresee(self):
        future_columns = [f'{col}_{season}' for season in self.SEASONS for col in ['player', 'age', 'caps']]
        sorted_positions = sorted(self.ORDER_NUM_BY_POSITION.items(), key=lambda kv: kv[1])
        for position, _ in sorted_positions:
            position_df = self.df.loc[self.df['ruolo'] == position]
            future_df = pandas.DataFrame(columns=future_columns)
            future_df['player_19/20'] = position_df['giocatore']
            future_df['age_19/20'] = position_df['eta']
            default_caps = self.CAPS_BY_NUM_PLAYERS[self.NUM_PLAYERS_BY_POSITION[position]]
            all_default_caps = default_caps + [0] * (len(future_df) - len(default_caps))
            future_df['caps_19/20'] = position_df['presenze nazionale 19-20']
            future_df['caps_19/20'] = np.maximum(future_df['caps_19/20'], all_default_caps)
            future_df['caps_19/20'].fillna(0, inplace=True)
            for i, season in enumerate(self.SEASONS):
                if i > 0:
                    past_season = self.SEASONS[i-1]
                    season_columns = [column for column in future_columns if season in column]
                    past_season_columns = [column for column in future_columns if past_season in column]
                    season_df = future_df[season_columns].copy()
                    past_season_df = future_df[past_season_columns]
                    season_df[f'caps_{season}'] = all_default_caps
                    for j, row in past_season_df.iterrows():
                        old_age = 30 if position in ['PR', 'HK'] else 28
                        positions_to_drop = max(0, row[f'age_{past_season}'] - old_age + 1)
                        new_rank = self.find_new_rank(j, positions_to_drop, season_df, season, position)
                        season_df.loc[new_rank, f'player_{season}'] = row[f'player_{past_season}']
                        season_df.loc[new_rank, f'age_{season}'] = row[f'age_{past_season}'] + 1
                    future_df.loc[:, season_columns] = season_df
            print(f'{future_df.to_string()}\n')


if __name__ == '__main__':
    Futuro().foresee()
