import pandas
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance


class Clusterer:
    TARGET_GROUP_SIZE = 10

    def __init__(self):
        self.root = pathlib.Path.cwd()
        self.filepath = self.root / "serie_a.csv"
        self.data = self.load_data()
        self.groups = ["A", "B", "C"]
        kmeans = KMeans(n_clusters=len(self.groups), random_state=0).fit(self.data)
        self.data["initial_label"] = kmeans.labels_
        self.data["proposed_group"] = np.nan

    def load_data(self):
        loaded_data = pandas.read_csv(self.filepath)
        columns = ["X", "Y"]
        data = loaded_data[columns]
        data.index = loaded_data.squadra
        return data

    @staticmethod
    def get_distance_to_centroid(df, centroid):
        distances = []
        for squadra in df.index.values:
            team_location = df.loc[df.index == squadra, ["X", "Y"]]
            team_distance = distance.euclidean(team_location, centroid)
            distances.append(team_distance)
        return distances

    @staticmethod
    def get_closest_team(df, distances):
        series = pandas.Series(distances, index=df.index)
        df = df.assign(distance=series.values)
        min_distance = df["distance"].min()
        closest_team_df = df.loc[df["distance"] == min_distance]
        assert len(closest_team_df) == 1
        return closest_team_df.index.values[0]

    def add_closest_group(self):
        closest_column = self.data.loc[:, ["distance_A", "distance_B", "distance_C"]].idxmin(axis=1)
        closest_group = np.array([closest[-1] for closest in closest_column.values])
        self.data = self.data.assign(closest_group=closest_group)

    def adjust_group_sizes(self, latest_moves):
        while True:
            size_by_group = {
                group: len(self.data.loc[self.data["proposed_group"] == group])
                for group in self.groups}
            min_size = min([size for size in size_by_group.values()])
            all_equal = all([size == min_size for size in size_by_group.values()])
            if not all_equal:
                smallest_group = [group for group, size in size_by_group.items() if size == min_size]
                assert len(smallest_group) == 1
                group_df = self.data.loc[self.data["proposed_group"] == smallest_group[0]]
                centroid = group_df.loc[:, ["X", "Y"]].mean(axis=0)
                other_groups_df = self.data[~self.data.isin(group_df)].dropna()
                keep_teams = [team for team in other_groups_df.index.values if team not in latest_moves]
                other_groups_df = other_groups_df.loc[keep_teams, :]
                distances = self.get_distance_to_centroid(other_groups_df, centroid)
                other_groups_df = other_groups_df.assign(distance=distances)
                closest_in_other_groups = other_groups_df["distance"].idxmin()
                print(f"moving {closest_in_other_groups} from "
                      f"{self.data.loc[closest_in_other_groups, 'proposed_group']} to {smallest_group[0]}")
                self.data.loc[closest_in_other_groups, "proposed_group"] = smallest_group[0]
                latest_moves.append(closest_in_other_groups)
            else:
                break

    def correct_groups(self):
        moved_teams = []
        while True:
            for group in self.groups:
                group_df = self.data.loc[self.data["proposed_group"] == group]
                centroid = group_df.loc[:, ["X", "Y"]].mean(axis=0)
                distances = self.get_distance_to_centroid(self.data, centroid)
                self.data[f"distance_{group}"] = distances
            self.add_closest_group()
            belong_to_correct_group = np.array([self.data.proposed_group == self.data.closest_group])
            if not belong_to_correct_group.all():
                distance_proposed_group = np.array([
                    row[f"distance_{row.proposed_group}"] for i, row in self.data.iterrows()])
                distance_closest_group = np.array([
                    row[f"distance_{row.closest_group}"] for i, row in self.data.iterrows()])
                metric = distance_proposed_group - distance_closest_group
                self.data = self.data.assign(metric=metric)
                furthest_team = self.data["metric"].idxmax()
                if furthest_team not in moved_teams:
                    old_group = self.data.loc[furthest_team, 'proposed_group']
                    new_group = self.data.loc[furthest_team, 'closest_group']
                    print(f"\nmoving {furthest_team} from {old_group} to {new_group} ---!")
                    self.data.loc[furthest_team, 'proposed_group'] = new_group
                    moved_teams.append(furthest_team)
                    self.adjust_group_sizes([furthest_team])
                else:
                    break
            else:
                break

    def print_results(self):
        for group_id, group in enumerate(self.groups):
            print(f"\n--- Group {group} ---")
            group_df = self.data.loc[self.data["proposed_group"] == group]
            for team in group_df.index.values:
                print(team)

    def plot_results(self):
        colour_by_group = {"A": "red", "B": "blue", "C": "green"}
        colours = [colour_by_group[group] for group in self.data["proposed_group"]]
        plt.scatter(self.data["Y"], self.data["X"], c=colours)
        plt.axis(aspect='equal')
        plt.show()

    def run(self):
        for i in range(self.TARGET_GROUP_SIZE):
            for label, group in enumerate(self.groups):
                unassigned_mask = self.data["proposed_group"].isnull()
                df = self.data[unassigned_mask]
                if i == 0:
                    centroid_df = df.loc[df["initial_label"] == label]
                else:
                    assigned_df = self.data[-unassigned_mask]
                    centroid_df = assigned_df.loc[assigned_df["proposed_group"] == group]
                centroid = centroid_df.loc[:, ["X", "Y"]].mean(axis=0)
                distances = self.get_distance_to_centroid(df, centroid)
                closest_team = self.get_closest_team(df, distances)
                self.data.loc[self.data.index == closest_team, "proposed_group"] = group
        self.correct_groups()
        self.plot_results()
        self.print_results()


if __name__ == "__main__":
    clusterer = Clusterer()
    clusterer.run()
