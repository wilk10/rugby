import pandas
import pathlib
import numpy as np
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
        self.data["final_group"] = np.nan

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

    def print_results(self):
        for group_id, group in enumerate(self.groups):
            print(f"\nGroup {group}")
            group_df = self.data.loc[self.data["final_group"] == group]
            for team in group_df.index.values:
                print(team)

    def run(self):
        for i in range(self.TARGET_GROUP_SIZE):
            for label, group in enumerate(self.groups):
                unassigned_mask = self.data["final_group"].isnull()
                df = self.data[unassigned_mask]
                if i == 0:
                    centroid_df = df.loc[df["initial_label"] == label]
                else:
                    assigned_df = self.data[-unassigned_mask]
                    centroid_df = assigned_df.loc[assigned_df["final_group"] == group]
                centroid = centroid_df.loc[:, ["X", "Y"]].mean(axis=0)
                distances = self.get_distance_to_centroid(df, centroid)
                closest_team = self.get_closest_team(df, distances)
                self.data.loc[self.data.index == closest_team, "final_group"] = group
        self.print_results()


if __name__ == "__main__":
    clusterer = Clusterer()
    clusterer.run()
