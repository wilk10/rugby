import pandas
import pathlib
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance


class Clusterer:
    def __init__(self):
        self.root = pathlib.Path.cwd()
        self.filepath = self.root / "serie_a.csv"
        self.data = self.load_data()
        self.groups = ["A", "B", "C"]
        self.kmeans = KMeans(
            n_clusters=len(self.groups), random_state=0).fit(self.data)
        self.data["label"] = self.kmeans.labels_
        self.cluster_centers = self.kmeans.cluster_centers_

    def load_data(self):
        loaded_data = pandas.read_csv(self.filepath)
        columns = ["X", "Y"]
        data = loaded_data[columns]
        data.index = loaded_data.squadra
        return data

    def add_distance_to_centre(self):
        distances = []
        for squadra in self.data.index.values:
            team_location = self.data.loc[squadra, ["X", "Y"]]
            label = self.data.loc[squadra, "label"]
            cluster_center = self.cluster_centers[label]
            team_distance = distance.euclidean(team_location, cluster_center)
            distances.append(team_distance)
        return distances

    def adjust_data(self):
        while True:
            self.data["distance_to_centre"] = self.add_distance_to_centre()
            size_by_group = dict.fromkeys(self.groups)
            for group_id, group in enumerate(size_by_group):
                ids = [label for label in self.data.label if label == group_id]
                size_by_group[group] = len(ids)
            sizes = [size for group, size in size_by_group.items()]
            if not min(sizes) == len(self.data) / len(self.groups):
                first_smallest_group = [
                    group for group, size in size_by_group.items()
                    if size == min(sizes)][0]

                import pdb
                pdb.set_trace()
            else:
                break

    def print_results(self, data):
        for group_id, group in enumerate(self.groups):
            print(f"\nGroup {group}")
            for team_id, team in enumerate(data.index):
                if self.data.label.iloc[team_id] == group_id:
                    print(team)

    def run(self):
        adjusted_data = self.adjust_data()
        self.print_results(adjusted_data)
        import pdb
        pdb.set_trace()


if __name__ == "__main__":
    clusterer = Clusterer()
    clusterer.run()
