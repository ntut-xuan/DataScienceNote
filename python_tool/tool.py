import numpy as np
import copy


class MultinomialNaiveBayesClassifier:
    def __init__(self):
        self.value_map = {}
        self.result = np.array([])
        self.data = np.array([])

    def fit(self, x, y):
        self.result = y
        unique, counts = np.unique(self.result, return_counts=True)
        for v in unique:
            self.value_map[v] = []
        for i in range(len(y)):
            for s in x[i]:
                self.value_map[y[i]].append(s)

    def add_duplicated_value(self, temp_result, temp_value_map, missing_value):
        unique, counts = np.unique(temp_result, return_counts=True)
        for k in unique:
            value_unique = np.unique(temp_value_map[k])
            for v in value_unique:
                if missing_value == v:
                    continue
                temp_value_map[k].append(v)
        for k in unique:
            temp_value_map[k].append(missing_value)

    def print_unique(self, y):
        for v in np.unique(self.result):
            unique, counts = np.unique(y[v], return_counts=True)
            result_dict = dict(zip(unique, counts))
            print(result_dict)

    def predict(self, y):
        temp_result = self.result
        temp_value_map = self.value_map
        unique, counts = np.unique(self.result, return_counts=True)
        result_dict = dict(zip(unique, counts))
        predict = {}
        for v in unique:
            print("calculate P(%s) = %d/%d" % (v, result_dict[v], int(np.sum(counts))))
            predict[v] = float(result_dict[v] / np.sum(counts))
        for s in y:
            for v in unique:
                if s not in self.value_map[v]:
                    self.add_duplicated_value(temp_result, temp_value_map, s)
        self.print_unique(temp_value_map)
        for s in y:
            for v in unique:
                unique_s, counts_s = np.unique(self.value_map[v], return_counts=True)
                result_dict_s = dict(zip(unique_s, counts_s))
                total = np.sum(counts_s)
                print("calculate P(%s|%s) = %d/%d" % (s, v, result_dict_s[s], int(total)))
                predict[v] *= float(result_dict_s[s] / int(total))
        print(predict)
        return max(predict, key=predict.get)


class UPGMAClustering:

    def __init__(self, n_clusters=2):
        self.labels_ = None
        self.n = None
        self.n_clusters = n_clusters
        self.cluster_list = None
        self.distance_table = None
        self.threshold = None

    def fit(self, distance_table):
        self.n = len(distance_table)
        self.distance_table = distance_table
        self.cluster_list = []
        self.threshold = []
        self.labels_ = [0 for _ in range(self.n)]
        for i in range(self.n):
            self.cluster_list.append([i])

    def clustering(self):
        for r in range(self.n - (self.n_clusters-1)):

            print("\n=============== Iter%5d ===============\n" % (r + 1))

            original_cluster_list = copy.deepcopy(self.cluster_list)
            self.distance_table[self.distance_table == 0] = np.iinfo(np.int32).max

            # choose minimum pair
            minimum_distance = np.min(self.distance_table)
            minimum_pair = np.unravel_index(np.argmin(self.distance_table, axis=None), self.distance_table.shape)
            print("Minimum pair: %s = %d" % (str(minimum_pair), minimum_distance))

            if r != self.n - self.n_clusters:

                # combine two cluster
                target_list1 = copy.deepcopy(self.cluster_list[minimum_pair[0]])
                target_list2 = copy.deepcopy(self.cluster_list[minimum_pair[1]])
                self.cluster_list[minimum_pair[0]] += (self.cluster_list[minimum_pair[1]])
                self.cluster_list.remove(self.cluster_list[minimum_pair[1]])

                print("target list l1=", target_list1, "- l2=", target_list2)
                print("combine list", original_cluster_list, "->", self.cluster_list)

                # recalculate distance between new cluster and another cluster
                self.distance_table[self.distance_table == np.iinfo(np.int32).max] = 0
                new_distance_table = np.zeros((len(self.cluster_list), len(self.cluster_list)))
                for i in range(len(self.cluster_list)):
                    for j in range(len(self.cluster_list)):
                        index1 = original_cluster_list.index(target_list1)
                        index2 = original_cluster_list.index(target_list2)
                        index4 = self.cluster_list.index(self.cluster_list[j])
                        if i == index1 or j == index1:
                            if index1 == index4:
                                continue
                            index3 = original_cluster_list.index(self.cluster_list[j])
                            distance1 = self.distance_table[index1][index3]
                            distance2 = self.distance_table[index2][index3]
                            new_distance_table[index1][index4] = (len(target_list1) * distance1 + len(
                                target_list2) * distance2) / float(len(target_list1) + len(target_list2))
                            new_distance_table[index4][index1] = (len(target_list1) * distance1 + len(
                                target_list2) * distance2) / float(len(target_list1) + len(target_list2))
                        else:
                            index3 = original_cluster_list.index(self.cluster_list[j])
                            index5 = original_cluster_list.index(self.cluster_list[i])
                            new_distance_table[i][j] = self.distance_table[index5][index3]

                self.distance_table = new_distance_table

                # print
                print("Update Table\n", str(new_distance_table))
                print(self.cluster_list)

            # calculate branch length estimation
            branch_length_estimation = minimum_distance / 2.0
            self.threshold += [branch_length_estimation]
            print("distance_threshold =", branch_length_estimation)

        for i in range(len(self.cluster_list)):
            for v in self.cluster_list[i]:
                self.labels_[v] = i+1


class WPGMAClustering:

    def __init__(self, n_clusters=2):
        self.labels_ = None
        self.n = None
        self.n_clusters = n_clusters
        self.cluster_list = None
        self.distance_table = None
        self.threshold = None

    def fit(self, distance_table):
        self.n = len(distance_table)
        self.distance_table = distance_table
        self.cluster_list = []
        self.threshold = []
        self.labels_ = [0 for _ in range(self.n)]
        for i in range(self.n):
            self.cluster_list.append([i])

    def clustering(self):
        for r in range(self.n - (self.n_clusters-1)):

            print("\n=============== Iter%5d ===============\n" % (r + 1))

            original_cluster_list = copy.deepcopy(self.cluster_list)
            self.distance_table[self.distance_table == 0] = np.iinfo(np.int32).max

            # choose minimum pair
            minimum_distance = np.min(self.distance_table)
            minimum_pair = np.unravel_index(np.argmin(self.distance_table, axis=None), self.distance_table.shape)
            print("Minimum pair: %s = %d" % (str(minimum_pair), minimum_distance))

            if r != self.n - self.n_clusters:

                # combine two cluster
                target_list1 = copy.deepcopy(self.cluster_list[minimum_pair[0]])
                target_list2 = copy.deepcopy(self.cluster_list[minimum_pair[1]])
                self.cluster_list[minimum_pair[0]] += (self.cluster_list[minimum_pair[1]])
                self.cluster_list.remove(self.cluster_list[minimum_pair[1]])

                print("target list l1=", target_list1, "- l2=", target_list2)
                print("combine list", original_cluster_list, "->", self.cluster_list)

                # recalculate distance between new cluster and another cluster
                self.distance_table[self.distance_table == np.iinfo(np.int32).max] = 0
                new_distance_table = np.zeros((len(self.cluster_list), len(self.cluster_list)))
                for i in range(len(self.cluster_list)):
                    for j in range(len(self.cluster_list)):
                        index1 = original_cluster_list.index(target_list1)
                        index2 = original_cluster_list.index(target_list2)
                        index4 = self.cluster_list.index(self.cluster_list[j])
                        if i == index1 or j == index1:
                            if index1 == index4:
                                continue
                            index3 = original_cluster_list.index(self.cluster_list[j])
                            distance1 = self.distance_table[index1][index3]
                            distance2 = self.distance_table[index2][index3]
                            new_distance_table[index1][index4] = (distance1 + distance2) / 2.0
                            new_distance_table[index4][index1] = (distance1 + distance2) / 2.0
                        else:
                            index3 = original_cluster_list.index(self.cluster_list[j])
                            index5 = original_cluster_list.index(self.cluster_list[i])
                            new_distance_table[i][j] = self.distance_table[index5][index3]

                self.distance_table = new_distance_table

                # print
                print("Update Table\n", str(new_distance_table))
                print(self.cluster_list)

            # calculate branch length estimation
            branch_length_estimation = minimum_distance / 2.0
            self.threshold += [branch_length_estimation]
            print("distance_threshold =", branch_length_estimation)

        for i in range(len(self.cluster_list)):
            for v in self.cluster_list[i]:
                self.labels_[v] = i+1
