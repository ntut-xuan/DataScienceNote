import numpy as np


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
