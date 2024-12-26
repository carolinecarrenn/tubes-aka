// Menggunakan Desecion Tree

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
import time
import matplotlib.pyplot as plt

# Fungsi untuk mengukur waktu eksekusi
def measure_time(algorithm, X, y):
    start_time = time.time()
    algorithm.fit(X, y)
    return time.time() - start_time

# Decision Tree (Iteratif)
class IterativeDecisionTree:
    def __init__(self, max_depth=None):
        self.model = DecisionTreeClassifier(max_depth=max_depth)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

# Decision Tree (Rekursif)
class RecursiveDecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        # Kondisi berhenti untuk rekursi
        if len(set(y)) == 1 or (self.max_depth and depth >= self.max_depth):
            return {'label': np.bincount(y).argmax()}

        # Mencari fitur terbaik untuk pemisahan
        best_feature, best_value = self._best_split(X, y)

        # Membagi dataset
        left_mask = X[:, best_feature] <= best_value
        right_mask = ~left_mask
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {'feature': best_feature, 'value': best_value, 'left': left_tree, 'right': right_tree}

    def _best_split(self, X, y):
        best_feature = None
        best_value = None
        best_gini = float('inf')

        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            for value in values:
                left_mask = X[:, feature] <= value
                right_mask = ~left_mask
                left_y = y[left_mask]
                right_y = y[right_mask]
                gini = self._gini_index(left_y, right_y)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_value = value

        return best_feature, best_value

    def _gini_index(self, left_y, right_y):
        def gini(y):
            proportions = np.bincount(y) / len(y)
            return 1 - np.sum(proportions ** 2)

        return (len(left_y) / (len(left_y) + len(right_y))) * gini(left_y) + \
               (len(right_y) / (len(left_y) + len(right_y))) * gini(right_y)

    def predict(self, X):
        return [self._predict_single(x, self.tree) for x in X]

    def _predict_single(self, x, tree):
        if 'label' in tree:
            return tree['label']
        if x[tree['feature']] <= tree['value']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])

# Menyimulasikan dan mengukur waktu eksekusi untuk banyak ukuran dataset
print("Simulasi Decision Tree - Iteratif vs Rekursif")
n_values = input("Masukkan ukuran dataset : ")
n_values = [int(n.strip()) for n in n_values.split(",")]

# Menyimpan hasil pengukuran waktu
results = []

# Mengukur waktu eksekusi untuk setiap ukuran dataset
for n in n_values:
    # Membuat dataset sintetis
    X, y = make_classification(n_samples=n, n_features=20, random_state=42)

    # Iterative Decision Tree
    iterative_dt = IterativeDecisionTree(max_depth=5)
    iterative_time = measure_time(iterative_dt, X, y)

    # Recursive Decision Tree
    recursive_dt = RecursiveDecisionTree(max_depth=5)
    recursive_time = measure_time(recursive_dt, X, y)

    # Menyimpan hasil pengukuran waktu
    results.append([n, iterative_time, recursive_time])

# Konversi ke DataFrame
df_results = pd.DataFrame(results, columns=["n", "Iterative Time (s)", "Recursive Time (s)"])

# Menampilkan DataFrame hasil
print("\nHasil simulasi:")
from tabulate import tabulate
print(tabulate(df_results, headers='keys', tablefmt='psql', showindex=False))

# Membuat grafik perbandingan waktu eksekusi
plt.figure(figsize=(8, 6))
plt.plot(df_results["n"], df_results["Iterative Time (s)"], label="Decision Tree (Iteratif)", marker='o', color='blue')
plt.plot(df_results["n"], df_results["Recursive Time (s)"], label="Decision Tree (Rekursif)", marker='o', color='red')

# Menambahkan label dan judul
plt.title("Perbandingan Waktu Eksekusi: Decision Tree Iteratif vs Rekursif")
plt.xlabel("Ukuran Dataset (n)")
plt.ylabel("Waktu Eksekusi (detik)")
plt.legend()
plt.grid(True)

# Menampilkan grafik
plt.show()
