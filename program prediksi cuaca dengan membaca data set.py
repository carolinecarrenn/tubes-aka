import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from google.colab import files

# Unggah file CSV
print("Silakan unggah file CSV:")
uploaded = files.upload()

# Membaca file CSV
file_name = list(uploaded.keys())[0]
data = pd.read_csv(file_name)
print(f"\nDataset berhasil dibaca dari file: {file_name}\n")
print(data.head())

# Memastikan kolom yang diperlukan ada di dataset
required_columns = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Precip Type']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Kolom '{col}' tidak ditemukan di dataset. Pastikan dataset memiliki kolom: {required_columns}")

# Membuat kolom 'will_rain' berdasarkan kolom 'Precip Type'
data['will_rain'] = data['Precip Type'].apply(lambda x: 1 if 'rain' in str(x).lower() else 0)

# Memisahkan fitur dan target
X = data[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)']].values  # Fitur yang digunakan
y = data['will_rain'].values  # Target 'will_rain'

# Fungsi untuk mengukur waktu eksekusi
def measure_time(algorithm, X, y):
    start_time = time.time()
    algorithm.fit(X, y)
    return time.time() - start_time

# Decision Tree Iteratif (menggunakan scikit-learn)
class IterativeDecisionTree:
    def __init__(self, max_depth=None):
        self.model = DecisionTreeClassifier(max_depth=max_depth)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

# Decision Tree Rekursif (implementasi manual)
class RecursiveDecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if len(set(y)) == 1 or (self.max_depth and depth >= self.max_depth):
            return {'label': np.bincount(y).argmax()}

        best_feature, best_value = self._best_split(X, y)
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
                gini = self._gini_index(y[left_mask], y[right_mask])
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_value = value
        return best_feature, best_value

    def _gini_index(self, left_y, right_y):
        def gini(y):
            proportions = np.bincount(y) / len(y)
            return 1 - np.sum(proportions ** 2)
        total = len(left_y) + len(right_y)
        return (len(left_y) / total) * gini(left_y) + (len(right_y) / total) * gini(right_y)

    def predict(self, X):
        return [self._predict_single(x, self.tree) for x in X]

    def _predict_single(self, x, tree):
        if 'label' in tree:
            return tree['label']
        if x[tree['feature']] <= tree['value']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])

# Input ukuran dataset dari pengguna
n_values = input("Masukkan ukuran dataset : ")
n_values = [int(n.strip()) for n in n_values.split(",")]

# Simulasi dan analisis running time
results = []

for n in n_values:
    # Mengambil subset data
    subset_X = X[:n]
    subset_y = y[:n]

    # Decision Tree Iteratif
    iterative_dt = IterativeDecisionTree(max_depth=5)
    iterative_time = measure_time(iterative_dt, subset_X, subset_y)
    y_pred_iterative = iterative_dt.predict(subset_X)
    accuracy_iterative = accuracy_score(subset_y, y_pred_iterative)

    # Decision Tree Rekursif
    recursive_dt = RecursiveDecisionTree(max_depth=5)
    recursive_time = measure_time(recursive_dt, subset_X, subset_y)
    y_pred_recursive = recursive_dt.predict(subset_X)
    accuracy_recursive = accuracy_score(subset_y, y_pred_recursive)

    # Simpan hasil
    results.append([n, iterative_time, accuracy_iterative, recursive_time, accuracy_recursive])

# Konversi hasil ke DataFrame
df_results = pd.DataFrame(results, columns=["Dataset Size", "Iterative Time (s)", "Iterative Accuracy",
                                            "Recursive Time (s)", "Recursive Accuracy"])

# Tampilkan hasil
from tabulate import tabulate
print("\nHasil simulasi:")
print(tabulate(df_results, headers="keys", tablefmt="psql", showindex=False))

# Plot grafik waktu eksekusi
plt.figure(figsize=(10, 6))
plt.plot(df_results["Dataset Size"], df_results["Iterative Time (s)"], label="Iterative Time", marker="o", color="blue")
plt.plot(df_results["Dataset Size"], df_results["Recursive Time (s)"], label="Recursive Time", marker="o", color="red")
plt.title("Running Time Comparison: Iterative vs Recursive Decision Tree")
plt.xlabel("Dataset Size")
plt.ylabel("Execution Time (s)")
plt.legend()
plt.grid(True)
plt.show()
