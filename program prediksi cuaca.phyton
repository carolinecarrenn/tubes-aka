import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import time

# Fungsi untuk mengukur waktu eksekusi
def measure_time(algorithm, X, y):
    start_time = time.time()
    algorithm.fit(X, y)
    return time.time() - start_time

# Meminta input angka n dari pengguna
print("Simulasi algoritma rekursif dan iteratif.")
n_values = input("Masukkan ukuran dataset : ")
n_values = [int(n.strip()) for n in n_values.split(",")]

# Menyimpan hasil pengukuran waktu
results = []

# Mengukur waktu eksekusi untuk setiap ukuran dataset
for n in n_values:
    # Membuat dataset sintetis dengan n sampel dan 20 fitur
    X, y = make_classification(n_samples=n, n_features=20, random_state=42)

    # Decision Tree (Iteratif)
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_time = measure_time(dt_model, X, y)

    # Random Forest (Rekursif)
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    rf_time = measure_time(rf_model, X, y)

    # Menyimpan hasil pengukuran waktu
    results.append([n, rf_time, dt_time])

# Konversi ke DataFrame
df_results = pd.DataFrame(results, columns=["n", "Recursive Time (s)", "Iterative Time (s)"])

# Menampilkan DataFrame hasil
print("\nHasil simulasi:")

from tabulate import tabulate
print(tabulate(df_results, headers='keys', tablefmt='psql', showindex=False))

# Kesimpulan per algoritma
print("\nKesimpulan per algoritma:")
print("- Random Forest (Recursive): Secara umum, waktu eksekusi bertambah seiring dengan peningkatan ukuran dataset karena proses rekursif membangun banyak pohon keputusan.")
print("- Decision Tree (Iterative): Waktu eksekusi lebih cepat dibanding Random Forest pada dataset kecil, tetapi memiliki batasan skalabilitas.")

# Kesimpulan gabungan
print("\nKesimpulan gabungan:")
print("Random Forest membutuhkan waktu lebih lama dibanding Decision Tree karena melibatkan proses rekursif yang kompleks. "
      "Namun, Random Forest sering kali memberikan hasil prediksi yang lebih akurat karena prinsip ensemble learning.")

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(df_results["n"], df_results["Recursive Time (s)"], label="Random Forest (Recursive)", marker='o', color='blue')
plt.plot(df_results["n"], df_results["Iterative Time (s)"], label="Decision Tree (Iterative)", marker='o', color='pink')

# Menambahkan label dan judul
plt.title("Perbandingan Waktu Eksekusi: Random Forest vs Decision Tree")
plt.xlabel("Input (n)")
plt.ylabel("Execution Time (seconds)")
plt.legend()
plt.grid(True)

# Menampilkan grafik
plt.show()
