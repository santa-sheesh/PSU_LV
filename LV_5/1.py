import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("occupancy_processed.csv")

X = df[["S3_Temp", "S5_CO2"]]
y = df["Room_Occupancy_Count"]

plt.figure()
for value, label in [(0, "Slobodna"), (1, "Zauzeta")]:
    subset = X[y == value]
    plt.scatter(subset["S3_Temp"], subset["S5_CO2"], label=label)

plt.xlabel("S3_Temp")
plt.ylabel("S5_CO2")
plt.title("Zauzetost prostorije")
plt.legend()
plt.show()
