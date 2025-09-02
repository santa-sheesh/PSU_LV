import pandas as pd
import matplotlib.pyplot as plt

mtcars = pd.read_csv("mtcars.csv")

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
mtcars.groupby("cyl")["mpg"].mean().plot(kind="bar", alpha=0.7)
plt.title("Potrošnja prema broju cilindara")
plt.ylabel("mpg")

plt.subplot(2, 2, 2)
mtcars.boxplot(column="wt", by="cyl")
plt.title("Težina prema cilindrima")
plt.suptitle("")
plt.ylabel("Težina (lbs)")

plt.subplot(2, 2, 3)
mtcars.groupby("am")["mpg"].mean().plot(kind="bar", alpha=0.7)
plt.title("Potrošnja prema mjenjaču")
plt.xticks([0, 1], ["Automatski", "Ručni"])
plt.ylabel("mpg")

plt.subplot(2, 2, 4)
for am, label in [(0, "Automatski"), (1, "Ručni")]:
    subset = mtcars[mtcars["am"] == am]
    plt.scatter(subset["hp"], subset["qsec"], label=label, alpha=0.7)
plt.title("Ubrzanje i snaga prema mjenjaču")
plt.xlabel("hp")
plt.ylabel("qsec")
plt.legend()

plt.tight_layout()
plt.show()
