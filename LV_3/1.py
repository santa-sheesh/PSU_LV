import pandas as pd

mtcars = pd.read_csv("mtcars.csv")
mtcars["Weight_kg"] = mtcars["wt"] * 0.453592

print("1. 5 automobila s najvećom potrošnjom:")
print(mtcars.sort_values("mpg").head(5)[["car", "mpg"]])

print("\n2. 3 automobila s 8 cilindara i najmanjom potrošnjom:")
print(mtcars[mtcars["cyl"] == 8].sort_values("mpg").head(3)[["car", "mpg"]])

print("\n3. Prosječna potrošnja automobila sa 6 cilindara:")
print(mtcars[mtcars["cyl"] == 6]["mpg"].mean())

print("\n4. Prosječna potrošnja 4-cilindarskih automobila mase 2000–2200 lbs:")
print(mtcars[(mtcars["cyl"] == 4) & (mtcars["wt"].between(2.0, 2.2))]["mpg"].mean())

print("\n5. Broj automobila po tipu mjenjača:")
print(mtcars["am"].value_counts())

print("\n6. Broj automatskih automobila sa snagom > 100 KS:")
print(mtcars[(mtcars["am"] == 0) & (mtcars["hp"] > 100)].shape[0])

print("\n7. Masa svakog automobila u kg:")
print(mtcars[["car", "Weight_kg"]])
