import requests
import pandas as pd

url = "http://iszz.azo.hr/iskzl/rest/ispitivanjapodataka/getAll"
params = {"p_godina": 2017, "p_grad": "Osijek"}
data = requests.get(url, params=params).json()["items"]

df = pd.DataFrame(data)
df["Datum"] = pd.to_datetime(df["Datum"])

pm10 = df[(df["Parameter"] == "PM10") & (df["Grad"] == "Osijek")]

print("PM10 mjerenja Osijek 2017:")
print(pm10)

print("\nTop 3 datuma s najvećom koncentracijom PM10:")
print(pm10.nlargest(3, "Vrijednost")[["Datum", "Vrijednost"]])
