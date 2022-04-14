import pandas as pd
















if __name__ == '__main__':
    data = pd.read_csv("steam.csv")
    owners = data[data["owners"]=="10000000-20000000"]
    print(owners.head())
