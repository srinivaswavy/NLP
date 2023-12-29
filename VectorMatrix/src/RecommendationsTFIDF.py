import datetime

import pandas as pd

data = pd.read_csv("tmdb_5000_movies.csv")


def budget(row):
    if (row[0] >= 0 and row[0] < 8400000):
        return "small"
    elif (row[0] >= 8400000 and row[0] < 58400000):
        return "medium"
    else:
        return "big"


def popularity(row):
    if (row[8] >= 0 and row[8] < 50):
        return "less"
    elif (row[8] >= 50 and row[8] < 100):
        return "medium"
    else:
        return "high"


def genres(row):
    return ",".join([str(g).replace(" ", "") for g in pd.read_json(row[1])['name'].values])


def keywords(row):
    return ",".join([str(g).replace(" ", "") for g in pd.read_json(row[4])['name'].values])


def production_companies(row):
    return ",".join([str(g).replace(" ", "") for g in pd.read_json(row[9])['name'].values])


def production_countries(row):
    return ",".join([str(g).replace(" ", "") for g in pd.read_json(row[10])['name'].values])


newdata = [(r[6] +
            " with " + budget(r) +
            " budget and belongs to genres " + genres(r) +
            " keywords are " + keywords(r) +
            " original language " + r[5] +
            " with overview of " + r[7] +
            " with " + popularity(r) + " popularity. " +
            " production companies are " + production_companies(r) + ". " +
            " production countries are " + production_countries(r) + ". ")
           for r in data.values
           ]

print(newdata[0])
