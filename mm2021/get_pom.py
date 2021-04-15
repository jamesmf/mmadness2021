# !pip install bs4 pandas requests lxml
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import requests

base_urls = [
    "https://web.archive.org/web/20110311233233/http://www.kenpom.com/",
    "https://web.archive.org/web/20120311165019/http://kenpom.com/",
    "https://web.archive.org/web/20130318221134/http://kenpom.com/",
    "https://web.archive.org/web/20140318100454/http://kenpom.com/",
    "https://web.archive.org/web/20150316212936/http://kenpom.com/",
    "https://web.archive.org/web/20160314134726/http://kenpom.com/",
    "https://web.archive.org/web/20170312131016/http://kenpom.com/",
    "https://web.archive.org/web/20180311122559/https://kenpom.com/",
    "https://web.archive.org/web/20190317211809/https://kenpom.com/",
    "https://kenpom.com/index.php",
]

base = "../mens"


def scrape_archive(url, year):
    """
    Imports raw data from a kenpom archive into a dataframe
    """

    page = requests.get(url)
    soup = BeautifulSoup(page.text)
    table_full = soup.find_all("table", {"id": "ratings-table"})

    thead = table_full[0].find_all("thead")
    table = table_full[0]

    for weird in thead:
        table = str(table).replace(str(weird), "")

    df = pd.read_html(table)[0]
    if year == 2020:
        df["year"] = 2021
    else:
        df["year"] = year

    return df


def scraping(df, year):

    for url in base_urls:

        print(f"Scrapping: {url}")
        archive = scrape_archive(url, year)

        df = pd.concat((df, archive), axis=0)
        year += 1

    df.columns = [
        "Rank",
        "Team",
        "Conference",
        "W-L",
        "Pyth",
        "AdjustO",
        "AdjustO Rank",
        "AdjustD",
        "AdjustD Rank",
        "AdjustT",
        "AdjustT Rank",
        "Luck",
        "Luck Rank",
        "SOS Pyth",
        "SOS Pyth Rank",
        "SOS OppO",
        "SOS OppO Rank",
        "SOS OppD",
        "SOS OppD Rank",
        "NCSOS Pyth",
        "NCSOS Pyth Rank",
        "Year",
    ]

    df = df[["Year", "Team", "AdjustO", "AdjustD", "Luck", "Rank", "Pyth"]]
    df.columns = ["Season", "TeamName", "adj_o", "adj_d", "luck", "rank", "pyth"]

    df.TeamName = df.TeamName.apply(lambda x: re.sub("\d", "", x).strip()).replace(
        ".", ""
    )

    return df


df = None
year = 2011
df = scraping(df, year)
df.head()

df.TeamName = df.TeamName.apply(lambda x: x.replace("-", " "))
df.TeamName = df.TeamName.apply(lambda x: x.lower())
df.TeamName = df.TeamName.apply(lambda x: x.strip())
df.TeamName = df.TeamName.replace("mississippi valley st.", "mississippi valley state")
# df.TeamName=df.TeamName.replace('texas a&m corpus chris','texas a&m corpus christi')
df.TeamName = df.TeamName.replace("dixie st.", "dixie st")
df.TeamName = df.TeamName.replace("st. francis pa", "st francis pa")
df.TeamName = df.TeamName.replace("ut rio grande valley", "texas rio grande valley")
df.TeamName = df.TeamName.replace("southeast missouri st.", "southeast missouri state")
df.TeamName = df.TeamName.replace("tarleton st.", "tarleton st")
df.TeamName = df.TeamName.replace("liu", "liu brooklyn")
df.TeamName = df.TeamName.replace("cal st. bakersfield", "cal state bakersfield")

df.TeamName = df.TeamName.replace("virginia military inst", "virginia military	")
df.TeamName = df.TeamName.replace("louisiana saint", "louisiana state")
df.TeamName = df.TeamName.replace("nj inst of technology", "njit")

df.TeamName = df.TeamName.replace("texas a&m corpus chris", "texas a&m corpus")
df.TeamName = df.TeamName.replace("md baltimore county", "maryland baltimore county")
# -------------------------------------------------------
# merge with spelling file to get the TeamID
spelling = pd.read_csv(
    f"{base}/MDataFiles_Stage2/MTeamSpellings.csv",
    encoding="cp1252",
)
spelling.columns = ["TeamName", "TeamID"]
spelling.TeamName = spelling.TeamName.apply(lambda x: x.replace("-", " "))
df.TeamName = df.TeamName.apply(lambda x: x.strip())


df = df.merge(spelling[["TeamName", "TeamID"]], on="TeamName", how="left")

df.TeamName = df.TeamName.apply(lambda x: x.replace("st.", "saint"))
df.TeamName = df.TeamName.apply(lambda x: x.replace(";", ""))
df.TeamName = df.TeamName.apply(lambda x: x.replace("\t", ""))
df.TeamName = df.TeamName.replace("texas a&m corpus chris", "texas a&m corpus")
df.TeamName = df.TeamName.replace("louisiana saint", "louisiana state")

df = df.merge(spelling[["TeamName", "TeamID"]], on="TeamName", how="left")

df.TeamID_x.fillna(df.TeamID_y, inplace=True)

df = df[["Season", "TeamID_x", "adj_o", "adj_d", "luck", "rank", "pyth"]]
df.columns = ["Season", "TeamID", "adj_o", "adj_d", "luck", "rank", "pyth"]
df.TeamID = df.TeamID.astype("int64")

df.columns = ["Season", "TeamID", "adj_o", "adj_d", "luck", "rank", "adj_em"]
