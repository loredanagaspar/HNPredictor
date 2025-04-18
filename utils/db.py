# utils/db.py
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import urlparse
from datetime import datetime
from urllib.parse import urlparse


# Connection string format: postgresql://user:pass@host:port/dbname
CONN_STRING = "postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"

engine = create_engine(CONN_STRING)

def fetch_data():
    query = """
    SELECT title, url, time, score
    FROM hacker_news.items
    WHERE title IS NOT NULL AND score IS NOT NULL AND time IS NOT NULL
    """
    df = pd.read_sql(query, engine)

    def safe_extract_domain(u):
        try:
             return urlparse(u).netloc
        except:
             return ""

    df["domain"] = df["url"].apply(safe_extract_domain)
    df["title_length"] = df["title"].apply(lambda t: len(t.split()))
    df["hour"] = df["time"].dt.hour
    df["weekday"] = df["time"].dt.weekday

    return df.dropna()
