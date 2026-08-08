import os
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict
from urllib.parse import urlencode
from urllib.request import urlopen, Request
import ssl
import certifi
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import feedparser
import pandas as pd


DEFAULT_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5ENSEI&region=IN&lang=en-IN",
    "https://www.livemint.com/rss/market",
    "https://www.moneycontrol.com/rss/markets.xml",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/INbusinessNews",
]

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _parse_entry(entry) -> Dict:
    title = entry.get("title", "")
    link = entry.get("link", "")
    source = entry.get("source", {}).get("title", "") if isinstance(
        entry.get("source"), dict) else ""
    published = entry.get("published", entry.get("updated", ""))
    date = pd.to_datetime(published, errors="coerce")
    if pd.isna(date):
        date = pd.Timestamp.utcnow()
    return {
        "date": date.normalize(),
        "title": title,
        "headline": title,
        "link": link,
        "source": source,
    }


def _fetch_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(req, timeout=20, context=context) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_newsapi(days_back: int) -> List[Dict]:
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return []

    from_date = (pd.Timestamp.utcnow() -
                 pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "q": "NIFTY OR Sensex OR Indian stock market",
        "from": from_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": api_key,
    }
    url = f"https://newsapi.org/v2/everything?{urlencode(params)}"
    data = _fetch_json(url)
    if data.get("status") != "ok":
        message = data.get("message", "Unknown error")
        raise ValueError(f"NewsAPI (everything): {message}")

    items = []
    for article in data.get("articles", []):
        title = article.get("title", "") or ""
        published = article.get("publishedAt", "")
        date = pd.to_datetime(published, errors="coerce")
        if pd.isna(date):
            continue
        items.append({
            "date": date.normalize(),
            "title": title,
            "headline": title,
            "link": article.get("url", ""),
            "source": (article.get("source") or {}).get("name", ""),
        })
    return items


def _fetch_newsapi_top_headlines(days_back: int) -> List[Dict]:
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return []

    params = {
        "country": "in",
        "category": "business",
        "pageSize": 100,
        "apiKey": api_key,
    }
    url = f"https://newsapi.org/v2/top-headlines?{urlencode(params)}"
    data = _fetch_json(url)
    if data.get("status") != "ok":
        message = data.get("message", "Unknown error")
        raise ValueError(f"NewsAPI (top-headlines): {message}")

    cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=days_back)
    items = []
    for article in data.get("articles", []):
        title = article.get("title", "") or ""
        published = article.get("publishedAt", "")
        date = pd.to_datetime(published, errors="coerce")
        if pd.isna(date) or date.normalize() < cutoff:
            continue
        items.append({
            "date": date.normalize(),
            "title": title,
            "headline": title,
            "link": article.get("url", ""),
            "source": (article.get("source") or {}).get("name", ""),
        })
    return items


def _fetch_gnews(days_back: int) -> List[Dict]:
    api_key = os.getenv("GNEWS_API_KEY")
    if not api_key:
        return []

    from_date = (pd.Timestamp.utcnow() - pd.Timedelta(days=days_back)
                 ).strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "q": "NIFTY OR Sensex OR Indian stock market",
        "lang": "en",
        "max": 100,
        "from": from_date,
        "token": api_key,
    }
    url = f"https://gnews.io/api/v4/search?{urlencode(params)}"
    data = _fetch_json(url)

    items = []
    for article in data.get("articles", []):
        title = article.get("title", "") or ""
        published = article.get("publishedAt", "")
        date = pd.to_datetime(published, errors="coerce")
        if pd.isna(date):
            continue
        items.append({
            "date": date.normalize(),
            "title": title,
            "headline": title,
            "link": article.get("url", ""),
            "source": article.get("source", {}).get("name", ""),
        })
    return items


def fetch_all_news(days_back: int = 7, feeds: List[str] | None = None) -> pd.DataFrame:
    feeds = feeds or DEFAULT_FEEDS
    cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=days_back)

    items = []
    errors: list[str] = []

    # API sources (optional)
    try:
        items.extend(_fetch_newsapi(days_back))
    except Exception as exc:
        errors.append(str(exc))
    try:
        items.extend(_fetch_newsapi_top_headlines(days_back))
    except Exception as exc:
        errors.append(str(exc))
    try:
        items.extend(_fetch_gnews(days_back))
    except Exception:
        errors.append("GNews failed")

    # RSS sources (fallback)
    for url in feeds:
        parsed = feedparser.parse(url)
        for entry in parsed.entries:
            item = _parse_entry(entry)
            if item["date"] >= cutoff:
                items.append(item)

    if not items:
        df = pd.DataFrame(
            columns=["date", "title", "headline", "link", "source"])
        df.attrs["errors"] = errors
        return df

    df = pd.DataFrame(items)
    df = df.drop_duplicates(subset=["title", "link"])
    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    df.attrs["errors"] = errors
    return df
