"""
REAL-TIME NEWS FETCHER  (college-network friendly version)
===========================================================
Fetches LIVE financial news. Works even on restricted college/institute networks.

SOURCES (tried in this priority order):
  1. NewsAPI   — HTTPS port 443, works on most networks  ← PRIMARY
  2. GNews     — HTTPS port 443, works on most networks  ← SECONDARY
  3. RSS Feeds — may be blocked on college networks      ← FALLBACK

SETUP (one time, takes 2 minutes):
  Step 1: Go to  https://newsapi.org/register  (use phone hotspot if needed)
  Step 2: Sign up free → copy your API key
  Step 3: Open your .env file and add:   NEWS_API_KEY=your_key_here
  Step 4: Run:  python src/news_fetcher.py

WHY NEWSAPI WORKS ON COLLEGE NETWORKS:
  RSS feeds use odd ports/domains that firewalls often block.
  NewsAPI uses standard HTTPS (port 443) — looks like normal web traffic,
  almost never blocked.

Free tier limits:
  NewsAPI → 100 requests/day, articles from last 30 days
  GNews   → 100 requests/day
"""

import os
import re
import time
import hashlib
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv

# Always load .env from project root, regardless of where script is run from
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

# ─────────────────────────────────────────────────────────────────────────────
# SEARCH QUERIES
# ─────────────────────────────────────────────────────────────────────────────

NEWSAPI_QUERIES = [
    "stock market",
    "financial markets economy",
    "NIFTY OR Sensex OR BSE OR NSE",
    "interest rate inflation economy",
]

GNEWS_QUERIES = [
    "NIFTY Sensex India stock market",
    "Indian economy inflation RBI",
]

MARKET_KEYWORDS = [
    "sensex", "nifty", "bse", "nse", "stock", "market", "share", "equity",
    "rally", "crash", "bull", "bear", "ipo", "fii", "dii", "sebi",
    "rupee", "inflation", "rbi", "rate", "gdp", "earnings", "profit",
    "loss", "dividend", "buyback", "merger", "acquisition", "trading",
]

RSS_FEEDS = {
    "Economic Times – Markets":
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Economic Times – Stocks":
        "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "Moneycontrol":
        "https://www.moneycontrol.com/rss/MCrecentnews.xml",
    "Business Standard":
        "https://www.business-standard.com/rss/markets-106.rss",
    "LiveMint":
        "https://www.livemint.com/rss/markets",
    "Financial Express":
        "https://www.financialexpress.com/market/feed/",
    "NDTV Profit":
        "https://feeds.feedburner.com/ndtvprofit-latest",
    "Reuters – Business":
        "https://feeds.reuters.com/reuters/businessNews",
    "Reuters – Markets":
        "https://feeds.reuters.com/reuters/INbusinessNews",
}


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY: NEWSAPI
# ─────────────────────────────────────────────────────────────────────────────

def fetch_newsapi(days_back: int = 7, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch Indian financial news from NewsAPI.org.
    Works on virtually all networks including college/institute WiFi.
    """
    api_key = api_key or os.getenv("NEWS_API_KEY", "").strip()

    if not api_key:
        print("  ⚠️  NewsAPI: No key set.")
        print("       Register FREE at https://newsapi.org/register")
        print("       Then add to .env:  NEWS_API_KEY=your_key")
        return pd.DataFrame()

    from_date = (datetime.now() - timedelta(days=min(days_back, 29))
                 ).strftime("%Y-%m-%d")
    all_articles = []
    seen_hashes = set()

    print(f"  Fetching from NewsAPI (last {days_back} days)...")

    for query in NEWSAPI_QUERIES:
        params = {
            "q":        query,
            "from":     from_date,
            "sortBy":   "publishedAt",
            "language": "en",
            "pageSize": 100,
            "apiKey":   api_key,
        }
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params=params,
                timeout=15,
            )
            if resp.status_code == 401:
                print("  ✗ NewsAPI: Invalid API key — check your .env file.")
                return pd.DataFrame()
            if resp.status_code == 429:
                print("  ✗ NewsAPI: Daily limit hit (100/day). Try tomorrow.")
                return pd.DataFrame()
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            print("  ✗ NewsAPI: No connection. Check internet.")
            return pd.DataFrame()
        except Exception as exc:
            print(f"  ✗ NewsAPI error: {exc}")
            return pd.DataFrame()

        for item in data.get("articles", []):
            headline = _clean(item.get("title", ""))
            if not headline or headline == "[Removed]":
                continue
            h = hashlib.md5(headline.lower().encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            pub = _parse_iso(item.get("publishedAt", ""))
            if pub is None:
                continue
            summary = _clean(item.get("description", ""))
            all_articles.append({
                "date":            pub.strftime("%Y-%m-%d"),
                "datetime":        pub,
                "source":          item.get("source", {}).get("name", "NewsAPI"),
                "headline":        headline,
                "summary":         summary[:300],
                "url":             item.get("url", ""),
                "relevance_score": _relevance(headline + " " + summary),
            })
        time.sleep(0.4)

    df = pd.DataFrame(all_articles)
    if df.empty:
        print("  ✗ NewsAPI: No articles returned.")
        return df

    df.drop_duplicates(subset=["headline"], inplace=True)
    df.sort_values("datetime", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    relevant = df[df["relevance_score"] >= 1]
    print(f"  ✓ NewsAPI → {len(df)} total  |  {len(relevant)} market-relevant")
    return relevant


# ─────────────────────────────────────────────────────────────────────────────
# SECONDARY: GNEWS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_gnews(days_back: int = 7, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch from GNews API.  Get free key at https://gnews.io/
    Add to .env:  GNEWS_API_KEY=your_key
    """
    api_key = api_key or os.getenv("GNEWS_API_KEY", "").strip()
    if not api_key:
        print("  ⚠️  GNews: No key. Get one free at https://gnews.io/")
        return pd.DataFrame()

    from_dt = (datetime.now() - timedelta(days=days_back)
               ).strftime("%Y-%m-%dT%H:%M:%SZ")
    all_articles = []
    seen_hashes = set()

    print(f"  Fetching from GNews (last {days_back} days)...")

    for query in GNEWS_QUERIES:
        try:
            resp = requests.get(
                "https://gnews.io/api/v4/search",
                params={"q": query, "lang": "en", "country": "in",
                        "max": 100, "from": from_dt, "apikey": api_key},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"  ✗ GNews error: {exc}")
            continue

        for item in data.get("articles", []):
            headline = _clean(item.get("title", ""))
            if not headline:
                continue
            h = hashlib.md5(headline.lower().encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            pub = _parse_iso(item.get("publishedAt", ""))
            if pub is None:
                continue
            summary = _clean(item.get("description", ""))
            all_articles.append({
                "date":            pub.strftime("%Y-%m-%d"),
                "datetime":        pub,
                "source":          item.get("source", {}).get("name", "GNews"),
                "headline":        headline,
                "summary":         summary[:300],
                "url":             item.get("url", ""),
                "relevance_score": _relevance(headline + " " + summary),
            })
        time.sleep(0.4)

    df = pd.DataFrame(all_articles)
    if not df.empty:
        df.drop_duplicates(subset=["headline"], inplace=True)
        df.sort_values("datetime", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"  ✓ GNews → {len(df)} articles")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK: RSS FEEDS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_rss_news(days_back: int = 7) -> pd.DataFrame:
    """
    Try RSS feeds. Silently returns empty DataFrame if network blocks them.
    On college networks these usually return 0 articles — that is normal.
    """
    try:
        import feedparser
    except ImportError:
        return pd.DataFrame()

    cutoff = datetime.now() - timedelta(days=days_back)
    articles = []
    seen = set()
    any_success = False

    for name, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            count = 0
            for entry in feed.entries[:50]:
                pub = _parse_feedparser_date(entry)
                if pub is None or pub < cutoff:
                    continue
                headline = _clean(getattr(entry, "title", ""))
                if not headline:
                    continue
                h = hashlib.md5(headline.lower().encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                summary = _clean(getattr(entry, "summary", ""))
                articles.append({
                    "date":            pub.strftime("%Y-%m-%d"),
                    "datetime":        pub,
                    "source":          name,
                    "headline":        headline,
                    "summary":         summary[:300],
                    "url":             getattr(entry, "link", ""),
                    "relevance_score": _relevance(headline + " " + summary),
                })
                count += 1
            if count > 0:
                any_success = True
                print(f"  ✓ RSS {name:35s} → {count:3d} articles")
            time.sleep(0.3)
        except Exception:
            continue

    if not any_success:
        print("  ℹ️  RSS: All feeds blocked (restricted network). Using API sources only.")

    df = pd.DataFrame(articles)
    if not df.empty:
        df = df[df["relevance_score"] >= 1]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_news(days_back: int = 7) -> pd.DataFrame:
    """
    Fetch from all sources. Safe to call even with no keys — will just
    return empty DataFrame and the pipeline skips sentiment silently.
    """
    print("\n" + "=" * 55)
    print("  REAL-TIME NEWS AGGREGATOR")
    print("=" * 55)

    has_newsapi = bool(os.getenv("NEWS_API_KEY", "").strip())
    has_gnews = bool(os.getenv("GNEWS_API_KEY", "").strip())

    if not has_newsapi and not has_gnews:
        print("\n  ⚠️  NO API KEYS FOUND — AND RSS IS BLOCKED ON YOUR NETWORK")
        print("  ─────────────────────────────────────────────────────")
        print("  Quick fix (2 minutes):")
        print("  1. Open  https://newsapi.org/register  on your phone hotspot")
        print("  2. Sign up free, copy your API key")
        print("  3. Add to your .env file:   NEWS_API_KEY=paste_key_here")
        print("  4. Re-run:  python src/news_fetcher.py")
        print("  ─────────────────────────────────────────────────────\n")

    frames = []

    if has_newsapi:
        df_api = fetch_newsapi(days_back=days_back)
        if not df_api.empty:
            frames.append(df_api)

    if has_gnews:
        df_gnews = fetch_gnews(days_back=days_back)
        if not df_gnews.empty:
            frames.append(df_gnews)

    # Always try RSS — works on open networks, silently skipped on restricted ones
    df_rss = fetch_rss_news(days_back=days_back)
    if not df_rss.empty:
        frames.append(df_rss)

    if not frames:
        print("\n  ❌ No news collected. Pipeline will run without sentiment.")
        print("     Add NEWS_API_KEY to .env to enable live news.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined.drop_duplicates(subset=["headline"], inplace=True)
    combined.sort_values("datetime", ascending=False, inplace=True)
    combined.reset_index(drop=True, inplace=True)

    print(f"\n  ✅ Total unique articles: {len(combined)}")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────────────────────────────────────

def save_news(df: pd.DataFrame, filename: str = "raw_news.csv") -> None:
    os.makedirs("data", exist_ok=True)
    df_save = df.copy()
    for col in df_save.columns:
        if df_save[col].dtype == object:
            df_save[col] = df_save[col].apply(
                lambda x: str(x) if isinstance(x, list) else x
            )
    df_save.to_csv(f"data/{filename}", index=False)
    print(f"  💾 Saved → data/{filename}")


def load_news(filename: str = "raw_news.csv") -> pd.DataFrame:
    return pd.read_csv(f"data/{filename}", parse_dates=["date", "datetime"])


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+",     " ", text)
    return text.strip()


def _relevance(text: str) -> int:
    t = text.lower()
    return sum(1 for kw in MARKET_KEYWORDS if kw in t)


def _parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        # fromisoformat handles all ISO 8601 formats; replace Z → +00:00 for Python < 3.11
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.replace(tzinfo=None)
    except ValueError:
        pass
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d")
    except ValueError:
        return None


def _parse_feedparser_date(entry) -> Optional[datetime]:
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        val = getattr(entry, attr, None)
        if val:
            try:
                return datetime(*val[:6])
            except Exception:
                pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    news = fetch_all_news(days_back=3)

    if not news.empty:
        print("\n── Sample Headlines ─────────────────────────────────────")
        for _, row in news.head(8).iterrows():
            print(f"\n  [{row['date']}]  {row['source']}")
            print(f"  {row['headline'][:90]}")
        save_news(news)
        print(f"\n  ✅ {len(news)} articles saved to data/raw_news.csv")
    else:
        print("\n  No news fetched.")
        print("  → Add NEWS_API_KEY to your .env file")
        print("  → Get free key: https://newsapi.org/register")
