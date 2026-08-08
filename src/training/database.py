import os
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load .env from the repo root, regardless of the current working directory.
# __file__ = <repo>/src/training/database.py  ->  parents[2] = <repo>
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

def get_engine():
    """Create and return a SQLAlchemy database engine."""
    user     = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host     = os.getenv("DB_HOST")
    port     = os.getenv("DB_PORT")
    name     = os.getenv("DB_NAME")
    
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
    engine = create_engine(url)
    return engine


def create_tables(engine):
    """Create all required tables if they don't already exist."""
    with engine.connect() as conn:
        
        # Table 1: Raw market data
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS market_data (
                id          SERIAL PRIMARY KEY,
                ticker_id   VARCHAR(20) NOT NULL,
                date        DATE NOT NULL,
                open        FLOAT,
                high        FLOAT,
                low         FLOAT,
                close       FLOAT,
                volume      BIGINT,
                log_return  FLOAT,
                UNIQUE(ticker_id, date)
            );
        """))
        
        # Table 2: Z-score labels
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS bubble_analysis (
                id             SERIAL PRIMARY KEY,
                ticker_id      VARCHAR(20) NOT NULL,
                date           DATE NOT NULL,
                zscore_value   FLOAT,
                rolling_mean   FLOAT,
                rolling_std    FLOAT,
                label          VARCHAR(10),
                UNIQUE(ticker_id, date)
            );
        """))
        
        # Table 3: Sentiment (Arjun will fill this)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sentiment_logs (
                id             SERIAL PRIMARY KEY,
                ticker_id      VARCHAR(20),
                source         VARCHAR(100),
                text_content   TEXT,
                finbert_score  FLOAT,
                polarity_score FLOAT,
                confidence     FLOAT,
                timestamp      TIMESTAMP
            );
        """))
        
        # Table 4: Forecast results (Jerome will fill this)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS forecast_results (
                id              SERIAL PRIMARY KEY,
                ticker_id       VARCHAR(20),
                prediction_date DATE,
                bubble_prob     FLOAT,
                crash_prob      FLOAT,
                normal_prob     FLOAT,
                model_used      VARCHAR(50)
            );
        """))
        
        conn.commit()
    
    print("All tables created successfully.")


if __name__ == "__main__":
    engine = get_engine()
    create_tables(engine)
    print("Database setup complete.")
