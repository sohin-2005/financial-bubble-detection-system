"""Offline training and data pipeline.

These scripts produced the artifacts in data/models/ and models/. They are not
imported by the dashboard at request time, with one exception: app.py imports
`training.macro` to rebuild data/exports/macro_daily.csv when it goes stale.

Most scripts here resolve their inputs relative to the current directory, so run
them from inside this folder:

    cd src/training
    python pipeline.py
"""
