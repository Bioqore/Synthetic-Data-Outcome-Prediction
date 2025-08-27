#!/usr/bin/env python3
"""
Main launcher for the Flask web portal
-------------------------------------

Usage:
  python3 main.py

This starts the web portal (app.py). Users can upload a training CSV and an
input CSV and receive predicted outcomes for download.
"""

import os
import runpy
import sys

if __name__ == "__main__":
    # Optional: environment tweaks can be done here
    os.environ.setdefault("FLASK_SECRET_KEY", "dev-secret-key")

    # Delegate to the Flask app
    # Equivalent to `python -m flask run` with app factory, but we run app.py directly
    sys.argv = [sys.argv[0]]  # ensure app.py doesn't see unexpected args
    runpy.run_path("app.py", run_name="__main__")
