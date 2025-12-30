#!/usr/bin/env python
"""Run the JAVELANCE interactive Dash app."""

from javelance.app import app

if __name__ == "__main__":
    print("Starting JAVELANCE Interactive Grid...")
    print("Open your browser to http://127.0.0.1:8050")
    app.run(debug=True, port=8050)
