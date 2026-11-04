"""Platform engine: pure functions that turn any experiment table into a decision.

Nothing in here touches DuckDB, files or Streamlit, so the same code serves the notebooks, the CLI,
the FastAPI service and the Streamlit app.
"""
