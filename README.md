# AI Dashboard Agent

This project converts tabular files (Excel/CSV) into an automatic Streamlit dashboard.

Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the app with Streamlit:

```bash
streamlit run main.py
```

3. (Optional) Set `GOOGLE_API_KEY` in your environment or `.env` to enable Gemini insights.

Project layout

- `main.py` — minimal launcher
- `ai_agent/` — package containing core logic:
  - `dashboard_agent.py` — column inference and chart planning
  - `data_io.py` — file reading utilities
  - `dashboard_ui.py` — Streamlit UI
- `config/column_hints.yaml` — customizable column name hints
- `data/` — sample datasets

Next steps

- Add tests with `pytest` for `DashboardAgent` inference and request parsing.
- Add CI workflow to run tests and linting.
- Tunable config per-team in `config/column_hints.yaml`.
