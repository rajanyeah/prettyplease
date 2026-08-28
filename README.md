# prettyplease 💋

A content-based beauty product recommender built on the Nykaa product catalog,
free-text search or "find me something similar to X," with an optional
AI-grounded explanation layer.

**Live demo:** [https://prettyplease.streamlit.app]
**Full documentation:** [DOCUMENTATION.md](./DOCUMENTATION.md) — architecture,
methodology, evaluation, known limitations, and test cases
**Dataset:** [https://www.kaggle.com/datasets/susant4learning/nykaacosmeticsproductsreview2021]
## Quick start

```bash
git clone [your repo url]
cd prettyplease
pip install -r requirements.txt
streamlit run app.py
```

Add `.streamlit/secrets.toml` with `GEMINI_API_KEY` for the optional AI
explanation feature — the core recommender works without it.
