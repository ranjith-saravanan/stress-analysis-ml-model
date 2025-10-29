# Social Media Detox Effect Analyzer

This repository contains the Social Media Detox Effect Analyzer — an AI-driven project to simulate, analyze, and predict the effects of reducing social media use on happiness, stress, and related wellbeing metrics.

Contents
- `Social_Media_Detox_Effect_Analyzer_Project_Plan.ipynb` — Jupyter notebook: project plan, synthetic data generator, EDA, preprocessing, modeling, and prototype dashboard code.
- `prompts/` — AI assistant prompt templates to accelerate development.
- `data/` — place your `Mental_Health_and_Social_Media_Balance_Dataset.csv` here (not included for privacy).

How to run (quick)
1. Create a Python 3.10+ virtual environment and activate it.

```powershell
python -m venv .env
# Windows PowerShell
.\.env\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Open the notebook in VS Code or Jupyter and run cells.

3. To run the Streamlit demo (if implemented):

```powershell
streamlit run streamlit_app.py
```

Notes
- The `data/` folder should contain your dataset CSV named `Mental_Health_and_Social_Media_Balance_Dataset.csv`.
- For privacy, do not commit sensitive personal data to public repos.

License
- Add a license of your choice.
