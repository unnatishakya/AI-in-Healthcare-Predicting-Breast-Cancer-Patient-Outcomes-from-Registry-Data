## AI in Healthcare - Breast Cancer Outcome Risk Model

A small, validated predictive-analytics application that estimates the probability that a breast cancer patient is recorded as deceased by the end of registry follow-up, and flags higher-risk profiles using a threshold chosen on training data only.

**Innovation researched:** *Clairity Breast*, the first FDA-authorized (De Novo, June 2025) AI tool that predicts five-year breast cancer risk from a routine screening mammogram, and its 2026 clinical rollout and NCCN guideline inclusion. This project is an educational analog of that risk-scoring workflow (routine data in, risk estimate out, threshold to trigger action) built on tabular registry data. It is not a replica of the product. The research write-up and references are in Section 1 and Section 13 of the notebook.

### Repository contents

| File | Purpose |
|---|---|
| `ITAI2372_M03_Shakya_Unnati_Final.ipynb` | Full project: research, data checks, EDA, leakage analysis, model comparison, threshold selection, test evaluation, fairness audit, application, limitations, references |
| `app.py` | Streamlit web app that loads the trained model and scores a patient |
| `models/` | Created by the notebook: `risk_model.joblib` (trained pipeline) and `model_meta.json` (allowed inputs, threshold, test metrics) |
| `requirements.txt` | Python dependencies |

### Data

SEER breast cancer data from Kaggle: [`reihanenamdari/breast-cancer`](https://www.kaggle.com/datasets/reihanenamdari/breast-cancer), derived from the National Cancer Institute's SEER Program (November 2017 update): 4,024 female patients with infiltrating duct and lobular carcinoma diagnosed 2006-2010. The notebook reads `data/Breast_Cancer.csv` if that file exists (unzip the Kaggle `archive.zip` into a `data/` folder), and otherwise downloads the dataset with `kagglehub`. The `data/` folder is listed in `.gitignore`, so the CSV is not committed; please respect the dataset's license and do not redistribute it.

### How to run

**Google Colab / Jupyter:** open the notebook and run all cells. The first run may ask you to authenticate with Kaggle.

**Locally:**

```bash
pip install -r requirements.txt
jupyter notebook ITAI2372_M03_Shakya_Unnati_revised.ipynb   # run all cells; creates models/
streamlit run app.py                                         # launch the interactive app
```

### Headline results (test set, 805 patients, 123 deaths)

| Metric | Value |
|---|---|
| Selected model | Logistic regression (chosen by cross-validated PR-AUC) |
| ROC-AUC | 0.72 (95% CI 0.67 to 0.77) |
| PR-AUC | 0.37 (chance = 0.15) |
| Recall / precision at chosen threshold (0.10) | 79% / 22%, flagging about 54% of patients |
| Accuracy of always predicting "Alive" | 84.7% (catches no deaths) |
| ROC-AUC if `survival_months` leaked in | 0.87, versus 0.73 without it (cross-validated) |

These numbers come from the run saved in the notebook. Re-running with a different scikit-learn version may shift them slightly.

### What was done, and why

1. **Removed data leakage.** `survival_months` is only known after the outcome, so it is not used as an input. The notebook quantifies how much it would have inflated the results.
2. **Clinically ordered encoding** for tumor stage, node stage, AJCC stage and differentiation, plus one-hot encoding for the rest, all inside a single scikit-learn `Pipeline`.
3. **Honest evaluation.** Cross-validation on the training set for model choice, PR-AUC as the headline metric (the outcome is rare), a do-nothing baseline, and a single final test-set evaluation with bootstrap confidence intervals.
4. **Threshold chosen without peeking at the test set**, using out-of-fold predictions and a recall target.
5. **Calibration, permutation importance, and a subgroup audit** (race, age band, marital status) with warnings for small groups.
6. **Input validation** in the application so out-of-range or misspelled inputs fail loudly instead of producing a misleading prediction.

### Limitations and responsible use

This is a learning project. It uses 2006-2010 data, an all-cause vital-status label that ignores censoring, a single retrospective dataset, and has no external validation. It must not be used for clinical decisions, resource allocation, or patient counseling. See Section 12 of the notebook for the full discussion of bias, privacy, and next steps.

### AI assistance

Developed with assistance from Google Gemini and Claude for design, code review, and documentation, as permitted by the assignment. Results come from running the notebook end to end.
