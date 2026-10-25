---
title: JobShield AI - Fake Job Posting Detection
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.23.1
app_file: app.py
pinned: false
---

# JobShield AI — Fake Job Posting Detection

An educational NLP classification project aligned with **AI Applications Lab Project Basket, Topic 53**. The PDF specifies fake job posting detection, the Kaggle Fake Job Postings dataset, job title/description/company profile as input features, TF-IDF with Logistic Regression or Random Forest, and accuracy/precision/recall as evaluation metrics.

## What it does

- Accepts job title, company profile, description, requirements, and benefits.
- Combines text fields and transforms them with TF-IDF (unigrams + bigrams).
- Uses Logistic Regression to estimate whether a listing resembles fraudulent examples in the training dataset.
- Displays a classification, estimated fraud score, and held-out evaluation metrics.

**Important:** A model score is not proof that a listing is fake or safe. Always verify the employer independently. The app is a student prototype, not a production fraud-detection service.

## Dataset setup

Dataset: Kaggle **Real / Fake Job Posting Prediction** (`shivamb/real-or-fake-fake-jobposting-prediction`), based on the Employment Scam Aegean Dataset (EMSCAD).

### Option A — automatic download

The training script tries to download the public dataset using `kagglehub`. If Kaggle requires authentication or terms acceptance, use Option B.

```bash
pip install -r requirements.txt
python train_model.py
```

### Option B — download the CSV manually

1. Open the dataset page: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
2. Sign in to Kaggle if asked, accept the dataset terms, and download the dataset ZIP.
3. Extract `fake_job_postings.csv` into this project's `data/` folder.
4. Run `python train_model.py`.

The CSV is not included in this repository because it is a large external dataset. The trained artifact and `metrics.json` are generated in `artifacts/`.

## Run locally

Python 3.10 or 3.11 recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Open the local URL printed by Gradio (normally `http://127.0.0.1:7860`).

## Deploy to Hugging Face Spaces

1. Create or sign in to a Hugging Face account: https://huggingface.co/
2. Go to https://huggingface.co/new-space
3. Choose a unique Space name and select **Gradio** as the SDK. Set visibility to **Public** if you want your teacher to open the demo link.
4. Upload `app.py`, `train_model.py`, `requirements.txt`, and this `README.md` to the Space repository.
5. The app trains on first startup if `artifacts/fake_job_model.joblib` is missing. This requires the dataset to download successfully and can take a few minutes. If startup fails because Kaggle authentication is required, train locally/Colab, then upload both `artifacts/fake_job_model.joblib` and `artifacts/metrics.json` into an `artifacts/` folder in the Space.
6. Wait for the Space build to finish. The Space page URL is your public demo link.

Hugging Face Spaces rebuilds the app after repository changes. Public Spaces expose their source code, so do not upload secrets or API keys.

## Files

```text
fake_job_detection_project/
├── app.py                 # Gradio web app
├── train_model.py         # Dataset loading, training and evaluation
├── requirements.txt       # Python dependencies
├── README.md              # Setup and deployment guide
├── data/                  # Put CSV here (not included)
└── artifacts/             # Generated model and metrics (not included)
```

## Evaluation

The training script uses a stratified 80/20 train-test split with a fixed random seed. It prints and saves accuracy, fraud-class precision, fraud-class recall, F1-score, and confusion matrix. Do not claim performance numbers until training completes and the saved metrics are reviewed. Because fraudulent posts are a minority class, accuracy alone can be misleading; inspect precision and recall too.

## Project summary

- **Problem type:** NLP classification
- **Target:** `fraudulent` (0 = genuine, 1 = fraudulent)
- **Text fields:** title, company profile, description, requirements, benefits (uses fields available in the dataset)
- **Algorithm:** TF-IDF + Logistic Regression
- **Metrics:** Accuracy, precision, recall, and F1-score
- **Outcome:** Educational support for spotting suspicious job advertisements
