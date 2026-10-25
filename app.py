from __future__ import annotations
import json
from pathlib import Path
import joblib
import gradio as gr

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "fake_job_model.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"

# If the trained artifact is absent, attempt training once on startup.
if not MODEL_PATH.exists():
    from train_model import train
    train()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
TEXT_COLUMNS = bundle.get("text_columns", ["title", "company_profile", "description", "requirements", "benefits"])
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}


def make_text(title, company_profile, description, requirements, benefits):
    values = {
        "title": title,
        "company_profile": company_profile,
        "description": description,
        "requirements": requirements,
        "benefits": benefits,
    }
    return " ".join(f"{col}: {values.get(col, '') or ''}" for col in TEXT_COLUMNS)


def predict_job(title, company_profile, description, requirements, benefits):
    if not (title or description or company_profile or requirements or benefits):
        return "### Add some job information first\nPlease enter at least a job title, description, or company profile.", "No prediction yet", 0.0
    text = make_text(title, company_profile, description, requirements, benefits)
    predicted = int(model.predict([text])[0])
    probabilities = model.predict_proba([text])[0]
    classes = list(model.named_steps["classifier"].classes_)
    fake_probability = float(probabilities[classes.index(1)]) if 1 in classes else 0.0
    if predicted == 1:
        heading = "### ⚠️ Potentially fraudulent posting"
        explanation = ("The model found text patterns associated with fraudulent postings in its training data. "
                       "Verify the employer independently before sharing personal information or paying any fee.")
        status = "Potentially fraudulent"
    else:
        heading = "### ✅ Likely genuine based on the model"
        explanation = ("The model did not classify this listing as fraudulent. This is not proof that the job is safe; "
                       "verify the company, recruiter, domain, and application process independently.")
        status = "Likely genuine"
    detail = f"{heading}\n\n**Estimated fraud score:** {fake_probability:.1%}\n\n{explanation}\n\n*Educational prototype — not a guarantee or background check.*"
    return detail, status, fake_probability


def metrics_markdown():
    if not metrics:
        return "Model evaluation metrics are not available yet. Train the model first."
    return f"""### Model evaluation (held-out test data)

| Metric | Result |
|---|---:|
| Accuracy | {metrics.get('accuracy', 0):.3f} |
| Fraud precision | {metrics.get('precision_fraud', 0):.3f} |
| Fraud recall | {metrics.get('recall_fraud', 0):.3f} |
| Fraud F1-score | {metrics.get('f1_fraud', 0):.3f} |

- Dataset rows used: **{metrics.get('rows_used', 'N/A')}**
- Training rows: **{metrics.get('train_rows', 'N/A')}**
- Test rows: **{metrics.get('test_rows', 'N/A')}**
- Model: **{metrics.get('model', 'TF-IDF + Logistic Regression')}**

*Metrics are computed by `train_model.py` from a held-out test split. They may differ if the dataset changes.*
"""

CSS = """
.gradio-container {max-width: 1120px !important; margin: auto !important;}
#hero {padding: 26px; border-radius: 18px; background: linear-gradient(120deg, #101a35, #263e6b); color: white;}
#hero h1 {color: white; margin-bottom: 8px;}
.small-note {font-size: 0.92rem; opacity: 0.88;}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"), css=CSS, title="JobShield AI | Fake Job Detection") as demo:
    gr.HTML("""<div id='hero'><h1>🛡️ JobShield AI</h1><p>AI-powered Fake Job Posting Detection</p><p class='small-note'>Paste a job listing, run the model, and review its risk signal. Built as an educational NLP project.</p></div>""")
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("## Check a job posting\nFill in any fields you have. More context can help the model analyse the text.")
            title = gr.Textbox(label="Job title", placeholder="e.g., Junior Data Analyst")
            company = gr.Textbox(label="Company profile", lines=3, placeholder="What does the company do?")
            description = gr.Textbox(label="Job description", lines=6, placeholder="Paste the full job description here...")
            requirements = gr.Textbox(label="Requirements", lines=3, placeholder="Skills, experience, qualifications...")
            benefits = gr.Textbox(label="Benefits / salary details", lines=2, placeholder="Salary, benefits, application details...")
            with gr.Row():
                check = gr.Button("🔎 Analyse posting", variant="primary")
                clear = gr.ClearButton([title, company, description, requirements, benefits])
            gr.Examples(
                examples=[
                    ["Work From Home Data Entry Clerk", "", "Earn $4,000 weekly from home. No experience needed. Send a processing fee to begin. Apply immediately by messaging us.", "No experience or qualifications required.", "High income, flexible hours."],
                    ["Software Engineering Intern", "Example Technology Company builds software tools for small businesses.", "We are seeking an intern to help test features, write Python scripts, and document bugs. The selected candidate will work with a mentor and attend weekly team meetings.", "Basic Python, Git familiarity, currently studying computer science. Apply through the careers page.", "Paid internship, mentorship, flexible schedule."],
                ], inputs=[title, company, description, requirements, benefits], label="Try sample listings"
            )
        with gr.Column(scale=2):
            gr.Markdown("## Analysis result")
            result = gr.Markdown("Your prediction will appear here.")
            status = gr.Textbox(label="Model classification", interactive=False)
            score = gr.Slider(minimum=0, maximum=1, step=0.01, label="Estimated fraud score (model output)", interactive=False)
            gr.Markdown("### Safety checklist\n- Never pay a recruiter to secure a job.\n- Check the official company website and email domain.\n- Be cautious with requests for banking details or identity documents early in the process.\n- Contact the company using independently verified details.")
    check.click(predict_job, inputs=[title, company, description, requirements, benefits], outputs=[result, status, score])
    gr.Markdown("---\n## Model performance")
    gr.Markdown(metrics_markdown())
    gr.Markdown("**Limitations:** This model learns patterns from a historical dataset. It can make false positives and false negatives, may not generalize to current listings, and must not be used as the only basis for a hiring or safety decision.")

if __name__ == "__main__":
    demo.launch()
