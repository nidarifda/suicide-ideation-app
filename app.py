import os
import torch
import gradio as gr

# --- NLP + tooling ---
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import language_tool_python

# Optional: lightweight spaCy for rule-based parsing (no heavy pipelines)
import spacy

# ============== Safety & Setup ==============

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "model"   # folder in repo

# LanguageTool: use public API (no Java needed)
# If you get rate-limited, you can switch to a hosted LT server later.
tool = language_tool_python.LanguageToolPublicAPI('en-US')

# Load a tiny spaCy English pipeline (tokenizer/lemmatizer/pos)
# The model is installed via requirements.txt
nlp = spacy.load("en_core_web_sm")

# Load fine-tuned model + tokenizer from local folder
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE).eval()

# ============== Rule-based Filters ==============

WHITELIST_PHRASES = [
    "kill a chicken", "kill the engine", "kill a bug", "kill time",
    "kill the lights", "kill the mood", "kill weeds", "kill pests",
    "kill a program"
]

AGGRESSION_PATTERNS = [
    "kill him", "kill her", "kill them", "shoot him", "stab her",
    "i will kill you", "murder them", "attack him", "attack her"
]

def correct_text(text: str) -> str:
    """Grammar/spelling correction with LanguageTool public API."""
    try:
        matches = tool.check(text)
        return language_tool_python.utils.correct(text, matches)
    except Exception:
        # Fallback: return original text if API fails
        return text

def is_false_positive(text: str) -> bool:
    t = text.lower()
    return any(phrase in t for phrase in WHITELIST_PHRASES)

def is_aggression_not_suicide(text: str) -> bool:
    t = text.lower()
    return any(phrase in t for phrase in AGGRESSION_PATTERNS)

def is_violence_directed_at_others(text: str) -> bool:
    doc = nlp(text.lower())
    # simple heuristic: "kill" root verb with object pronoun/noun != self
    for token in doc:
        if token.lemma_ == "kill" and token.dep_ == "ROOT":
            for child in token.children:
                if child.dep_ in ("dobj", "nsubj", "nmod"):
                    if child.text in {"him","her","them","man","woman","enemy","guy","people"}:
                        return True
    return False

# ============== Inference ==============

THRESHOLD = 0.70  # confidence threshold for Suicide Risk

def predict(text: str):
    if not text or not text.strip():
        return {"Prediction": "—", "Confidence (Suicide)": "0.0000", "Corrected": ""}

    corrected = correct_text(text)

    # Rule-based early exits (reduce false positives)
    if is_false_positive(corrected) or is_aggression_not_suicide(corrected) or is_violence_directed_at_others(corrected):
        return {"Prediction": "Not Suicide", "Confidence (Suicide)": "0.0000", "Corrected": corrected}

    # Transformer inference
    enc = tokenizer(corrected, return_tensors="pt", truncation=True, padding=True, max_length=256)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        probs = torch.nn.functional.softmax(out.logits, dim=1).flatten().cpu().numpy()

    conf_suicide = float(probs[1])
    label = "Suicide Risk" if conf_suicide >= THRESHOLD else "Not Suicide"

    return {"Prediction": label, "Confidence (Suicide)": f"{conf_suicide:.4f}", "Corrected": corrected}

# ============== Gradio UI ==============

DESCRIPTION = """
This tool detects **suicidal intent** in short text with a fine‑tuned DistilBERT model.
It applies grammar correction and rule-based filters to reduce false positives.

**Important:** This is a research tool and not a medical device.  
If you or someone you know is in crisis, please seek professional help or contact local emergency services immediately.
"""

with gr.Blocks(
    title="Suicide Ideation Detection",
    css="""
        /* Make the top info/disclaimer box purple-blue (#6666ff) */
        .gr-block.gr-markdown:first-child div {
            background-color: #6666ff !important;
            color: white !important;
            padding: 12px !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
        }

        /* App background */
        body, .gradio-container {
            background-color: #0b0b12 !important;
            color: white !important;
        }

        /* Text boxes and labels */
        textarea, input, .gr-textbox, .gr-label {
            background-color: #1c1f2b !important;
            color: white !important;
        }

        /* Buttons */
        button {
            background-color: #ff6600 !important;
            color: white !important;
            font-weight: bold !important;
        }

        button:hover {
            background-color: #e65c00 !important;
        }
    """
) as demo:
    gr.Markdown("# Suicide Ideation Detection")
    gr.Markdown(DESCRIPTION)

    inp = gr.Textbox(label="Enter a Tweet or Short Message", lines=4, placeholder="e.g., I can't do this anymore...")

    with gr.Row():
        out_pred = gr.Label(label="Prediction")
        out_conf = gr.Label(label="Confidence (Suicide)")
    out_text = gr.Textbox(label="Corrected Message", lines=4)

    btn = gr.Button("Analyze")
    btn.click(fn=predict, inputs=inp, outputs={"Prediction": out_pred, "Confidence (Suicide)": out_conf, "Corrected": out_text})

    gr.Markdown("**Disclaimer:** Outputs may be imperfect. Use human judgment and, when in doubt, escalate to a qualified professional.")

if __name__ == "__main__":
    demo.launch()
