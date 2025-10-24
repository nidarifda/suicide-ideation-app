import os
import torch
import gradio as gr
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import language_tool_python
import spacy

# ============== Setup ==============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "model"

tool = language_tool_python.LanguageToolPublicAPI('en-US')
nlp = spacy.load("en_core_web_sm")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE).eval()

# ============== Rule-based Filters ==============
WHITELIST_PHRASES = [
    "kill a chicken", "kill the engine", "kill a bug", "kill time",
    "kill the lights", "kill the mood", "kill weeds", "kill pests", "kill a program"
]
AGGRESSION_PATTERNS = [
    "kill him", "kill her", "kill them", "shoot him", "stab her",
    "i will kill you", "murder them", "attack him", "attack her"
]

def correct_text(text):
    try:
        matches = tool.check(text)
        return language_tool_python.utils.correct(text, matches)
    except Exception:
        return text

def is_false_positive(t): 
    return any(p in t.lower() for p in WHITELIST_PHRASES)

def is_aggression_not_suicide(t): 
    return any(p in t.lower() for p in AGGRESSION_PATTERNS)

def is_violence_directed_at_others(text):
    doc = nlp(text.lower())
    for token in doc:
        if token.lemma_ == "kill" and token.dep_ == "ROOT":
            for child in token.children:
                if child.dep_ in ("dobj", "nsubj", "nmod"):
                    if child.text in {"him","her","them","man","woman","enemy","guy","people"}:
                        return True
    return False

# ============== Prediction ==============
THRESHOLD = 0.70
def predict(text):
    if not text.strip():
        return {"Prediction":"—","Confidence (Suicide)":"0.0000","Corrected":""}

    corrected = correct_text(text)
    if is_false_positive(corrected) or is_aggression_not_suicide(corrected) or is_violence_directed_at_others(corrected):
        return {"Prediction":"Not Suicide","Confidence (Suicide)":"0.0000","Corrected":corrected}

    enc = tokenizer(corrected, return_tensors="pt", truncation=True, padding=True, max_length=256)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        probs = torch.nn.functional.softmax(out.logits, dim=1).flatten().cpu().numpy()
    conf_suicide = float(probs[1])
    label = "Suicide Risk" if conf_suicide >= THRESHOLD else "Not Suicide"
    return {"Prediction": label, "Confidence (Suicide)": f"{conf_suicide:.4f}", "Corrected": corrected}

# ============== Gradio Interface ==============
DESCRIPTION = """
This tool detects **suicidal intent** in short text with a fine-tuned DistilBERT model.
It applies grammar correction and rule-based filters to reduce false positives.

**Important:** This is a research tool and not a medical device.  
If you or someone you know is in crisis, please seek professional help or contact local emergency services immediately.
"""

with gr.Blocks(
    title="Suicide Ideation Detection",
    theme=gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="gray",
        neutral_hue="slate"
    ),
    css="""
        /* ==== 1. Disclaimer Box Styling ==== */
        .disclaimer-box {
            background-color: #6666ff !important;
            border-radius: 12px !important;
            padding: 16px 20px !important;
            margin: 10px 0 !important;
            border: 2px solid #8888ff !important;
        }
        
        .disclaimer-box p, 
        .disclaimer-box strong, 
        .disclaimer-box em {
            color: #000000 !important;
            margin: 0 !important;
            font-weight: 500 !important;
        }
        
        .disclaimer-box * {
            color: #000000 !important;
        }

        /* ==== 2. Main Container ==== */
        .gradio-container {
            background: linear-gradient(135deg, #0b0b12 0%, #1a1a2e 100%) !important;
            color: white !important;
            font-family: 'Segoe UI', system-ui, sans-serif !important;
        }

        /* ==== 3. Text Inputs ==== */
        .gr-textbox textarea, 
        .gr-textbox input {
            background-color: #1c1f2b !important;
            color: white !important;
            border: 1px solid #444 !important;
            border-radius: 8px !important;
            padding: 12px !important;
            font-size: 14px !important;
        }
        
        .gr-textbox textarea:focus,
        .gr-textbox input:focus {
            border-color: #ff6600 !important;
            box-shadow: 0 0 0 2px rgba(255, 102, 0, 0.2) !important;
        }

        /* ==== 4. Labels ==== */
        .gr-label {
            background-color: #2d3748 !important;
            color: white !important;
            border-radius: 6px !important;
            padding: 8px 12px !important;
            margin-bottom: 5px !important;
            font-weight: 600 !important;
        }

        /* ==== 5. Buttons ==== */
        button {
            background: linear-gradient(135deg, #ff6600 0%, #ff8533 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            border: none !important;
            padding: 12px 24px !important;
            font-size: 14px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px rgba(255, 102, 0, 0.3) !important;
        }
        
        button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 8px rgba(255, 102, 0, 0.4) !important;
            background: linear-gradient(135deg, #e65c00 0%, #ff751a 100%) !important;
        }
        
        button:active {
            transform: translateY(0) !important;
        }

        /* ==== 6. Output Labels ==== */
        .gr-box {
            background-color: #2d3748 !important;
            border: 1px solid #444 !important;
            border-radius: 8px !important;
            padding: 10px !important;
        }
        
        /* ==== 7. Prediction Output Styling ==== */
        .prediction-positive {
            background-color: #e53e3e !important;
            color: white !important;
            font-weight: bold !important;
            padding: 8px 12px !important;
            border-radius: 6px !important;
        }
        
        .prediction-negative {
            background-color: #38a169 !important;
            color: white !important;
            font-weight: bold !important;
            padding: 8px 12px !important;
            border-radius: 6px !important;
        }

        /* ==== 8. Header Styling ==== */
        .gr-markdown h1 {
            color: white !important;
            text-align: center !important;
            margin-bottom: 20px !important;
            font-weight: 700 !important;
            font-size: 2.5em !important;
            background: linear-gradient(135deg, #ff6600 0%, #ffa366 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
        }

        /* ==== 9. Footer Disclaimer ==== */
        .gr-markdown:last-of-type {
            background-color: #2d3748 !important;
            border-left: 4px solid #ff6600 !important;
            padding: 12px 16px !important;
            border-radius: 8px !important;
            margin-top: 20px !important;
        }
    """
) as demo:
    gr.Markdown("# Suicide Ideation Detection")
    
    # Disclaimer with custom class
    gr.Markdown(DESCRIPTION, elem_classes="disclaimer-box")

    inp = gr.Textbox(
        label="Enter a Tweet or Short Message",
        lines=4,
        placeholder="e.g., I can't do this anymore...",
        elem_id="input-textbox"
    )

    with gr.Row():
        out_pred = gr.Label(
            label="Prediction",
            value="—",
            elem_id="prediction-output"
        )
        out_conf = gr.Label(
            label="Confidence (Suicide)",
            value="0.0000",
            elem_id="confidence-output"
        )
    
    out_text = gr.Textbox(
        label="Corrected Message", 
        lines=4,
        elem_id="corrected-output"
    )

    btn = gr.Button(
        "🔍 Analyze Text", 
        variant="primary",
        elem_id="analyze-button"
    )
    
    btn.click(
        fn=predict,
        inputs=inp,
        outputs={
            "Prediction": out_pred,
            "Confidence (Suicide)": out_conf,
            "Corrected": out_text,
        },
    )

    gr.Markdown(
        "**Disclaimer:** Outputs may be imperfect. Use human judgment and, when in doubt, escalate to a qualified professional."
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0" if os.getenv("GRADIO_SHARE") else "127.0.0.1",
        share=bool(os.getenv("GRADIO_SHARE"))
    )
