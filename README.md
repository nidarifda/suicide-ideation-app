# Suicide Ideation Detection (DistilBERT + Gradio)

A lightweight Gradio app for detecting suicide ideation in short texts.
- Fine-tuned DistilBERT (local `./model`)
- Grammar correction with LanguageTool Public API (no Java)
- Rule-based false-positive filters

## Run locally
pip install -r requirements.txt
python app.py

## Hugging Face Spaces
- Create a new Space (Gradio)
- Upload files in this repo
- Set Space hardware: CPU Basic is fine for DistilBERT inference
