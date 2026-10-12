# Next Word Prediction System — Web Application

A production-style web application that serves **trained next-word
prediction models** (TensorFlow / Keras) over a REST API with a Bootstrap
frontend. Three recurrent architectures — **SimpleRNN, GRU, and LSTM** — are
trained on the same corpus with identical preprocessing and can be selected
per request; a comparison table (accuracy / perplexity / params / speed) is
produced by the training script. An optional DistilGPT2 script
(`notebooks/distilgpt2_compare.py`) provides a modern-transformer reference
point. It converts the experimentation notebook (`NX-NLP-2026-014`) into a
modular, deployable service.

The app **does not retrain** anything — it loads the existing trained
`.keras` model and pickled tokenizer and reproduces the notebook's
preprocessing, top-k prediction, and temperature-sampling logic exactly.

---

## Features

- Enter a sentence/phrase and **predict the next word**
- **Switch between LSTM / GRU / SimpleRNN** per request (UI dropdown or
  `"model"` field in the API); `GET /models` lists what's available
- **Top-k predictions** with probabilities (k = 3 / 5 / 10)
- **Multi-word text generation** from a seed
- Adjustable **temperature** (0 = greedy, higher = more creative)
- REST API with automatic **OpenAPI docs** at `/docs`
- Bootstrap single-page UI
- Dockerfile + Render/Railway configs for deployment
- Test suite that runs **without TensorFlow** (fakes) plus real-artifact readiness

---

## Project layout

```
next_word_predictor/
├── app/
│   ├── main.py              # FastAPI entrypoint (app instance)
│   ├── routes/              # /predict, /generate, /health
│   ├── services/            # predictor.py (predict_next_word, generate_text)
│   ├── models/              # model_loader, tokenizer_loader, schemas
│   ├── utils/               # text_cleaner, text_normalizer
│   └── config/              # settings.py
├── static/                  # css/, js/
├── templates/               # index.html
├── notebooks/               # export_artifacts.py (save model/tokenizer/config)
├── saved_models/            # <- put .keras + tokenizer.pkl + config.json here
├── data/                    # optional Articles.csv sample
├── tests/                   # pytest suite
├── requirements.txt
├── Dockerfile / Procfile / render.yaml
└── README.md
```

> **Naming note.** Python cannot have both a top-level `app.py` module and an
> `app/` package (the package shadows the module). The runnable entrypoint
> therefore lives at **`app/main.py`**, launched with `uvicorn app.main:app`.

---

## 1. Installation

```bash
# clone / unzip, then:
cd next_word_predictor

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **Trained models are already bundled** in `saved_models/` (LSTM, GRU, RNN +
> tokenizer/config). You do **not** need to train anything — go straight to
> section 2 to run the app.

> ⚠️ **Keras version matters.** The `.keras` artifacts were exported with
> **Keras 3.13.2**. `requirements.txt` therefore pins both `tensorflow==2.16.1`
> **and** `keras==3.13.2`. Installing TensorFlow alone pulls Keras 3.0.5, which
> **cannot** read these artifacts and produces the "model not loaded" error.
> Always install via `pip install -r requirements.txt`.

### Train the models

```bash
python notebooks/train_models.py --data Articles.csv --out saved_models
```

This trains all three architectures and writes everything the app needs:

| File | Purpose |
|------|---------|
| `next_word_lstm_best.keras` | LSTM model (default) |
| `next_word_gru_best.keras` | GRU model |
| `next_word_rnn_best.keras` | SimpleRNN baseline |
| `tokenizer.pkl` | fitted tokenizer, shared by all three |
| `config.json` | context length + hyperparameters |
| `comparison.json` / `.csv` | metrics table for the video |
| `training_curves.png` | validation loss/accuracy curves |

The app serves whichever architectures it finds on disk — a single LSTM
artifact still works exactly as before.

Use `notebooks/export_artifacts.py` — paste its block into your notebook after
training to produce all three files.

---

## 2. Run locally

```bash
uvicorn app.main:app --reload
# or
python app/main.py
```

- UI: <http://localhost:8000/>
- Interactive API docs: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health>

## ⚠️ Challenges Faced and Solutions

During the development of this Next Word Prediction System, I encountered several challenges and explored different approaches to improve prediction quality.

### 1. Lower Accuracy of RNN, LSTM, and GRU Models
- **Challenge:** The SimpleRNN, LSTM, and GRU models produced lower prediction accuracy than expected.
- **Solution:** I explored a pretrained DistilGPT-2 model to generate more contextually relevant next-word predictions.

### 2. Understanding Long-Range Dependencies
- **Challenge:** Recurrent models sometimes struggled to capture relationships between words that were far apart in a sentence.
- **Solution:** I explored DistilGPT-2, which uses the Transformer architecture and self-attention to understand contextual relationships.

### 3. Comparing Different Model Architectures
- **Challenge:** Comparing recurrent neural networks with a pretrained Transformer model required considering their different architectures and prediction approaches.
- **Solution:** I explored the models' prediction outputs to understand their strengths and limitations.

### 4. Generating Contextually Relevant Predictions
- **Challenge:** Producing relevant next-word suggestions for different input sentences was challenging.
- **Solution:** I used a pretrained language model approach to explore context-aware text generation.

### Key Learning
This project helped me understand the differences between recurrent neural networks and Transformer-based language models, and how pretrained models such as DistilGPT-2 can be used for next-word prediction.