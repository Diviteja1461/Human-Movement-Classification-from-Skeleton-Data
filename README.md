# Human Movement Classification from Skeleton Data

**Course:** Representation Learning: From Neural Networks to Transformers — THD Campus Cham  
**Task:** Task 3 — Human Movement Classification  
🚀 **Live Demo:** [Open Streamlit App](https://human-movement-classification-from-skeleton-data-fcwwcu2pdhwhq.streamlit.app/)

---

## What This Project Does

A camera records a person performing an activity. OpenPose tracks 25 body joints in every frame. Our models read the joint positions over time and predict which activity is being performed.

---

## Activities Classified

| Label | Activity | Description |
|---|---|---|
| 0 | Boxing | Fast wrist punch forward and back |
| 1 | Drums | Both arms alternating up-down strikes |
| 2 | Guitar | Both arms low at waist, finger movements |
| 3 | Rowing | Full body lean forward and back |
| 4 | Violin | Left arm raised and still, right arm slow arc |

---

## Dataset

| | |
|---|---|
| **Provider** | OpenPose BODY_25 skeleton recordings |
| **Train files** | 1,167 labelled CSV files |
| **Test files** | 305 unlabelled CSV files |
| **Each CSV** | Variable-length time series — one row per frame |
| **Columns** | 79 total = 25 joints × (X, Y, Confidence) + 4 angle features |
| **Label** | Encoded in filename e.g. `13812481_violin.csv` |

---

## Key Finding — Duplicate Data

> **"The first half of each CSV file is an EXACT DUPLICATE of the second half."**

Each CSV has ~300 rows. Rows 0–149 are duplicates (no angle features), rows 150–299 are the real data (all 79 columns). Fix applied: only the second half of every CSV is loaded for training.

**Impact:**
- Removed 50% useless duplicate rows
- Reduced training time significantly
- Improved model quality — no repeated patterns confusing training

---

## Models & Results

| Model | Val Accuracy | Val F1 | Parameters | Train Time |
|---|---|---|---|---|
| Random Forest (baseline) | 79.9% | 79.7% | 6.09M | 0.6s |
| LSTM | 83.8% | 83.7% | 275,461 | 272.8s |
| **GRU ⭐ (best)** | **93.6%** | **93.6%** | **209,413** | 273.7s |
| Transformer | 88.0% | 88.0% | 408,709 | 127.7s |
| BiLSTM | 88.0% | 88.1% | 678,917 | 616.2s |

**GRU wins** — fewest parameters among deep models → least overfitting on 934 training files.

---

## Streamlit App Features

The `app.py` includes a full interactive demo with 4 tabs:
- **Classify** — upload a skeleton CSV and get live predictions from all 4 models
- **Dataset Explorer** — visualise class distribution, sequence lengths, and skeleton animations
- **Model Comparison** — bar charts and metrics table comparing all models
- **About** — project overview and how to run locally

---

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Diviteja1461/Human-Movement-Classification-from-Skeleton-Data.git
cd Human-Movement-Classification-from-Skeleton-Data

# 2. Install dependencies
pip install streamlit torch pandas numpy matplotlib scikit-learn scipy tqdm

# 3. Add trained model files (from your notebook)
# Place gru_model.pt, lstm_model.pt, transformer_model.pt, bilstm_model.pt
# in the same folder as app.py

# 4. Run the app
streamlit run app.py
```

---

## Notebook Structure

| Section | Content |
|---|---|
| S1 | Imports and setup |
| S2 | Paths and label mapping |
| S3 | Duplicate data analysis and proof |
| S4 | Load data (second half only) |
| S5 | Normalisation |
| S6 | EDA — distributions, skeleton plots, animation |
| S7 | 80/20 train/validation split |
| S8 | Random Forest baseline |
| S9 | Deep learning setup |
| S10 | LSTM |
| S11 | GRU |
| S12 | Transformer Encoder |
| S13 | BiLSTM (bonus) |
| S14 | Evaluation — metrics + confusion matrices |
| S15 | Attention visualisation |
| S16 | Complexity analysis |
| S17 | Test predictions + final summary |

---

## Tech Stack

- Python, PyTorch, Streamlit
- scikit-learn, NumPy, Pandas
- Matplotlib, Seaborn, SciPy
- CUDA (GPU training)

---

## References

- **Attention Is All You Need**: Vaswani et al. (2017) — arXiv:1706.03762
