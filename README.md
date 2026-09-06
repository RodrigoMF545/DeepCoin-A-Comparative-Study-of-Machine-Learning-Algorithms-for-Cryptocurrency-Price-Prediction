# DeepCoin: A Comparative Study of Machine Learning Algorithms for Cryptocurrency Price Prediction

Final Year Project (Trabalho de Fim de Curso) in Computer and Telecommunications Engineering, Universidade Técnica do Atlântico (UTA), Mindelo, Cabo Verde — Academic Year 2025/2026. Graded 19/20.

## About the project

DeepCoin is a comparative study of Machine Learning and Deep Learning algorithms applied to cryptocurrency price prediction (Bitcoin, Ethereum, and Solana). Six algorithms are evaluated — **Random Forest, LSTM, GRU, Transformer, and TCN** — under two experimental conditions: with and without data augmentation techniques (jittering and magnitude warping).

The repository is organized into two complementary parts:

1. **[`data-analysis/`](./data-analysis)** — notebooks and results from the comparative study (preprocessing, augmentation, model training and evaluation).
2. **[`web-app/`](./web-app)** — interactive Streamlit dashboard for visualizing and testing predictions from the trained models.

## Repository structure

```
deepcoin/
├── docs/
│   └── relatorio.pdf          # full final-year thesis report
│
├── data-analysis/
│   ├── notebooks/
│   │   ├── 01_random_forest.ipynb
│   │   └── 02_deep_learning.ipynb
│   ├── results/                # generated metrics and figures
│   ├── requirements.txt
│   └── README.md
│
├── web-app/
│   ├── app.py
│   ├── src/
│   ├── components/
│   ├── requirements.txt
│   └── README.md
│
└── README.md                   # this file
```

## Methodology (summary)

- **Data collection:** daily historical price series for Bitcoin, Ethereum, and Solana.
- **Preprocessing:** cleaning, train/test split, normalization (MinMaxScaler).
- **Data augmentation:** jittering and magnitude warping, used to evaluate model robustness with additional synthetic data.
- **Modeling:** Random Forest, SVM, LSTM, GRU, TCN, and Transformer (with custom attention blocks and positional encoding).
- **Evaluation:** MAE, MAPE, RMSE, and R², comparing performance with and without augmentation, for each cryptocurrency.
- **Visualization:** Streamlit dashboard for interactive inspection of predictions.

Full methodology, results, and discussion are available in the report at [`docs/relatorio.pdf`](./docs/relatorio.pdf).

## Tech stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data analysis | NumPy, Pandas, SciPy |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow, Keras |
| Visualization | Matplotlib, Seaborn |
| Web app | Streamlit |
| Training environment | Kaggle Notebooks (GPU) |

## How to reproduce

See detailed instructions in each subproject:
- [`data-analysis/README.md`](./data-analysis/README.md) — reproduce model training and evaluation.
- [`web-app/README.md`](./web-app/README.md) — run the dashboard locally.

## Author

**Rodrigo Hendrick Monteiro Santos Fortes**
B.Sc. in Computer and Telecommunications Engineering — UTA
Advisor: Prof. Dr. Estanislau Lima
Co-advisor: MSc Roberto Carlos Medina

## License

_To be defined._
