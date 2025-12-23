# GQ Implied Volatility Forecasting

![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=flat&logo=kaggle&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-GBDT-02569B?style=flat&logo=lightgbm&logoColor=white)
![GPU](https://img.shields.io/badge/GPU-Enabled-76B900?style=flat&logo=nvidia&logoColor=white)
![Pearson](https://img.shields.io/badge/Pearson-0.7140-success?style=flat)

**Kaggle Competition Submission**

Forecasting cryptocurrency implied volatility using gradient boosting with cross-asset feature engineering and time-series validation [file:2].

---

## Methodology

### Data & Target
- **Primary Asset:** ETH (Ethereum) orderbook data at 1-second resolution [file:2]
- **Cross-Assets:** BTC, DOGE, DOT, LINK, SHIB, SOL [file:2]
- **Target:** Implied volatility forecast [file:2]
- **Features:** 253 engineered features [file:2]

### Feature Engineering

**ETH-Specific Features [file:2]:**
- Log returns (1s, 5s, 10s lags)
- Realized volatility, skewness, kurtosis (5s-300s windows)
- Order book imbalance across 5 levels
- Bid-ask spread dynamics
- EWMA momentum indicators
- Volatility-of-volatility

**Cross-Asset Features [file:2]:**
- Multi-asset return correlations
- Rolling 30s/60s correlation with ETH
- PCA decomposition (5 components) across asset returns
- Cross-asset volatility ratios
- Return difference features

**Temporal Features [file:2]:**
- Cyclical encoding (sin/cos) of minute, hour
- Day-of-week indicators

### Model

**Architecture:** LightGBM Gradient Boosting [file:2]
- 400 leaves, learning rate 0.008
- L1/L2 regularization (0.005)
- Extremely randomized trees
- GPU acceleration

**Validation:** 5-fold time-series walk-forward CV with 10-sample gap to prevent leakage [file:2]

---

## Results

| Metric | Score |
|--------|-------|
| **CV Overall Pearson** | 0.7015 [file:2] |
| **Holdout Pearson (Simulated Public)** | 0.7140 [file:2] |
| **Median Best Iteration** | 2103 [file:2] |

**Per-Fold Performance [file:2]:**
- Fold 1: 0.5308
- Fold 2: 0.5311  
- Fold 3: 0.6447
- Fold 4: 0.5690
- Fold 5: 0.7853

---

## Technical Implementation

**Key Design Choices:**
- Time-aligned 1-second master timeline for all assets [file:2]
- Sanitized features with quantile clipping (0.1%, 99.9%) [file:2]
- Batch prediction to handle memory constraints [file:2]
- 80/20 train-holdout split for validation [file:2]

**Dependencies:** pandas, numpy, lightgbm, torch, scipy, sklearn [file:2]

---

## Reproducibility

Trained on 605,938 labeled ETH samples [file:2]. Final model uses 2.5× median best iteration (≈5,258 rounds) [file:2].

