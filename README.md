# OKKO · Pulls 100 Intervention Classifier

**фОККУс-покус** · UCU Analytics Practicum · 2026

Interactive demo of the customer intervention classifier developed for OKKO's Pulls 100 (E0) premium fuel campaign.

## Live Demo

→ **[Open demo](https://<your-username>.github.io/<repo-name>/)**

Scan the QR code button in the top-right corner of the demo to open on mobile.

---

## What this is

A customer-level intervention recommender built on Fishka loyalty data analysis.  
For each customer it assigns one of four interventions:

| Code | Intervention |
|------|-------------|
| **A1** | Bundling offer — free Хот Кафе coffee promo tied to P100 fill |
| **A2** | Discount coupon — −3 UAH/L on Pulls 100 |
| **B1** | Educational push — P100 / E0 awareness content |
| **—**  | Do not recommend |

Assignment is based on a multi-factor decision tree:
- P100 status (active / lapsed / never)
- Average fill volume (≥30L threshold)
- Non-fuel station behaviour (≥2 pre-period transactions)
- Non-fuel spend drop after March price spike (>20%)
- Purchasing power tercile (7-component composite score)
- Coffee buyer flag

---

## Repository structure

```
├── index.html                        # Static demo (GitHub Pages) — mock data
├── okko_intervention_classifier_v3.py # Full Python classifier — requires real data
├── app.py                            # Flask API server wrapping the classifier
├── okko_visualizer_live.html         # Live HTML front-end (points to app.py)
└── README.md
```

## Running locally with real data

```bash
# 1. Install dependencies
pip install flask flask-cors pandas numpy scikit-learn openpyxl

# 2. Place data files in the expected folders:
#    test_data/   ← transaction CSVs
#    data/        ← product_table_fuel.xlsx, personal_info.csv

# 3. Start the API server
python app.py

# 4. Open okko_visualizer_live.html in your browser
```

The server starts on `http://localhost:5050`.  
The static `index.html` demo works without any server — open it directly in a browser.

---

*OKKO Analytics · Pulls 100 Campaign · UCU IT & Business Analytics*
