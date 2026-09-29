# Heavy Machinery Price Prediction

Predicting the resale price of used heavy equipment (excavators, wheel loaders, dozers, etc.) from the *Heavy Equipment Selling Price Prediction Challenge* dataset. The pipeline does class-wise EDA, target-independent feature engineering, and a weighted ensemble of **LightGBM + XGBoost + CatBoost** trained on `log1p(price)`.

| | |
|---|---|
| **Task** | Regression: predict `TargetValue` (sale price in USD) |
| **Metric** | RMSLE (RMSE on `log1p(TargetValue)`), lower is better |
| **Best validation RMSLE** | **0.19144** (weighted ensemble) |
| **Tech** | Python, pandas, NumPy, scikit-learn, LightGBM, XGBoost, CatBoost, Matplotlib, Seaborn |

---

## Repository structure

```
Heavy_Machinery/
├── Heavy_Machinery_Price_Prediction.ipynb   # Full pipeline: EDA → features → models → submission
├── data/
│   ├── train.csv                # 138,701 rows × 50 columns (includes TargetValue)
│   ├── test.csv                 # 15,000 rows × 49 columns
│   ├── sample_submission.csv    # Submission format (TransactionID, TargetValue)
│   └── metadata.csv             # Description of every column
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

Columns are grouped into seven "classes" in the notebook so each family can be explored and preprocessed on its own:

| Class | Focus | Example columns |
|---|---|---|
| 1 | Transaction & business context | `TransactionID`, `TargetValue`, `TransactionDate`, `RegionCode`, `DataOriginCode`, `VendorPartnerID` |
| 2 | Asset identity & work history | `AssetID`, `ProductConfigID`, `ManufactureYear`, `OperationalHoursMeter`, `UtilizationTier` |
| 3 | Equipment classification & taxonomy | `Spec_*`, `Inventory*`, `AssetScaleFactor`, `FunctionalClassification` |
| 4 | Engine, power & core systems | `DrivetrainType`, `col1`, `col5`, `col9`, `col10`, `col14`, ... |
| 5 | Arms, attachments & working tools | `Forks`, `col4`, `col6`, `col7`, `col12`, ... |
| 6 | Cabin, steering & ground interface | `CabinType`, `col3`, `col8`, `col11`, ... |
| 7 | Cross-column engineered features | built in the notebook |

See [`data/metadata.csv`](data/metadata.csv) for the description of each variable.

## Approach

1. **EDA (class-wise)**: The target is strongly right-skewed, so `log1p` is applied before modelling. Sales volume grows sharply from about 2008, and sales are concentrated in a few states (Florida, Texas, California).
2. **Unified dataset**: Train and test are concatenated for preprocessing so categories stay consistent. The test target is masked (NaN).
3. **Class-wise preprocessing and feature engineering**
   - Date decomposition (year, month, quarter, day-of-week, day-of-year, month-end flag, cyclical encodings)
   - Cleaning invalid `ManufactureYear` sentinels; machine age; hours-meter cleaning (`0` treated as missing) and log-hours
   - Ordinal encoding of `AssetScaleFactor` and `UtilizationTier`
   - Regex parsing of the alphanumeric model code (e.g. `950FII`, `320CL`) into structured features
   - "Equipment richness" proxies (e.g. count of "None or Unspecified" fields)
   - Cross-column features and frequency encoding of high-cardinality columns (train+test counts, leak-free)
4. **Missing values are informative**: Instead of median imputation, ordinals get a `-1` sentinel category and continuous columns keep their `NaN`, so "never recorded" stays a signal the tree models can use.
5. **Modelling**: 80/20 train/validation split, a baseline and then a tuned version of each of LightGBM, XGBoost and CatBoost (several hand-picked hyper-parameter configs, with early stopping picking the number of rounds).
6. **Ensemble**: Inverse-error weighted average, `weight = (1/RMSLE) / Σ(1/RMSLE)`. Predictions are converted back with `expm1` and clipped to the observed training price range.

## Results (validation RMSLE, lower is better)

| Model | Baseline | Tuned | Improvement |
|---|---|---|---|
| LightGBM | 0.19882 | **0.19398** | 0.00483 |
| XGBoost | 0.20458 | 0.19429 | 0.01028 |
| CatBoost | 0.21102 | 0.19992 | 0.01110 |
| **Weighted ensemble** | | **0.19144** | |

Ensemble weights: LightGBM 0.337 · XGBoost 0.336 · CatBoost 0.327

## How to run

```bash
git clone https://github.com/SULAK567/Heavy_Machinery.git
cd Heavy_Machinery
pip install -r requirements.txt
jupyter notebook Heavy_Machinery_Price_Prediction.ipynb
```

> **Note on the data path:** the notebook was written on Kaggle, so `DATA_DIR` in the data-loading cell points to  
> `/kaggle/input/competitions/heavy-equipment-selling-price-prediction-challenge`.  
> To run it locally, change that line to:
> ```python
> DATA_DIR = "data"
> ```

Running the full pipeline (including hyper-parameter tuning) took about 2 hours on Kaggle. The notebook has a timer to guard against long CatBoost runs. The output is `submission.csv` (15,000 rows: `TransactionID`, `TargetValue`).

## Author

[GitHub @SULAK567](https://github.com/SULAK567)
