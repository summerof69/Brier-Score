# Brier-Score

## Choosing a model with the Brier score

A hands-on notebook that uses the Brier score to decide which of three purchase propensity models gives probabilities you can trust. It runs on the same synthetic BrightCart dataset as the Purchase Propensity Model series.

## Read the blog

- **The Brier score explained:** https://www.harendrasahu.com/blog/model-eval-2-the-brier-score-explained
- Related: [Calibration with Platt scaling and isotonic regression](https://www.harendrasahu.com/blog/model-eval-1-calibration-platt-scaling-and-isotonic-regression)
- Full propensity model code: https://github.com/summerof69/purchase-propensity-model

## What the notebook does

| Section | Step |
| --- | --- |
| 0 | Setup: install libraries, write `config.py` and `utils.py` |
| 1 | Rebuild the dataset: simulate a messy clickstream, clean it, build features and the 7-day purchase target |
| 2 | Train three models on a time-based split: logistic regression, random forest, LightGBM |
| 3 | Compute the Brier score by hand, then check it against scikit-learn |
| 4 | Compute the **baseline** Brier score: always predict the average purchase rate |
| 5 | Compare the raw models with the baseline using the **Brier Skill Score** |
| 6 | Split the Brier score into calibration error, refinement and uncertainty, and draw a reliability plot |
| 7 | Calibrate all three models (Platt scaling and isotonic regression) and compare again |
| 8 | Pick the model: lowest Brier score among those that beat the baseline |

## The key takeaway

All three models rank customers about equally well (ROC-AUC ≈ 0.80), but their raw probabilities are **worse than a model that knows nothing**:

| Model (raw) | Brier score on test set |
| --- | --- |
| Baseline: always predict the average rate | 0.0232 |
| Logistic regression | 0.1467 |
| Random forest | 0.0887 |
| LightGBM | 0.0294 |

The class weights used to handle the 2% purchase rate inflate the probabilities. After calibration, LightGBM drops to about **0.0198**, roughly 15% better than the baseline. The notebook shows the calibrated results for all three models and picks the winner from the data.

Lessons:
- Never read a Brier score on its own. Compare it with the baseline `base_rate × (1 − base_rate)`.
- A good AUC does not mean good probabilities.
- Compare models after calibration.

## How to run

**Google Colab (easiest)**
1. Open `Brier_Score_Model_Comparison.ipynb` in Colab.
2. Choose *Runtime > Run all*. It takes a few minutes, most of it spent simulating the data.

**Locally**
```bash
pip install pandas numpy scikit-learn lightgbm pyarrow matplotlib shap jupyter
jupyter notebook Brier_Score_Model_Comparison.ipynb
```
Then run all cells from top to bottom.

The notebook creates these folders next to itself: `src/` (helper scripts), `data/` (generated data), `models/` (trained models), `outputs/` (predictions and tables) and `images/` (charts).

## About the data

All data is **synthetic**, generated for a fictional retailer called BrightCart with a fixed random seed, so every run gives the same numbers. No real customer data is used.

## Author

Harendra Sahu · [harendrasahu.com](https://www.harendrasahu.com)
