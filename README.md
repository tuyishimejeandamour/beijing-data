# Air Quality Forecasting - Kaggle Competition

A machine learning project for predicting air quality using time series forecasting techniques.

## Project Structure

```
├── final_notebook.ipynb       # Final submission notebook
├── data/
│   ├── train.csv              # Training dataset
│   ├── test.csv               # Test dataset
│   └── sample_submission.csv  # Submission format
└── experiments/
    ├── experiment1_univariate.ipynb      # Univariate models
    ├── experiment2_multivariate.ipynb    # Multivariate models
    ├── experiment3_bilstm.ipynb          # BiLSTM approach
    ├── experiment4_hybrid_ensemble.ipynb # Hybrid ensemble methods
    └── air_quality_optimized_final.ipynb # Optimized final model
```

## Getting Started

1. Place the competition data in the `data/` folder
2. Run experiments in the `experiments/` folder to explore different approaches
3. Use `final_notebook.ipynb` for the final submission

## Requirements

- Python 3.x
- Jupyter Notebook
- pandas, numpy, scikit-learn
- TensorFlow/Keras (for LSTM models)
