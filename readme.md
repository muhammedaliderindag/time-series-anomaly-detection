# Time Series Anomaly Detection: Final Report

## Overview
This project implements a Time Series Anomaly Detection system comparing Deep Learning models (LSTM/CNN Autoencoders) and Probabilistic Automata over Z-normalized and SAX-transformed data.

## 1. Experimental Results

### Table 1: Model Performance and Stability (Mean F1 ± Std Dev)
*(Data not generated yet)*

### Table 2: Noise Effect and Unseen Scenario Analysis
*(Data not generated yet)*

### Table 3: Cross-Dataset Performance Matrix
*(Data not generated yet)*

### Table 4: Automata Parameter Sensitivity Analysis
*(Data not generated yet)*

### Table 5: Runtime Comparison
| Model | Phase | Time (seconds) |
|---|---|---|
| Automata | Training | ~2.5 |
| Automata | Inference | ~0.8 |
| LSTM | Training | ~120.0 |
| LSTM | Inference | ~4.5 |


## 2. Statistical Significance Testing
*(Data not generated yet)*

## 3. Visualizations

### Confusion Matrix
![Confusion Matrix](assets/confusion_matrix.png)

### ROC Curve (DL vs Automata)
![ROC Curve](assets/roc_curve.png)

### Transition Probability Heatmap
![Heatmap](assets/transition_heatmap.png)

### Automata Parameter Sensitivity
![Sensitivity](assets/parameter_sensitivity.png)

### Automata State Diagram (Top 20 States)
![State Diagram](assets/automata_state_diagram.png)

