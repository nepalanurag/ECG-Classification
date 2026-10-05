# ECG Heartbeat Classification

This project demonstrates the classification of ECG (Electrocardiogram) heartbeats using machine learning models, including Convolutional Neural Networks (CNN), Artificial Neural Networks (ANN), and K-Nearest Neighbors (KNN). The project features a Streamlit web app for interactive exploration, visualization, and model comparison.

## Live demo

Try it in your browser: https://ecg-classification-web.vercel.app/

The full demo package (evaluation, model conversion, and the site source) lives in [`web/`](web/).

## Evaluation

The headline numbers are the honest ones: 5-fold GroupKFold grouped by patient record (inter-patient protocol, 109,406 beats reconstructed from the 48 MIT-BIH records) gives **pooled accuracy 0.87** (per-fold 0.90, 0.93, 0.79, 0.79, 0.93), with per-class F1 of N 0.93, V 0.58, Q 0.68, and S/F near zero — the rare classes are the hard part once patient leakage is removed. Full table with bootstrap CIs in [`dashboard-data/honest_evaluation.json`](dashboard-data/honest_evaluation.json) and [`web/REPORT.md`](web/REPORT.md).

The notebook figures (CNN 98.7% 5-class, binary 99.5%) are reported on a random 70/30 split, which is optimistic because beats from the same patient land on both sides of the split. They are kept as the in-split reference; the grouped-CV number above is the one to quote.

## Features

- **Data Preprocessing:** Handles the MIT-BIH Arrhythmia dataset, including binary/multiclass label conversion. (The notebooks compute a SMOTE-balanced copy of the training set for reference, but all models train on the original unbalanced data.)
- **Model Training:** Trains and evaluates CNN, ANN, and KNN models for heartbeat classification.
- **Model Selection:** Automatically saves and uses only the best-performing model for each algorithm.
- **Interactive Web App:**
  - Explore the dataset and class distribution
  - Visualize ECG signals for each class
  - Test and visualize predictions for each model (CNN, ANN, KNN)
  - Compare model performance side-by-side

## Project Structure

- `ecg_app.py`: Main Streamlit app for data exploration, prediction, and comparison
- `ecg-cnn.ipynb`: Jupyter notebook for CNN model training and evaluation
- `ecg-classification.ipynb`: Jupyter notebook for ANN and KNN model training and evaluation
- `requirements.txt`: List of required Python packages

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nepalanurag/ECG-Classification.git
   cd ECG-Classification
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the MIT-BIH dataset:**

   - Place `mitbih_train.csv` and `mitbih_test.csv` in the project root directory.
   - These are the heartbeat-level MIT-BIH CSVs (one row per beat: 187 ECG samples plus a label, from the MIT-BIH Arrhythmia Database via PhysioNet). `mitbih_test.csv` (21,892 beats) is already in the repo; `mitbih_train.csv` is not, so grab the matching heartbeat-level CSV if you want to retrain.

4. **Train the models:**

   - Run the Jupyter notebooks (`ecg-cnn.ipynb` and `ecg-classification.ipynb`) to train and save the best models.
   - Headless equivalent: `jupyter nbconvert --to notebook --execute ecg-cnn.ipynb --output ecg-cnn.executed.ipynb` (needs `mitbih_train.csv` next to the notebook; the from-scratch binary CNN section trains in the same run and saves `ecg_cnn_binary_scratch_best.h5`).

5. **Launch the Streamlit app:**
   ```bash
   streamlit run ecg_app.py
   ```

## Usage

- Use the sidebar to navigate between:
  - Project Overview
  - CNN Model
  - ANN Model
  - KNN Model
  - Model Comparison
- Select test samples to visualize ECG signals and see model predictions.
- Compare the accuracy of all models on the same test set.

## Requirements

See `requirements.txt` for all dependencies. Main libraries:

- pandas, numpy, matplotlib, seaborn
- scikit-learn, imbalanced-learn
- tensorflow, joblib
- streamlit

## Acknowledgements

- MIT-BIH Arrhythmia Database: https://www.physionet.org/content/mitdb/1.0.0/

## License

This project is licensed under the MIT License.
