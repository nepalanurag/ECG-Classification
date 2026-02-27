# ECG Heartbeat Classification

This project demonstrates the classification of ECG (Electrocardiogram) heartbeats using machine learning models, including Convolutional Neural Networks (CNN), Artificial Neural Networks (ANN), and K-Nearest Neighbors (KNN). The project features a Streamlit web app for interactive exploration, visualization, and model comparison.

## Features

- **Data Preprocessing:** Handles the MIT-BIH Arrhythmia dataset, including class balancing and binary/multiclass conversion.
- **Model Training:** Trains and evaluates CNN, ANN, and KNN models for heartbeat classification.
- **Model Selection:** Automatically saves and uses only the best-performing model for each algorithm.
- **Interactive Web App:**
  - Explore the dataset and class distribution
  - Visualize ECG signals for each class
  - Test and visualize predictions for each model (CNN, ANN, KNN)
  - Compare model performance side-by-side

## Project Structure

- `ecg_app.py` — Main Streamlit app for data exploration, prediction, and comparison
- `ecg-cnn.ipynb` — Jupyter notebook for CNN model training and evaluation
- `ecg-classification.ipynb` — Jupyter notebook for ANN and KNN model training and evaluation
- `requirements.txt` — List of required Python packages

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

4. **Train the models:**

   - Run the Jupyter notebooks (`ecg-cnn.ipynb` and `ecg-classification.ipynb`) to train and save the best models.

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
