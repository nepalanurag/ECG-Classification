import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout

st.set_page_config(page_title="ECG Classification with CNN", layout="wide")
st.title("ECG Heartbeat Classification using CNN")

st.markdown("""
This web app demonstrates the process and findings of classifying ECG heartbeats using a Convolutional Neural Network (CNN). The model is trained to distinguish between normal and abnormal heartbeats using the MIT-BIH dataset.
""")

@st.cache_data

def load_data():
    # Update the path to your local CSV files if needed
    df = pd.read_csv("mitbih_test.csv", header=None)
    return df

def preprocess_data(df, binary=False):
    df = df.fillna(0)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', normalize=True):
    fig, ax = plt.subplots(figsize=(6, 4))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)
    st.pyplot(fig)

def plot_roc_curve(fpr, tpr):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='orange', label='ROC')
    ax.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend()
    st.pyplot(fig)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("",["Project Overview", "CNN Model", "ANN Model", "KNN Model"])
    df = load_data()

    if page == "Project Overview":
        st.header("Project Overview")
        st.write("""
        - The dataset contains ECG heartbeat signals labeled as different heartbeat types.
        - The goal is to classify heartbeats as normal or abnormal using a CNN.
        - The project explores both multiclass and binary classification.
        """)
        st.write("Sample data:")
        st.dataframe(df.head())
        st.write("Class distribution:")
        st.bar_chart(df[187].value_counts())

        # Plot a sample ECG signal
        st.subheader("Example ECG Signal")
        sample_idx = 0
        ecg_signal = df.iloc[sample_idx, :-1].values
        fig1, ax1 = plt.subplots(figsize=(10, 3))
        ax1.plot(ecg_signal, color='blue')
        ax1.set_title(f'ECG Signal Sample #{sample_idx} (Label: {int(df.iloc[sample_idx, -1])})')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Amplitude')
        st.pyplot(fig1)

        # Plot one ECG sample from each class with labels
        st.subheader("ECG Signal Example for Each Class")
        label = ["Normal Beat (N:0)",
                 "Supraventricular (S:1)",
                 "Ventricular (V:2)",
                 "Fusion (F:3)",
                 "Unknown (Q:4)"]
        fig2, ax2 = plt.subplots(5, 1, sharex=True, sharey=True, figsize=(15, 20))
        for i, row in enumerate(ax2):
            sample = (df[df[187] == i].iloc[0])[:-1]
            row.plot(sample, label=f"Sample #{int(df[df[187] == i].index[0])}: {label[i]}")            
            row.legend(fontsize=12)
        fig2.suptitle('One ECG Sample from Each Class', fontsize=18)
        fig2.supxlabel('Time')
        fig2.supylabel('Amplitude')
        st.pyplot(fig2)

    elif page == "CNN Model":
        st.header("CNN Model Results")
        from tensorflow.keras.models import load_model
        import os
        model_path = 'ecg_cnn_best.h5'
        if not os.path.exists(model_path):
            st.warning("Best CNN model file not found. Please train and save the model first.")
            return
        model = load_model(model_path)
        X_test, y_test = preprocess_data(df, binary=False)
        X_test_arr = np.array(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)
        pred = model.predict(X_test_arr)
        y_pred = np.argmax(pred, axis=1)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {acc:.4f}")
        st.text(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, ["N:0", "S:1", "V:2", "F:3", "Q:4"], normalize=False)

        # User selects a test sample to view plot and diagnosis
        st.subheader("CNN: Predict and Plot a Test Sample")
        label = ["Normal Beat (N:0)",
                 "Supraventricular (S:1)",
                 "Ventricular (V:2)",
                 "Fusion (F:3)",
                 "Unknown (Q:4)"]
        sample_idx = st.number_input("Test sample index (CNN)", min_value=0, max_value=len(X_test)-1, value=0, step=1, key="cnn_test_idx")
        sample_signal = X_test.iloc[sample_idx].values
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(sample_signal, color='purple')
        ax.set_title(f'ECG Test Sample #{sample_idx}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        st.pyplot(fig)
        # Predict for this sample
        sample_input = sample_signal.reshape(1, -1, 1)
        pred_class = int(np.argmax(model.predict(sample_input), axis=1)[0])
        true_class = int(y_test.iloc[sample_idx])
        st.write(f"True Class: {label[true_class]}")
        st.write(f"Predicted Class: {label[pred_class]}")




    elif page == "ANN Model":
        st.header("ANN Model Results ")
        from tensorflow.keras.models import load_model
        import os
        model_path = 'ecg_ann_best.h5'
        if not os.path.exists(model_path):
            st.warning("Best ANN model file not found. Please train and save the model first.")
            return
        model = load_model(model_path)
        X_test, y_test = preprocess_data(df, binary=False)
        pred = model.predict(X_test)
        y_pred = np.argmax(pred, axis=1)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {acc:.4f}")
        st.text(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, ["N:0", "S:1", "V:2", "F:3", "Q:4"], normalize=False)

        # User selects a test sample to view plot and diagnosis
        st.subheader("ANN: Predict and Plot a Test Sample")
        label = ["Normal Beat (N:0)",
                 "Supraventricular (S:1)",
                 "Ventricular (V:2)",
                 "Fusion (F:3)",
                 "Unknown (Q:4)"]
        sample_idx = st.number_input("Test sample index (ANN)", min_value=0, max_value=len(X_test)-1, value=0, step=1, key="ann_test_idx")
        sample_signal = X_test.iloc[sample_idx].values
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(sample_signal, color='orange')
        ax.set_title(f'ECG Test Sample #{sample_idx}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        st.pyplot(fig)
        # Predict for this sample
        sample_input = sample_signal.reshape(1, -1)
        pred_class = int(np.argmax(model.predict(sample_input), axis=1)[0])
        true_class = int(y_test.iloc[sample_idx])
        st.write(f"True Class: {label[true_class]}")
        st.write(f"Predicted Class: {label[pred_class]}")

    elif page == "KNN Model":
        st.header("KNN Model Results")
        import joblib
        import os
        model_path = 'ecg_knn_best.joblib'
        if not os.path.exists(model_path):
            st.warning("Best KNN model file not found. Please train and save the model first.")
            return
        knn_model = joblib.load(model_path)
        X_test, y_test = preprocess_data(df, binary=False)
        y_pred = knn_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {acc:.4f}")
        st.text(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, ["N:0", "S:1", "V:2", "F:3", "Q:4"], normalize=False)

        # User selects a test sample to view plot and diagnosis
        st.subheader("KNN: Predict and Plot a Test Sample")
        label = ["Normal Beat (N:0)",
                 "Supraventricular (S:1)",
                 "Ventricular (V:2)",
                 "Fusion (F:3)",
                 "Unknown (Q:4)"]
        sample_idx = st.number_input("Test sample index (KNN)", min_value=0, max_value=len(X_test)-1, value=0, step=1, key="knn_test_idx")
        sample_signal = X_test.iloc[sample_idx].values
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(sample_signal, color='blue')
        ax.set_title(f'ECG Test Sample #{sample_idx}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        st.pyplot(fig)
        pred_class = int(knn_model.predict([sample_signal])[0])
        true_class = int(y_test.iloc[sample_idx])
        st.write(f"True Class: {label[true_class]}")
        st.write(f"Predicted Class: {label[pred_class]}")

    
if __name__ == "__main__":
    main()
