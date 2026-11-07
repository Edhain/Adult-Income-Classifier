import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

st.set_page_config(page_title="Income Classifier Evaluation", page_icon="📊", layout="wide")

st.title("📊 Adult Income Classification Model Evaluator")
st.markdown("### **Submitted by:** Manas Pareek")
st.markdown("""
#### Welcome to the model evaluation dashboard! 
Upload your **test_data.csv** file to see how different machine learning models perform.
""")

# --- Sidebar Configuration ---
st.sidebar.markdown("---")
st.sidebar.info("💡 **About this App:**\nThis dashboard evaluates 5 different ML classifiers to predict if an individual's income exceeds $50K/yr based on demographic data. Built for ML Assignment 2.")
st.sidebar.header("1. Upload Test Data")
# Requirement a: Dataset upload option[cite: 5]
uploaded_file = st.sidebar.file_uploader("Upload your test data (CSV format)", type=['csv'])

st.sidebar.header("2. Model Selection")
# Requirement b: Model selection dropdown[cite: 5]
model_dict = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "K-Nearest Neighbors": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/svm.pkl"
}
selected_model_name = st.sidebar.selectbox("Choose a classification model:", list(model_dict.keys()))

# --- Main App Logic ---
if uploaded_file is not None:
    # Load the uploaded test dataset
    test_df = pd.read_csv(uploaded_file)
    
    st.write("### Preview of Uploaded Data")
    st.dataframe(test_df.head())
    
    # Check if the required target column exists
    if 'Target' in test_df.columns:
        X_test = test_df.drop(columns=['Target'])
        y_test = test_df['Target']
        
        # Load the selected model
        model_path = model_dict[selected_model_name]
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            
            # Make predictions without retraining[cite: 5]
            y_pred = model.predict(X_test)
            
            # Handle probability predictions for AUC (SVM needs probability=True during training)
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = None
            
            # --- Requirement c: Display of evaluation metrics ---
            st.markdown("---")
            # Prominently display the selected model name in the header
            st.markdown(f"### Performance Metrics: {selected_model_name}")
            
            # 1. Calculate the metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1_val = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
            
            # 2. Create 6 evenly spaced columns for the "big numbers"
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            col1.metric("Accuracy", f"{acc:.4f}")
            if isinstance(auc, float):
                col2.metric("AUC Score", f"{auc:.4f}")
            else:
                col2.metric("AUC Score", auc)
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1_val:.4f}")
            col6.metric("MCC", f"{mcc:.4f}")
                
            st.markdown("---")
            
            # --- Requirement d: Confusion matrix or classification report[cite: 5] ---
            # tab1, tab2 = st.tabs(["📉 Confusion Matrix", "📋 Classification Report"])

            # with tab1:
            #     st.write("### Confusion Matrix")
            #     cm = confusion_matrix(y_test, y_pred)
            #     fig, ax = plt.subplots(figsize=(5, 4))
            #     sns.heatmap(cm, annot=True, fmt='d', cmap='rocket', ax=ax, cbar=False)
            #     ax.set_xlabel('Predicted Label')
            #     ax.set_ylabel('True Label')
            #     st.pyplot(fig)

            # with tab2:
            #     st.write("### Classification Report")
            #     report_dict = classification_report(y_test, y_pred, output_dict=True)
            #     report_df = pd.DataFrame(report_dict).transpose()
            #     st.dataframe(report_df.style.format(precision=4))
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.write("### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax, cbar=False)
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                st.pyplot(fig)
                
            with col_b:
                st.write("### Classification Report")
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                # st.dataframe(report_df.style.format(precision=4))
                st.table(
                    report_df.style.format({
                        'precision': '{:.3f}', 
                        'recall': '{:.3f}', 
                        'f1-score': '{:.3f}', 
                        'support': '{:.0f}' 
                    })
                )
                
        else:
            st.error(f"Could not find the model file at `{model_path}`. Ensure your 'model' folder is uploaded alongside app.py.")
    else:
        st.error("The uploaded CSV must contain the 'Target' column to evaluate the model.")
else:
    st.info("👈 Awaiting test data upload. Please use the sidebar to upload 'test_data.csv'.")