import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 加载训练好的SVM模型
model = joblib.load("RF_model.joblib")


# 单次预测
def predict_single(data):
    probability = model.predict_proba([data])[0, 1]  # 假设是二分类
    return probability


# 批量预测
def predict_batch(df):
    probabilities = model.predict_proba(df)[:, 1]
    return probabilities

# Streamlit UI 设置
st.set_page_config(page_title="BOT Recurrence Prediction", page_icon="🩺", layout="wide")
st.title("🩺 BOT Recurrence Prediction Tool")
st.markdown(
    """
    Welcome to the **BOT Recurrence Prediction Tool**. 
    This tool predicts the probability of BOT recurrence based on clinical input variables. 
    Please fill in the required inputs or upload a CSV file for batch predictions.
    """
)

# 侧边栏输入表单
st.sidebar.header("Input Variables")
st.sidebar.markdown("Fill in the following details:")

input_features = []


def add_input(title, options=None, is_int=False):
    if options:
        return st.sidebar.selectbox(title, options, format_func=lambda x: options[x])
    return st.sidebar.number_input(title, value=0 if is_int else 0.0, format="%d" if is_int else "%.2f")


# 定义输入变量
titles_options = {
    "Parity": {0: "No", 1: "Yes"},
    "Menopausal Status": {0: "No", 1: "Yes"},
    "Fertility-Sparing Surgery": {0: "No", 1: "Yes"},
    "Completeness of Surgery": {0: "Incomplete", 1: "Complete"},
    "Omentectomy": {0: "No", 1: "Yes"},
    "Lymphadenectomy": {0: "No", 1: "Yes"},
    "Peritoneal implantation": {0: "No", 1: "Yes"},
    "Unilateral or Bilateral": {0: "Unilateral", 1: "Bilateral"},
}
input_features.append(add_input("Age", is_int=True))
for title, options in titles_options.items():
    input_features.append(add_input(title, options=options) if options else add_input(title, is_int=True))
# 单次预测
st.subheader("🧪 Single Case Prediction")
if st.button("Predict Single Case"):
    prob = predict_single(input_features)
    st.markdown("### 🧐 **Prediction Result**")
    st.markdown(
        f"<h2 style='color:#FF6F61; text-align:center;'>**Predicted BOT recurrence Outcome Probability**: **{prob:.3f}**</h2>",
        unsafe_allow_html=True)

# 批量预测
st.subheader("📑 Batch Prediction")
st.markdown("Upload a CSV file for batch prediction. Make sure the file matches the input format.")
uploaded_file = st.file_uploader("Upload CSV File", type="csv")
if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    probabilities = predict_batch(batch_data)
    batch_data["Predicted Probability"] = probabilities

    st.markdown("### 📊 **Batch Prediction Results**")
    st.dataframe(batch_data.style.applymap(lambda x: 'background-color: yellow' if x > 0.5 else '',
                                           subset=["Predicted Probability"]))

    csv = batch_data.to_csv(index=False)
    st.download_button(label="📥 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

# 页脚
st.markdown(""" 
--- 
© Shengjing Hospital of China Medical University 
""", unsafe_allow_html=True)

