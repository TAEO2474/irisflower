# iris_dashboard.py
import streamlit as st
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
import glob

# ----------------------------------------
# 1. 모델 로드 (가장 최근 .keras 파일)
# ----------------------------------------
MODEL_DIR = "saved_models"

def get_latest_model():
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]
    if not models:
        return None
    models.sort(reverse=True)
    return os.path.join(MODEL_DIR, models[0])

latest_model_path = get_latest_model()
if latest_model_path:
    from tensorflow.keras.layers import Dropout  # Dropout 포함한 모델 로딩
    model = load_model(latest_model_path)
else:
    st.error("❌ 저장된 모델(.keras)을 찾을 수 없습니다. 먼저 iris_model.py를 실행해 주세요.")
    st.stop()

# ----------------------------------------
# 2. 스케일러 로드
# ----------------------------------------
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("❌ scaler.pkl 파일이 없습니다. iris_model.py에서 학습 후 저장해야 합니다.")
    st.stop()

# ----------------------------------------
# 3. Streamlit UI 구성
# ----------------------------------------
st.set_page_config(page_title="아이리스 꽃 분류기", page_icon="🌸", layout="centered")
st.title("🌸 아이리스 꽃 분류기")
st.markdown("꽃받침과 꽃잎의 길이와 너비를 기준으로 꽃의 종류를 예측합니다.")

# 슬라이더 입력 UI
sepal_length = st.slider("1️⃣ 꽃받침 길이 (Sepal Length)", 4.0, 8.0, 5.8)
sepal_width  = st.slider("2️⃣ 꽃받침 너비 (Sepal Width)", 2.0, 4.5, 3.0)
petal_length = st.slider("3️⃣ 꽃잎 길이 (Petal Length)", 1.0, 7.0, 4.35)
petal_width  = st.slider("4️⃣ 꽃잎 너비 (Petal Width)", 0.1, 2.5, 1.3)

# 입력 데이터를 배열로 정리
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
input_scaled = scaler.transform(input_data)

# ----------------------------------------
# 4. 예측
# ----------------------------------------
if st.button("🌼 예측 실행"):
    pred = model.predict(input_scaled)
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    predicted_class = class_names[np.argmax(pred)]

    # 결과 출력
    st.subheader("📢 예측 결과:")
    st.success(f"이 꽃은 **{predicted_class}** 로 예측됩니다.")

    # 확률 출력
    st.markdown("### 📊 클래스별 예측 확률:")
    for i in range(3):
        st.write(f"- {class_names[i]}: **{pred[0][i]:.2%}**")
