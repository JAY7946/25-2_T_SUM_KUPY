import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. 모델과 전처리기 불러오기
try:
    model = joblib.load('kmeans_model.pkl')
    preprocess = joblib.load('preprocessor.pkl')
except:
    st.error("모델 파일(kmeans_model.pkl, preprocessor.pkl)이 없습니다. 먼저 모델을 생성해주세요.")
    st.stop()

# 2. 웹페이지 제목
st.title("🎓 학생 공부 효율 진단 시스템")
st.write("본인의 생활 습관을 입력하면, 현재 어떤 유형의 학습 패턴인지 진단해 드립니다.")

# 3. 사용자 입력 받기
st.subheader("생활 습관 입력")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("나이", 18, 30, 20)
    gender = st.selectbox("성별", ["Male", "Female"])
    study_hours = st.slider("하루 공부 시간(시간)", 0.0, 15.0, 3.0)
    sleep_hours = st.slider("하루 수면 시간(시간)", 0.0, 12.0, 7.0)
    social_media = st.slider("SNS 사용 시간(시간)", 0.0, 10.0, 2.0)

with col2:
    netflix = st.slider("OTT(넷플릭스 등) 시청 시간", 0.0, 10.0, 1.0)
    attendance = st.slider("출석률(%)", 0, 100, 90)
    exam_score = st.number_input("직전 시험 점수", 0, 100, 70)
    mental_health = st.slider("멘탈/기분 점수 (1-10)", 1, 10, 5)
    exercise = st.selectbox("운동 빈도 (주당)", [0, 1, 2, 3, 4, 5, 6, 7])

part_time = st.selectbox("아르바이트 여부", ["Yes", "No"])
diet = st.selectbox("식습관 품질", ["Good", "Average", "Poor"])
internet = st.selectbox("인터넷 환경", ["Good", "Average", "Poor"])
extra = st.selectbox("동아리/대외활동 여부", ["Yes", "No"])

# 4. 데이터프레임 변환
input_data = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'study_hours_per_day': [study_hours],
    'social_media_hours': [social_media],
    'netflix_hours': [netflix],
    'part_time_job': [part_time],
    'attendance_percentage': [attendance],
    'sleep_hours': [sleep_hours],
    'diet_quality': [diet],
    'exercise_frequency': [exercise],
    'internet_quality': [internet],
    'mental_health_rating': [mental_health],
    'extracurricular_participation': [extra],
    'exam_score': [exam_score] 
})

# 5. 진단 버튼
if st.button("내 학습 유형 진단하기"):
    try:
        input_processed = preprocess.transform(input_data)
        cluster = model.predict(input_processed)[0]
        
        st.divider()
        st.subheader("진단 결과")
        
        # 클러스터 1이 우등생 그룹이라고 가정 (이전 코드 결과 기반)
        if cluster == 1: 
             st.success("🎉 **'고효율 우등생'** 유형입니다!")
             st.write("학습 밸런스가 아주 좋습니다. 지금 패턴을 유지하세요!")
        else:
             st.error("⚠️ **'학습 개선 필요'** 유형입니다.")
             st.write("공부 시간을 늘리고 SNS/OTT 시간을 줄이는 것이 좋겠습니다.")
             
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
