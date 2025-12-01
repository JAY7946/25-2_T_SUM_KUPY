import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import platform

# ==============================================================================
# [설정] 폰트 설정 제거 (기본 영문 폰트 사용)
# ==============================================================================
# 한글 폰트 설정을 제거하고 기본값(영문)을 사용하면 깨짐 현상이 사라집니다.
plt.rcParams['axes.unicode_minus'] = False 

# ==============================================================================
# 0. 핵심 설정 (학습 코드와 함수명/로직 완전 일치 필수)
# ==============================================================================
SNS_WEIGHT = 1.1      
STUDY_WEIGHT = 1.2    

def apply_sns_weight(x):
    """SNS 사용 시간에 가중치를 부여하는 함수"""
    return x * SNS_WEIGHT

def apply_study_weight(x):
    """공부 시간에 가중치를 부여하는 함수"""
    return x * STUDY_WEIGHT

# ==============================================================================
# 1. 설정 및 데이터 로드
# ==============================================================================
st.set_page_config(page_title="Student Study Diagnosis", layout="wide", page_icon="🎓")

@st.cache_data
def load_resources():
    model = None
    preprocess = None
    
    model_path = 'kmeans_model.pkl'
    prep_path = 'preprocessor.pkl'
    
    try:
        if os.path.exists(model_path) and os.path.exists(prep_path):
            model = joblib.load(model_path)
            preprocess = joblib.load(prep_path)
        else:
            st.warning("⚠️ Model files not found ('kmeans_model.pkl', 'preprocessor.pkl').")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

    df = pd.DataFrame()
    try:
        if os.path.exists('student_habits_performance.xlsx'):
            df = pd.read_excel('student_habits_performance.xlsx', engine='openpyxl')
        elif os.path.exists('student_habits_performance.csv'):
            df = pd.read_csv('student_habits_performance.csv')
    except Exception as e:
        pass 
            
    return model, preprocess, df

model, preprocess, df_ref = load_resources()

# ==============================================================================
# 2. UI 구성 (사용자 입력)
# ==============================================================================
st.title("🎓 Student Study Efficiency & Habit Diagnosis")
st.markdown("Enter your habits to analyze your study type.")

st.divider()

with st.sidebar:
    st.header("📝 My Habits")
    
    st.subheader("1. Basic Info")
    age = st.number_input("Age", 15, 30, 18)
    gender = st.selectbox("Gender", ["Male", "Female"])
    
    st.subheader("2. Time Management")
    study_hours = st.slider("✍️ Study Hours (per day)", 0.0, 15.0, 3.0, step=0.5)
    social_media = st.slider("📱 SNS Hours (per day)", 0.0, 10.0, 2.0, step=0.5)
    sleep_hours = st.slider("💤 Sleep Hours (per day)", 0.0, 12.0, 7.0, step=0.5)
    netflix = st.slider("🎬 Netflix/OTT Hours", 0.0, 10.0, 1.0, step=0.5)
    
    st.subheader("3. Life & Mental")
    attendance = st.slider("Attendance (%)", 0, 100, 90)
    mental_health = st.slider("Mental Health (1-10)", 1, 10, 5)
    exam_score = st.number_input("Previous Exam Score", 0, 100, 70)
    
    st.subheader("4. Environment")
    exercise = st.selectbox("Exercise (days/week)", [0, 1, 2, 3, 4, 5, 6, 7])
    part_time = st.selectbox("Part-time Job", ["Yes", "No"])
    diet = st.selectbox("Diet Quality", ["Good", "Average", "Poor"])
    internet = st.selectbox("Internet Quality", ["Good", "Average", "Poor"])
    extra = st.selectbox("Extracurricular", ["Yes", "No"])

# 입력 데이터 DataFrame 변환
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

# ==============================================================================
# 3. 진단 및 결과 출력
# ==============================================================================
if st.button("🚀 Analyze Result", use_container_width=True):
    
    # ---------------------------
    # (1) AI 클러스터링 예측
    # ---------------------------
    cluster = -1
    
    if model and preprocess:
        try:
            # 전처리 실행
            input_processed = preprocess.transform(input_data)
            cluster = model.predict(input_processed)[0]
        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.error("Model files are missing.")

    # ---------------------------
    # (2) 화면 레이아웃 및 결과 표시
    # ---------------------------
    col_res1, col_res2 = st.columns([1, 1.2], gap="large")
    
    # [왼쪽] AI 분석 결과 (텍스트는 한글 유지, 필요시 영어로 변경 가능)
    with col_res1:
        st.subheader("🔍 Analysis Result")
        
        target_cluster_good = 1  
        
        if cluster == target_cluster_good:   
            st.success("🎉 **Type: Self-Directed Learner**")
            st.write("Great balance between study and rest!")
        elif cluster != -1:
            st.warning("⚠️ **Type: Needs Improvement**")
            st.write("SNS or media usage might be hindering your potential.")
        
        st.markdown("---")
        st.caption("💡 **Feedback**")
        
        feedbacks = []
        
        if social_media > 3.0:
            feedbacks.append(f"❗ **High SNS Usage ({social_media} hrs).** Try to reduce it.")
        
        if study_hours < 2.0:
            feedbacks.append(f"❗ **Low Study Time ({study_hours} hrs).** Aim for at least 2-3 hours.")
        elif study_hours > 5.0 and social_media < 2.0:
            feedbacks.append("✅ **Perfect Study Habit.** Keep it up!")

        if sleep_hours < 5.5:
            feedbacks.append("💤 **Lack of Sleep.** Sleep affects concentration.")
        
        if mental_health <= 4:
            feedbacks.append("🍀 **Manage Stress.** Take breaks or meditate.")

        if exercise == 0:
             feedbacks.append("🏃 **Need Exercise.** Physical health aids brain function.")

        if not feedbacks:
            feedbacks.append("👌 No major bad habits detected!")

        for fb in feedbacks:
            st.markdown(fb)

    # [오른쪽] 남들과 비교하기 그래프 (영어로 변경)
    with col_res2:
        st.subheader("📊 My Position in Distribution")
        
        if not df_ref.empty:
            tab1, tab2, tab3 = st.tabs(["SNS", "Study", "Sleep"])
            
            def plot_ranking(col_name, user_val, title, invert=False):
                """히스토그램 (영문 라벨 적용)"""
                fig, ax = plt.subplots(figsize=(6, 3.5))
                
                # 전체 분포
                sns.histplot(df_ref[col_name], kde=True, ax=ax, color='#6C5CE7', alpha=0.5, edgecolor=None)
                
                # 내 위치
                ax.axvline(user_val, color='#E84393', linestyle='--', linewidth=2.5, label='Me')
                
                # 상위 % 계산
                percentile = (df_ref[col_name] < user_val).mean() * 100
                if invert: # 낮을수록 좋은 것 (SNS)
                    rank = percentile 
                    rank_text = f"Top {rank:.1f}%" if rank < 50 else f"Bottom {100-rank:.1f}%"
                else: # 높을수록 좋은 것 (공부, 수면)
                    rank = 100 - percentile
                    rank_text = f"Top {rank:.1f}%"
                
                # ★★★ [수정됨] 영어 라벨 적용 ★★★
                ax.set_title(f"{title}\n(Me: {user_val} hrs - {rank_text})", fontsize=12)
                ax.set_xlabel("Time (Hours)") # X축: Time
                ax.set_ylabel("Density")      # Y축: Density
                ax.legend()
                st.pyplot(fig)

            with tab1:
                # SNS 그래프
                plot_ranking('social_media_hours', social_media, "SNS Hours", invert=True)
                
            with tab2:
                # 공부 시간 그래프
                plot_ranking('study_hours_per_day', study_hours, "Study Hours", invert=False)
                
            with tab3:
                # 수면 시간 그래프 (요청하신 대로 Sleep으로 변경)
                plot_ranking('sleep_hours', sleep_hours, "Sleep Hours", invert=False)
        else:
            st.warning("⚠️ Reference data (student_habits_performance.xlsx) not found.")
