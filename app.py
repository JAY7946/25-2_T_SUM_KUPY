import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 0. 핵심 설정 (학습 코드와 동일한 함수 정의 필수!)
# ==============================================================================
# ★ 이 부분이 없으면 joblib.load 할 때 에러가 납니다.
SNS_WEIGHT = 3.0
def apply_weight(x):
    return x * SNS_WEIGHT

# ==============================================================================
# 1. 설정 및 데이터 로드
# ==============================================================================
st.set_page_config(page_title="학생 공부 진단", layout="wide")

@st.cache_data
def load_resources():
    # 1. 모델 및 전처리기 로드
    try:
        model = joblib.load('kmeans_model.pkl')
        preprocess = joblib.load('preprocessor.pkl')
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        st.warning("⚠️ 학습 코드에서 생성된 'kmeans_model.pkl'과 'preprocessor.pkl' 파일이 필요합니다.")
        return None, None, None

    # 2. 비교 분석을 위한 원본 데이터 로드
    df = pd.DataFrame()
    try:
        # 엑셀 읽기 시도 (openpyxl 필요)
        df = pd.read_excel('student_habits_performance.xlsx', engine='openpyxl')
    except:
        try:
            # CSV 읽기 시도
            df = pd.read_csv('student_habits_performance.csv')
        except:
            pass # 파일이 없으면 그래프 기능만 비활성화
            
    return model, preprocess, df

model, preprocess, df_ref = load_resources()

# ==============================================================================
# 2. UI 구성 (사용자 입력)
# ==============================================================================
st.title("🎓 학생 공부 효율 & 습관 진단기")
st.markdown("""
**"SNS 사용 시간"**이 학습 유형에 큰 영향을 미치도록 설계된 AI 모델입니다.  
나의 생활 습관을 입력하고 **학습 유형**과 **전체 학생 중 나의 위치**를 확인해보세요.
""")

st.divider()

with st.sidebar:
    st.header("📝 내 습관 입력하기")
    
    age = st.number_input("나이", 18, 30, 20)
    gender = st.selectbox("성별", ["Male", "Female"])
    
    st.subheader("시간 관리 (중요)")
    study_hours = st.slider("하루 공부 시간 (시간)", 0.0, 15.0, 3.0, step=0.5)
    # SNS 가중치가 높으므로 강조 표시
    social_media = st.slider("📱 SNS 사용 시간 (시간)", 0.0, 10.0, 2.0, step=0.5, help="이 항목은 결과에 큰 영향을 줍니다!")
    sleep_hours = st.slider("하루 수면 시간 (시간)", 0.0, 12.0, 7.0, step=0.5)
    netflix = st.slider("OTT(넷플릭스) 시청 시간", 0.0, 10.0, 1.0, step=0.5)
    
    st.subheader("생활 및 멘탈")
    attendance = st.slider("출석률 (%)", 0, 100, 90)
    mental_health = st.slider("멘탈/기분 점수 (1-10)", 1, 10, 5)
    exam_score = st.number_input("직전 시험 점수", 0, 100, 70)
    
    exercise = st.selectbox("운동 빈도 (주당)", [0, 1, 2, 3, 4, 5, 6, 7])
    part_time = st.selectbox("아르바이트 여부", ["Yes", "No"])
    diet = st.selectbox("식습관 품질", ["Good", "Average", "Poor"])
    internet = st.selectbox("인터넷 환경", ["Good", "Average", "Poor"])
    extra = st.selectbox("동아리/대외활동 여부", ["Yes", "No"])

# 입력 데이터 DataFrame 변환
input_data = pd.DataFrame({
    'age': [age], 'gender': [gender], 'study_hours_per_day': [study_hours],
    'social_media_hours': [social_media], 'netflix_hours': [netflix],
    'part_time_job': [part_time], 'attendance_percentage': [attendance],
    'sleep_hours': [sleep_hours], 'diet_quality': [diet],
    'exercise_frequency': [exercise], 'internet_quality': [internet],
    'mental_health_rating': [mental_health], 'extracurricular_participation': [extra],
    'exam_score': [exam_score] 
})

# ==============================================================================
# 3. 진단 및 결과 출력
# ==============================================================================
if st.button("🚀 진단 결과 확인하기", use_container_width=True):
    
    # ---------------------------
    # (1) AI 클러스터링 예측
    # ---------------------------
    cluster = -1
    if model and preprocess:
        try:
            # 전처리기에 apply_weight 함수가 포함되어 있으므로 자동 적용됨
            input_processed = preprocess.transform(input_data)
            cluster = model.predict(input_processed)[0]
        except Exception as e:
            st.error(f"진단 중 오류 발생: {e}")

    # ---------------------------
    # (2) 화면 레이아웃
    # ---------------------------
    col_res1, col_res2 = st.columns([1, 1.5], gap="large")
    
    # [왼쪽] AI 분석 결과
    with col_res1:
        st.subheader("🔍 AI 분석 결과")
        
        # 클러스터 해석 (학습 결과에 따라 0, 1 의미가 다를 수 있음. 일반적인 경향성 반영)
        # Cluster 1: 우등생 (High Study, Low SNS) / Cluster 0: 개선 필요 (Low Study, High SNS)
        # 만약 결과가 반대로 나오면 이 숫자를 스왑해주세요.
        if cluster == 1:  
            st.success("🎉 **'고효율 우등생' 유형**")
            st.write("공부 시간과 SNS 사용의 균형이 아주 훌륭합니다!")
        elif cluster == 0:
            st.error("⚠️ **'생활 습관 개선 필요' 유형**")
            st.write("SNS 사용 시간이 공부 효율을 방해하고 있을 수 있습니다.")
        else:
            st.info("데이터 분석 준비 중입니다.")

        st.markdown("---")
        st.caption("💡 **맞춤형 피드백**")
        
        # 규칙 기반 피드백 생성
        feedbacks = []
        
        # SNS 피드백 (가장 중요)
        if social_media >= 3.0:
            feedbacks.append(f"❗ **SNS 사용이 많아요({social_media}시간).** 하루 1시간만 줄여도 등급이 바뀔 수 있어요.")
        elif social_media <= 1.5:
            feedbacks.append(f"✅ **SNS 관리가 완벽해요!** 집중력을 유지하는 비결이시네요.")
            
        # 멘탈 피드백
        if mental_health >= 7:
            feedbacks.append("✅ **멘탈 관리가 훌륭합니다.** 긍정적인 마음이 성적 향상의 열쇠입니다.")
        elif mental_health <= 4:
            feedbacks.append("❗ **스트레스가 높아 보입니다.** 잠시 산책이나 휴식이 필요해요.")

        # 수면 피드백
        if sleep_hours < 5:
            feedbacks.append("❗ **수면이 부족합니다.** 잠을 줄이는 건 장기적으로 손해예요.")
        elif 6 <= sleep_hours <= 8:
            feedbacks.append("✅ **수면 시간이 아주 이상적입니다.**")

        # 출력
        for fb in feedbacks:
            st.markdown(fb)

    # [오른쪽] 남들과 비교하기 그래프
    with col_res2:
        st.subheader("📊 전체 학생 중 나의 위치")
        
        if not df_ref.empty:
            # 탭으로 구분해서 보여주기
            tab1, tab2, tab3 = st.tabs(["SNS 시간 (핵심)", "공부 시간", "수면 시간"])
            
            def plot_ranking(col_name, user_val, title, invert=False):
                """히스토그램과 나의 위치를 그려주는 함수"""
                fig, ax = plt.subplots(figsize=(8, 4))
                
                # 전체 분포 그리기
                sns.histplot(df_ref[col_name], kde=True, ax=ax, color='#4A90E2', alpha=0.6)
                
                # 내 위치 표시 (빨간 점선)
                ax.axvline(user_val, color='red', linestyle='--', linewidth=2, label='Me')
                
                # 상위 % 계산
                # 공부/수면은 많을수록 상위(False), SNS는 적을수록 상위(True)
                percentile = (df_ref[col_name] < user_val).mean() * 100
                if invert: # SNS 같은 경우 (적을수록 좋음)
                    rank_text = f"상위 {100 - percentile:.1f}% (적게 쓰는 편)" if percentile < 50 else f"하위 {percentile:.1f}% (많이 쓰는 편)"
                else: # 공부 같은 경우 (많을수록 좋음)
                    rank_text = f"상위 {100 - percentile:.1f}%"
                
                ax.set_title(f"{title} (나: {user_val}시간 - {rank_text})", fontsize=12, fontweight='bold')
                ax.set_xlabel("Hours")
                ax.set_ylabel("Count")
                ax.legend()
                st.pyplot(fig)

            with tab1:
                st.info("📉 SNS는 **왼쪽(시간이 적음)**에 있을수록 좋습니다.")
                plot_ranking('social_media_hours', social_media, "Social Media Hours", invert=True)
                
            with tab2:
                st.info("📈 공부 시간은 **오른쪽(시간이 많음)**에 있을수록 상위권입니다.")
                plot_ranking('study_hours_per_day', study_hours, "Study Hours", invert=False)
                
            with tab3:
                plot_ranking('sleep_hours', sleep_hours, "Sleep Hours", invert=False)
        else:
            st.warning("⚠️ 비교용 데이터(xlsx)가 없어 그래프를 그릴 수 없습니다.")

