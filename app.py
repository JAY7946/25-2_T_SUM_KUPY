import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. 설정 및 데이터 로드
# ==============================================================================
st.set_page_config(page_title="학생 공부 진단", layout="wide")

@st.cache_data
def load_resources():
    # 모델 및 전처리기 로드
    try:
        model = joblib.load('kmeans_model.pkl')
        preprocess = joblib.load('preprocessor.pkl')
    except:
        st.error("모델 파일(pkl)이 없습니다.")
        return None, None, None

    # 비교 분석을 위한 원본 데이터 로드 (엑셀 또는 CSV)
    # GitHub에 파일이 올라가 있어야 합니다.
    try:
        # 엑셀 파일 우선 시도
        df = pd.read_excel('student_habits_performance.xlsx')
    except:
        try:
            # CSV 파일 시도
            df = pd.read_csv('student_habits_performance.csv')
        except:
            st.warning("원본 데이터 파일이 없어 '비교 그래프'를 그릴 수 없습니다. (student_habits_performance.xlsx 필요)")
            df = pd.DataFrame() # 빈 데이터프레임
            
    return model, preprocess, df

model, preprocess, df_ref = load_resources()

# ==============================================================================
# 2. UI 구성
# ==============================================================================
st.title("🎓 학생 공부 효율 & 습관 진단기")
st.markdown("""
자신의 생활 습관을 입력하면 **AI가 분석한 유형**과,  
다른 학생들과 비교했을 때 **나의 위치**를 알려드립니다.
""")

st.divider()

# 입력 폼
with st.sidebar:
    st.header("📝 내 습관 입력하기")
    
    age = st.number_input("나이", 18, 30, 20)
    gender = st.selectbox("성별", ["Male", "Female"])
    
    st.subheader("시간 관리")
    study_hours = st.slider("하루 공부 시간", 0.0, 15.0, 3.0, step=0.5)
    sleep_hours = st.slider("하루 수면 시간", 0.0, 12.0, 7.0, step=0.5)
    social_media = st.slider("SNS 사용 시간", 0.0, 10.0, 2.0, step=0.5)
    netflix = st.slider("OTT(넷플릭스 등) 시청", 0.0, 10.0, 1.0, step=0.5)
    
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
    if model and preprocess:
        try:
            input_processed = preprocess.transform(input_data)
            cluster = model.predict(input_processed)[0]
        except Exception as e:
            st.error(f"예측 중 오류 발생: {e}")
            cluster = -1
    else:
        cluster = -1

    # ---------------------------
    # (2) 결과 화면 구성
    # ---------------------------
    col_res1, col_res2 = st.columns([1, 1.5])
    
    with col_res1:
        st.subheader("🔍 AI 분석 결과")
        
        # 클러스터 결과 표시
        if cluster == 1:  # 우등생 그룹 (데이터 분석 결과 기반)
            st.success("🎉 **'고효율 우등생' 유형**")
            st.write("학업 성취도와 생활 밸런스가 아주 좋습니다!")
        elif cluster == 0:
            st.warning("⚠️ **'습관 개선 필요' 유형**")
            st.write("학습 시간을 조금 늘리고 생활 패턴을 잡아보면 어떨까요?")
        else:
            st.info("데이터 분석 준비 중입니다.")

        # ★ [추가 기능] 규칙 기반 상세 피드백 (공부 시간 외 요소 칭찬하기)
        st.markdown("---")
        st.caption("💡 **상세 피드백**")
        
        good_points = []
        bad_points = []
        
        # 멘탈
        if mental_health >= 7: good_points.append("멘탈 관리를 아주 잘하고 계시네요! 긍정적인 마인드가 큰 무기입니다.")
        elif mental_health <= 3: bad_points.append("스트레스가 많아 보입니다. 잠시 휴식이 필요할 수 있어요.")
        
        # 수면
        if 6 <= sleep_hours <= 8: good_points.append("수면 시간이 6~8시간으로 아주 이상적입니다.")
        elif sleep_hours < 5: bad_points.append("수면 부족은 집중력을 떨어뜨려요. 잠을 좀 더 주무세요.")
        
        # SNS
        if social_media <= 2: good_points.append("SNS 사용을 아주 잘 절제하고 계십니다.")
        elif social_media >= 4: bad_points.append("SNS 시간이 다소 깁니다. 하루 30분만 줄여볼까요?")

        # 출력
        if good_points:
            for p in good_points: st.markdown(f"- ✅ {p}")
        else:
            st.write("- 특별히 눈에 띄는 장점이 아직 없네요. 작은 습관부터 만들어봐요!")
            
        if bad_points:
            st.markdown("")
            for p in bad_points: st.markdown(f"- ❗ {p}")

    with col_res2:
        st.subheader("📊 남들과 비교하기 (나의 위치)")
        
        if not df_ref.empty:
            # 비교할 항목 선택 탭
            tab1, tab2, tab3 = st.tabs(["공부 시간", "수면 시간", "SNS 시간"])
            
            # 그래프 그리는 함수
            def plot_distribution(column, user_value, title):
                fig, ax = plt.subplots(figsize=(6, 3))
                # 전체 분포 (히스토그램 + KDE)
                sns.histplot(df_ref[column], kde=True, ax=ax, color='skyblue', stat='density')
                # 사용자 위치 (빨간 점선)
                ax.axvline(user_value, color='red', linestyle='--', linewidth=2, label='Me')
                ax.legend()
                ax.set_title(title, fontsize=12)
                ax.set_xlabel("")
                ax.set_ylabel("Density")
                st.pyplot(fig)
                
                # 상위 % 계산
                percentile = (df_ref[column] < user_value).mean() * 100
                st.caption(f"당신은 전체 학생 중 **상위 {100 - percentile:.1f}%** (하위 {percentile:.1f}%)에 해당합니다.")

            with tab1:
                plot_distribution('study_hours_per_day', study_hours, "Study Hours Distribution")
            with tab2:
                plot_distribution('sleep_hours', sleep_hours, "Sleep Hours Distribution")
            with tab3:
                plot_distribution('social_media_hours', social_media, "Social Media Hours Distribution")
        
        else:
            st.info("비교할 원본 데이터(xlsx)가 없어 그래프를 표시할 수 없습니다.")
