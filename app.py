import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==============================================================================
# 0. 핵심 설정 (학습 코드와 동일한 함수 정의 필수!)
# ==============================================================================
# ★ 이 부분이 없으면 joblib.load 할 때 에러가 납니다.
# 학습 때 사용한 가중치 로직과 동일해야 합니다.
SNS_WEIGHT = 1.5

def apply_weight(x):
    """SNS 사용 시간에 가중치를 부여하는 함수 (학습 파이프라인과 동일)"""
    return x * SNS_WEIGHT

# ==============================================================================
# 1. 설정 및 데이터 로드
# ==============================================================================
st.set_page_config(page_title="학생 공부 진단", layout="wide")

@st.cache_data
def load_resources():
    # 1. 모델 및 전처리기 로드
    model = None
    preprocess = None
    
    try:
        if os.path.exists('kmeans_model.pkl') and os.path.exists('preprocessor.pkl'):
            model = joblib.load('kmeans_model.pkl')
            preprocess = joblib.load('preprocessor.pkl')
        else:
            st.warning("⚠️ 학습 모델 파일('kmeans_model.pkl', 'preprocessor.pkl')을 찾을 수 없습니다.")
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None, None, None

    # 2. 비교 분석을 위한 원본 데이터 로드
    df = pd.DataFrame()
    try:
        # 엑셀 읽기 시도 (openpyxl 라이브러리 필요)
        if os.path.exists('student_habits_performance.xlsx'):
            df = pd.read_excel('student_habits_performance.xlsx', engine='openpyxl')
        # CSV 읽기 시도
        elif os.path.exists('student_habits_performance.csv'):
            df = pd.read_csv('student_habits_performance.csv')
    except Exception as e:
        pass # 파일이 없거나 읽기 실패 시 그래프 기능만 비활성화 (오류 출력 최소화)
            
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
    
    age = st.number_input("나이", 15, 30, 18)
    gender = st.selectbox("성별", ["Male", "Female"])
    
    st.subheader("시간 관리 (중요)")
    study_hours = st.slider("하루 공부 시간 (시간)", 0.0, 15.0, 3.0, step=0.5)
    # SNS 가중치가 높으므로 강조 표시 및 도움말 추가
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
            # 전처리기에 apply_weight 함수가 포함되어 있으므로 transform 시 자동 적용됨
            input_processed = preprocess.transform(input_data)
            cluster = model.predict(input_processed)[0]
        except Exception as e:
            st.error(f"진단 중 오류 발생: {e}")
            st.info("모델 학습 시 사용된 scikit-learn 버전 차이 혹은 사용자 정의 함수 문제일 수 있습니다.")

    # ---------------------------
    # (2) 화면 레이아웃 및 결과 표시
    # ---------------------------
    col_res1, col_res2 = st.columns([1, 1.5], gap="large")
    
    # [왼쪽] AI 분석 결과
    with col_res1:
        st.subheader("🔍 AI 분석 결과")
        
        # 클러스터 해석 (학습 결과에 따라 0, 1 의미가 다를 수 있음. 일반적인 경향성 반영)
        # Cluster 1: 우등생 (High Study, Low SNS) / Cluster 0: 개선 필요 (Low Study, High SNS)
        # ※ 만약 실제 실행 시 결과가 반대로 나온다면 아래 숫자를 스왑해주세요.
        if cluster == 1:   
            st.success("🎉 **'고효율 우등생' 유형**")
            st.write("공부 시간과 SNS 사용의 균형이 아주 훌륭합니다! 현재 패턴을 유지하세요.")
        elif cluster == 0:
            st.error("⚠️ **'생활 습관 개선 필요' 유형**")
            st.write("SNS 사용 시간이 공부 효율을 방해하고 있을 수 있습니다. 학습 시간을 조금 더 늘려보세요.")
        else:
            st.info("데이터 분석 준비 중입니다.")

        st.markdown("---")
        st.caption("💡 **1:1 맞춤 피드백**")
        
        # 규칙 기반 피드백 생성
        feedbacks = []
        
        # 1. SNS 피드백 (가중치 1.5배로 가장 중요)
        if social_media >= 3.0:
            feedbacks.append(f"❗ **SNS 사용이 많아요({social_media}시간).** 하루 1시간만 줄여도 학습 효율 등급이 바뀔 수 있어요.")
        elif social_media <= 1.5:
            feedbacks.append(f"✅ **SNS 관리가 완벽해요!** 디지털 디톡스를 잘 실천하고 계시네요.")
            
        # 2. 멘탈 피드백
        if mental_health >= 8:
            feedbacks.append("✅ **멘탈 관리가 훌륭합니다.** 긍정적인 마음이 성적 향상의 열쇠입니다.")
        elif mental_health <= 4:
            feedbacks.append("❗ **스트레스가 높아 보입니다.** 공부도 중요하지만 잠시 산책이나 명상이 필요해요.")

        # 3. 수면 피드백
        if sleep_hours < 5:
            feedbacks.append("❗ **수면이 너무 부족해요.** 잠을 줄이는 건 집중력 저하로 이어져 장기적으로 손해입니다.")
        elif 6 <= sleep_hours <= 8:
            feedbacks.append("✅ **수면 시간이 아주 이상적입니다.** 뇌가 기억을 정리할 시간이 충분해요.")
            
        # 4. 출석률 피드백
        if attendance < 80:
            feedbacks.append("⚠️ **학교/학원 출석률이 낮아요.** 성실함이 기본 바탕이 되어야 합니다.")

        # 5. 운동 피드백
        if exercise == 0:
             feedbacks.append("🏃‍♂️ **가벼운 운동을 시작해보세요.** 체력이 뒷받침되어야 책상에 오래 앉아있을 수 있어요.")

        # 피드백이 없으면 기본 메시지
        if not feedbacks:
            feedbacks.append("👌 전반적으로 무난한 습관을 가지고 계십니다.")

        # 출력
        for fb in feedbacks:
            st.markdown(fb)

    # [오른쪽] 남들과 비교하기 그래프
    with col_res2:
        st.subheader("📊 전체 학생 중 나의 위치")
        
        if not df_ref.empty:
            # 탭으로 구분해서 보여주기
            tab1, tab2, tab3 = st.tabs(["SNS 시간", "공부 시간", "수면 시간"])
            
            def plot_ranking(col_name, user_val, title, invert=False):
                """히스토그램과 나의 위치를 그려주는 함수"""
                fig, ax = plt.subplots(figsize=(8, 4))
                
                # 전체 분포 그리기 (Seaborn)
                # 한글 폰트 깨짐 방지를 위해 영문 라벨 사용 권장 혹은 별도 폰트 설정 필요
                sns.histplot(df_ref[col_name], kde=True, ax=ax, color='#4A90E2', alpha=0.6)
                
                # 내 위치 표시 (빨간 점선)
                ax.axvline(user_val, color='red', linestyle='--', linewidth=2, label='Me')
                
                # 상위 % 계산
                # invert=True인 경우(예: SNS) 낮을수록 상위, invert=False인 경우(예: 공부) 높을수록 상위
                percentile = (df_ref[col_name] < user_val).mean() * 100
                
                if invert: # SNS, 넷플릭스 등 (적을수록 좋음)
                    # 내 값이 작을수록 percentile은 작아짐 -> 상위권
                    # 예: 내가 1시간(하위 10%), 남들은 5시간 -> 나는 상위 10% 생활습관
                    rank_perc = percentile 
                    rank_text = f"상위 {rank_perc:.1f}% (적게 쓰는 편)" if rank_perc < 50 else f"하위 {100-rank_perc:.1f}% (많이 쓰는 편)"
                else: # 공부, 수면 등 (많을수록 좋음)
                    # 내 값이 클수록 percentile은 커짐 -> 상위권
                    # 예: 내가 10시간(상위 90% 지점) -> 나는 상위 10% 공부량
                    rank_perc = 100 - percentile
                    rank_text = f"상위 {rank_perc:.1f}%"
                
                # 그래프 제목 및 라벨
                ax.set_title(f"{title} (Me: {user_val}h - {rank_text})", fontsize=12, fontweight='bold')
                ax.set_xlabel("Hours")
                ax.set_ylabel("Number of Students")
                ax.legend()
                
                st.pyplot(fig)

            with tab1:
                st.info("📉 SNS 사용시간")
                plot_ranking('social_media_hours', social_media, "Social Media Hours", invert=True)
                
            with tab2:
                st.info("📈 공부 시간")
                plot_ranking('study_hours_per_day', study_hours, "Study Hours", invert=False)
                
            with tab3:
                st.info("💤 수면 시간")
                plot_ranking('sleep_hours', sleep_hours, "Sleep Hours", invert=False)
        else:
            st.warning("⚠️ 비교용 데이터(xlsx/csv)가 없어 그래프를 그릴 수 없습니다. 폴더에 데이터 파일을 넣어주세요.")

