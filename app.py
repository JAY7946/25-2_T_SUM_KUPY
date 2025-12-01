import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==============================================================================
# 0. 핵심 설정 (학습 코드와 함수명/로직 완전 일치 필수)
# ==============================================================================
# ★ 주의: joblib으로 저장된 파이프라인을 불러올 때, 
# 학습 코드에 정의된 함수와 '이름' 및 '로직'이 동일한 함수가 현재 파일에도 존재해야 합니다.
# (FunctionTransformer가 __main__ 네임스페이스의 함수를 참조하기 때문)

SNS_WEIGHT = 1.1      # 학습 코드와 동일하게 1.1로 수정
STUDY_WEIGHT = 1.2    # 학습 코드와 동일하게 1.2로 추가

def apply_sns_weight(x):
    """SNS 사용 시간에 가중치를 부여하는 함수 (학습 파이프라인과 동일 이름 필수)"""
    return x * SNS_WEIGHT

def apply_study_weight(x):
    """공부 시간에 가중치를 부여하는 함수 (학습 파이프라인과 동일 이름 필수)"""
    return x * STUDY_WEIGHT

# ==============================================================================
# 1. 설정 및 데이터 로드
# ==============================================================================
st.set_page_config(page_title="학생 공부 진단 AI", layout="wide", page_icon="🎓")

@st.cache_data
def load_resources():
    # 1. 모델 및 전처리기 로드
    model = None
    preprocess = None
    
    # 경로 설정 (로컬 실행 환경 또는 Streamlit Cloud 환경 고려)
    model_path = 'kmeans_model.pkl'
    prep_path = 'preprocessor.pkl'
    
    try:
        if os.path.exists(model_path) and os.path.exists(prep_path):
            model = joblib.load(model_path)
            preprocess = joblib.load(prep_path)
        else:
            st.warning("⚠️ 학습 모델 파일('kmeans_model.pkl', 'preprocessor.pkl')을 찾을 수 없습니다. 폴더에 파일을 넣어주세요.")
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        st.info("참고: 학습 코드와 app.py의 함수 정의(apply_sns_weight 등)가 일치해야 합니다.")
        return None, None, None

    # 2. 비교 분석을 위한 원본 데이터 로드
    df = pd.DataFrame()
    try:
        # 엑셀 읽기 시도
        if os.path.exists('student_habits_performance.xlsx'):
            df = pd.read_excel('student_habits_performance.xlsx', engine='openpyxl')
        # CSV 읽기 시도
        elif os.path.exists('student_habits_performance.csv'):
            df = pd.read_csv('student_habits_performance.csv')
    except Exception as e:
        pass # 파일이 없거나 읽기 실패 시 그래프 기능만 비활성화
            
    return model, preprocess, df

model, preprocess, df_ref = load_resources()

# ==============================================================================
# 2. UI 구성 (사용자 입력)
# ==============================================================================
st.title("🎓 학생 공부 효율 & 습관 진단기 (Ver 2.0)")
st.markdown(f"""
이 AI는 학생들의 생활 습관 데이터를 학습하여 **'고효율 우등생'** 그룹과 **'습관 개선 필요'** 그룹을 분류합니다.
이번 버전에서는 **SNS 가중치를 완화({SNS_WEIGHT}배)**하고, **공부 시간 보너스({STUDY_WEIGHT}배)**를 추가하여 더 정밀하게 진단합니다.
""")

st.divider()

with st.sidebar:
    st.header("📝 내 생활 기록부")
    
    st.subheader("1. 기본 정보")
    age = st.number_input("나이", 15, 30, 18)
    gender = st.selectbox("성별", ["Male", "Female"])
    
    st.subheader("2. 시간 관리 (핵심)")
    study_hours = st.slider("✍️ 하루 공부 시간 (시간)", 0.0, 15.0, 3.0, step=0.5, help="공부 시간이 길수록 긍정적인 평가를 받습니다.")
    social_media = st.slider("📱 SNS 사용 시간 (시간)", 0.0, 10.0, 2.0, step=0.5, help="이 시간이 길면 패널티가 적용됩니다.")
    sleep_hours = st.slider("💤 하루 수면 시간 (시간)", 0.0, 12.0, 7.0, step=0.5)
    netflix = st.slider("🎬 OTT/넷플릭스 시청 시간", 0.0, 10.0, 1.0, step=0.5)
    
    st.subheader("3. 생활 및 멘탈")
    attendance = st.slider("출석률 (%)", 0, 100, 90)
    mental_health = st.slider("멘탈/기분 점수 (1:나쁨 ~ 10:좋음)", 1, 10, 5)
    exam_score = st.number_input("직전 시험 점수", 0, 100, 70)
    
    st.subheader("4. 기타 환경")
    exercise = st.selectbox("운동 빈도 (주당 횟수)", [0, 1, 2, 3, 4, 5, 6, 7])
    part_time = st.selectbox("아르바이트 여부", ["Yes", "No"])
    diet = st.selectbox("식습관 품질", ["Good", "Average", "Poor"])
    internet = st.selectbox("인터넷 환경", ["Good", "Average", "Poor"])
    extra = st.selectbox("동아리/대외활동 여부", ["Yes", "No"])

# 입력 데이터 DataFrame 변환 (학습 코드의 preprocess 순서/이름과 동일해야 함)
# 학습 코드: target_col_sns + target_col_study + num_cols_normal + cat_cols 순서로 처리됨
# 하지만 DataFrame 컬럼 이름만 맞으면 ColumnTransformer가 알아서 찾습니다.
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
if st.button("🚀 AI 진단 결과 확인하기", use_container_width=True):
    
    # ---------------------------
    # (1) AI 클러스터링 예측
    # ---------------------------
    cluster = -1
    
    if model and preprocess:
        try:
            # 전처리 실행 (자동으로 apply_sns_weight, apply_study_weight 적용됨)
            input_processed = preprocess.transform(input_data)
            cluster = model.predict(input_processed)[0]
        except Exception as e:
            st.error(f"진단 중 오류 발생: {e}")
            st.write("상세 에러:", e)
    else:
        st.error("모델 파일이 로드되지 않아 진단할 수 없습니다.")

    # ---------------------------
    # (2) 화면 레이아웃 및 결과 표시
    # ---------------------------
    col_res1, col_res2 = st.columns([1, 1.2], gap="large")
    
    # [왼쪽] AI 분석 결과
    with col_res1:
        st.subheader("🔍 분석 결과")
        
        # 클러스터 해석 (학습 데이터 경향에 따름)
        # Cluster 1: 일반적으로 성적 높고, 공부 많고, SNS 적은 그룹
        # Cluster 0: 성적 낮고, 공부 적고, SNS 많은 그룹
        # (실제 학습 결과에 따라 0과 1이 바뀔 수 있으므로, 결과가 이상하면 이 숫자를 반대로 바꾸세요)
        
        target_cluster_good = 1  # 우등생 클러스터 번호 (가정)
        
        if cluster == target_cluster_good:   
            st.success("🎉 **'자기주도 학습 마스터' 유형**")
            st.write("공부와 휴식의 밸런스가 아주 좋습니다! SNS나 OTT에 시간을 뺏기지 않고, 학습에 집중하는 모습이 훌륭합니다.")
        elif cluster != -1:
            st.warning("⚠️ **'디지털 디톡스가 필요한' 유형**")
            st.write("학습 잠재력은 충분하지만, SNS나 미디어 시청 시간이 성적 향상을 방해하고 있을 수 있습니다. 조금만 줄여볼까요?")
        
        st.markdown("---")
        st.caption("💡 **맞춤형 피드백**")
        
        # 규칙 기반 피드백 생성
        feedbacks = []
        
        # 1. SNS & 공부 밸런스 피드백 (가중치 고려)
        if social_media > 3.0:
            feedbacks.append(f"❗ **SNS 사용({social_media}시간)이 너무 많습니다.** AI가 가장 큰 감점 요인으로 보고 있어요.")
        
        if study_hours < 2.0:
            feedbacks.append(f"❗ **절대적인 공부 시간({study_hours}시간)이 부족해요.** 하루 30분만 더 늘려보세요. 가산점이 큽니다!")
        elif study_hours > 5.0 and social_media < 2.0:
            feedbacks.append("✅ **완벽한 학습 패턴입니다.** 공부량은 많고 방해 요소는 적네요.")

        # 2. 수면 & 멘탈
        if sleep_hours < 5.5:
            feedbacks.append("💤 **잠이 부족해요.** 수면 부족은 집중력을 30% 이상 떨어뜨립니다.")
        
        if mental_health <= 4:
            feedbacks.append("🍀 **스트레스 관리가 필요해요.** 가벼운 산책이나 명상이 도움이 될 수 있습니다.")

        # 3. 운동
        if exercise == 0:
             feedbacks.append("🏃 **운동을 전혀 안 하시네요.** 체력이 곧 성적입니다. 가벼운 걷기부터 시작하세요.")

        if not feedbacks:
            feedbacks.append("👌 특별히 지적할 나쁜 습관이 없습니다. 지금처럼만 유지하세요!")

        for fb in feedbacks:
            st.markdown(fb)

    # [오른쪽] 남들과 비교하기 그래프
    with col_res2:
        st.subheader("📊 나의 위치 분포 그래프")
        
        if not df_ref.empty:
            tab1, tab2, tab3 = st.tabs(["SNS 시간", "공부 시간", "시험 점수"])
            
            def plot_ranking(col_name, user_val, title, invert=False, unit="시간"):
                """히스토그램과 나의 위치를 그려주는 함수"""
                fig, ax = plt.subplots(figsize=(6, 3.5))
                
                # 전체 분포
                sns.histplot(df_ref[col_name], kde=True, ax=ax, color='#6C5CE7', alpha=0.5, edgecolor=None)
                
                # 내 위치
                ax.axvline(user_val, color='#E84393', linestyle='--', linewidth=2.5, label='Me')
                
                # 상위 % 계산
                percentile = (df_ref[col_name] < user_val).mean() * 100
                if invert: # 낮을수록 좋은 것 (SNS)
                    rank = percentile # 값이 작으면 percentile도 작음 -> 상위권
                    rank_text = f"상위 {rank:.1f}% (적은 편)" if rank < 50 else f"하위 {100-rank:.1f}% (많은 편)"
                else: # 높을수록 좋은 것 (공부, 점수)
                    rank = 100 - percentile
                    rank_text = f"상위 {rank:.1f}%"
                
                ax.set_title(f"{title}\n(나: {user_val}{unit} - {rank_text})", fontsize=12)
                ax.set_xlabel(unit)
                ax.set_ylabel("학생 수")
                ax.legend()
                st.pyplot(fig)

            with tab1:
                st.info("📉 SNS 사용시간 (낮을수록 좋음)")
                plot_ranking('social_media_hours', social_media, "SNS 사용 시간 분포", invert=True)
                
            with tab2:
                st.info("📈 공부 시간 (높을수록 좋음)")
                plot_ranking('study_hours_per_day', study_hours, "하루 공부 시간 분포", invert=False)
                
            with tab3:
                st.info("💯 시험 점수 (높을수록 좋음)")
                plot_ranking('exam_score', exam_score, "시험 점수 분포", invert=False, unit="점")
        else:
            st.warning("⚠️ 비교용 데이터 파일(student_habits_performance.xlsx)이 없어 그래프를 그릴 수 없습니다.")
