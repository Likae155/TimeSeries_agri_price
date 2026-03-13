import pandas as pd
import numpy as np

# =============================================================================
# [설정 상수] - standardize_origin() 및 전처리 전반에서 사용됨
# =============================================================================

# 1. 시도 명칭 표준화 맵: '강원도' -> '강원특별자치도' 등 다양한 표기를 하나로 통일
SIDO_CLEAN_MAP = {
    '강원': '강원특별자치도', '강원도': '강원특별자치도', '강원특별자치도': '강원특별자치도',
    '전북': '전북특별자치도', '전라북도': '전북특별자치도', '전북특별자치도': '전북특별자치도',
    '제주': '제주특별자치도', '제주도': '제주특별자치도', '제주특별자치도': '제주특별자치도',
    '세종시': '세종특별자치시', '세종특별자치시': '세종특별자치시',
    '경북': '경상북도', '경상북도': '경상북도', '경남': '경상남도', '경상남도': '경상남도',
    '전남': '전라남도', '전라남도': '전라남도', '충남': '충청남도', '충청남도': '충청남도',
    '충북': '충청북도', '충청북도': '충청북도', '경기': '경기도', '경기도': '경기도',
    '서울': '서울특별시', '서울특별시': '서울특별시', '부산': '부산광역시', '부산광역시': '부산광역시',
    '대구': '대구광역시', '대구광역시': '대구광역시', '인천': '인천광역시', '인천광역시': '인천광역시',
    '광주': '광주광역시', '광주광역시': '광주광역시', '대전': '대전광역시', '대전광역시': '대전광역시',
    '울산': '울산광역시', '울산광역시': '울산광역시'
}

# 2. 수입 국가 리스트: 산지 정보가 국가명으로 들어온 경우를 판별
COUNTRY_LIST = [
    '베트남', '중국', '미국', '미얀마', '뉴질랜드', '대만', '스페인', '호주', 
    '오스트리아', '멕시코', '필리핀', '가나', '벨기에', '일본', '네덜란드', 
    '칠레', '태국', '페루', '마이너'
]

# 3. 수입 판정 키워드: 국가명 외에 수입산임을 나타내는 키워드
IMPORT_KEYWORDS = ['수입산', '기타국', '원양산', '태평양', '대서양']

# 수입 판정 통합 키워드 셋 (검색 속도 최적화용)
OVERSEAS_KEYWORDS = set(COUNTRY_LIST + IMPORT_KEYWORDS)

# 4. 시군구 정제 제외 리스트: 시군구 컬럼에 시도 명칭이 중복 기입된 경우를 걸러내기 위함
SIDO_NAMES = set(list(SIDO_CLEAN_MAP.keys()) + list(SIDO_CLEAN_MAP.values()))


# =============================================================================
# [메인 파이프라인]
# =============================================================================

def preprocess(df):
    """모든 전처리 단계를 로직 순서에 따라 실행하는 메인 함수"""
    original_len = len(df)
    
    # 1. 기본 수치 데이터 정제
    df = clean_numeric_columns(df)
    df = filter_invalid_rows(df)
    
    # 2. 산지 정보 정제
    df = standardize_origin(df)
    
    # 3. 시간 데이터 파생 변수 생성
    _preprocess_date(df)
    
    # 4. 핵심 컬럼 정리 및 재정렬
    df = select_essential_columns(df)
    
    # 5. 인적 오류(Human Error) 제거 (프로세스 가장 마지막 단계)
    df = remove_human_errors(df)
    
    # 6. 인덱스 재정렬 및 최종 리포트 출력
    df = df.reset_index(drop=True)
    final_len = len(df)
    removed_len = original_len - final_len
    removed_ratio = (removed_len / original_len * 100) if original_len > 0 else 0
    print(f"--- [전처리 완료] 초기 데이터: {original_len:,}행 -> 정제 후: {final_len:,}행 (총 {removed_ratio:.2f}% 제거됨) ---")
    
    return df


# =============================================================================
# [1단계: 수치 및 유효성 전처리]
# =============================================================================

def clean_numeric_columns(df):
    """수치형 컬럼(총거래금액, 총거래물량, 평균가격)의 콤마 제거 및 숫자 변환"""
    target_cols = ['총거래금액', '총거래물량', '평균가격']
    for col in target_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
            # dtype이 object(문자열 포함)인 경우에만 콤마 제거 로직 실행 (regex=False로 속도 향상)
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
            else:
            # 이미 숫자형이라면 단순 변환
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def filter_invalid_rows(df):
    """유효한 데이터(0보다 큰 값)만 필터링"""
    initial_count = len(df)
    # 총거래물량은 1kg 이상이어야 함
    valid_mask = (
        (df['총거래금액'] > 0) & 
        (df['총거래물량'] >= 1) & 
        (df['평균가격'] > 0)
    )
    df = df[valid_mask].copy()
    
    final_count = len(df)
    dropped_count = initial_count - final_count
    
    print(f"작업 전 데이터 수: {initial_count:,} 행")
    print(f"삭제된 데이터 수: {dropped_count:,} 행")
    print(f"최종 정제 데이터 수: {final_count:,} 행")
    print(f"데이터 유지율: {(final_count / initial_count) * 100:.2f}%" if initial_count > 0 else "데이터 없음")
    
    return df


# =============================================================================
# [2단계: 산지 및 지역 정보 정제]
# =============================================================================

def standardize_origin(df):
    """산지 정보를 정제하는 통합 모듈"""
    df = _classify_origin_type(df)
    df = _clean_sigungu_names(df)
    df = _format_final_origin(df)
    return df

def _classify_origin_type(df):
    """국산/수입 여부 및 기본 광역시도 분류"""
    df['광역시도'] = '기타/미상'
    df['국산여부'] = '기타/미상'

    # 판별 마스크
    is_domestic = df['산지-광역시도'].isin(SIDO_CLEAN_MAP.keys())
    is_overseas_direct = df['산지-광역시도'].isin(OVERSEAS_KEYWORDS)
    is_overseas_hyphen = (df['산지-광역시도'] == '-') & (df['품종'].str.contains(r'\(수입\)', na=False, regex=True))
    is_overseas = is_overseas_direct | is_overseas_hyphen

    # 국산 데이터 매핑
    df.loc[is_domestic, '광역시도'] = df['산지-광역시도'].map(SIDO_CLEAN_MAP)
    df.loc[is_domestic, '국산여부'] = '국산'

    # 수입 데이터 매핑
    df.loc[is_overseas, '국산여부'] = '수입'
    is_country_name = df['산지-광역시도'].isin(COUNTRY_LIST)
    df.loc[is_country_name, '광역시도'] = df['산지-광역시도']
    
    return df

def _clean_sigungu_names(df):
    """시군구 정보 정제 및 예외 처리"""
    df['시군구'] = df['산지-시군구']
    
    # 국산/수입 여부에 따른 시군구 강제 할당
    df.loc[df['국산여부'] == '수입', '시군구'] = '수입'
    df.loc[df['국산여부'] == '기타/미상', '시군구'] = '기타/미상'

    # 시군구 컬럼에 광역 지명이 잘못 들어간 경우 제거
    is_sido_in_sigungu = df['시군구'].isin(SIDO_NAMES)
    df.loc[is_sido_in_sigungu, '시군구'] = '기타/미상'
    
    return df

def _format_final_origin(df):
    """최종 통합 '산지' 컬럼 생성 (예: '경기도 안성시', '수입(미국)')"""
    sido_etc = (df['광역시도'] == '기타/미상')
    sigungu_etc = (df['시군구'] == '기타/미상')

    # 케이스별 결합 조건
    conditions = [
        (~sido_etc) & (~sigungu_etc), # 둘 다 정보 있음 -> '시도 + 시군구'
        (~sido_etc) & (sigungu_etc),  # 시도만 있음 -> '시도'
        (sido_etc) & (~sigungu_etc),  # 시군구만 있음 -> '시군구'
    ]
    choices = [
        df['광역시도'] + " " + df['시군구'], 
        df['광역시도'],                      
        df['시군구']                         
    ]
    df['산지'] = np.select(conditions, choices, default='기타/미상')

    # 수입산 특수 포맷팅
    is_import = (df['국산여부'] == '수입')
    temp_country = df.loc[is_import, '광역시도'].replace('기타/미상', '기타')
    df.loc[is_import, '산지'] = "수입(" + temp_country + ")"
    
    return df


# =============================================================================
# [3단계: 시간 데이터 파싱]
# =============================================================================

def _preprocess_date(df):
    """날짜 변환 및 파생 변수 생성"""
    # 2. 날짜 변환 (여러 형식이 섞여 있는 경우를 위한 format='mixed' 적용)
    df['DATE'] = pd.to_datetime(df['DATE'], format='mixed')
    df['Year'] = df['DATE'].dt.year
    df['Quarter'] = df['DATE'].dt.quarter
    df['Month'] = df['DATE'].dt.month
    df['Week'] = df['DATE'].dt.isocalendar().week
    df['Day'] = df['DATE'].dt.day
    df['DayOfWeek'] = df['DATE'].dt.day_name()
    
    # 요일 순서 정렬
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['DayOfWeek'] = pd.Categorical(df['DayOfWeek'], categories=days_order, ordered=True)


# =============================================================================
# [4단계: 핵심 컬럼 선택]
# =============================================================================

def select_essential_columns(df):
    """분석에 필요한 핵심 컬럼만 유지하고 재정렬"""
    # 1. 평균가격 새로 계산 (kg당 가격)
    df['평균가격'] = df['총거래금액'] / df['총거래물량']
    
    # 2. 필수 컬럼 정의 및 정렬 순서
    # DATE (앞) + 핵심 지표 (중간) + 시계열 파생변수 (뒤)
    essential_cols = [
        'DATE', '총거래금액', '총거래물량', '평균가격', '품종', '등급', '국산여부', '산지',
        'Year', 'Quarter', 'Month', 'Week', 'Day', 'DayOfWeek'
    ]
    
    # 실제 존재하는 컬럼만 선택
    existing_cols = [col for col in essential_cols if col in df.columns]
    return df[existing_cols].copy()


# =============================================================================
# [5단계: 인적 오류 제거]
# =============================================================================

def remove_human_errors(df):
    """
    평균가격(단가)을 기준으로 인적 오류(Human Error) 가능성이 높은 행 제거.
    - 그룹핑 수준: 1차(품종+등급) -> 2차(품종) -> 3차(전체) 단계별 방어 로직 적용
    """
    if df.empty:
        return df
        
    # 결측치가 있으면 groupby 연산에서 누락될 수 있으므로 임시 채움
    df['품종'] = df['품종'].fillna('미상')
    df['등급'] = df['등급'].fillna('미상')
    
    # 1차 기준: [Year, Week, 품종, 등급] 단위의 세밀한 중앙값
    med_lvl1 = df.groupby(['Year', 'Week', '품종', '등급'])['평균가격'].transform('median')
    
    # 2차 기준 (방어 로직 1): 등급별 데이터가 부족해 중앙값이 누락된 경우 [Year, Week, 품종] 상위 단위로 보완
    med_lvl2 = df.groupby(['Year', 'Week', '품종'])['평균가격'].transform('median')
    
    # 3차 기준 (방어 로직 2): 특정 품종도 부족한 극단적 경우 [Year, Week] 주 단위 전체로 보완
    med_lvl3 = df.groupby(['Year', 'Week'])['평균가격'].transform('median')
    
    # 결측치(NaN) 값을 순차적으로 메꿈 (Fallback)
    group_med = med_lvl1.fillna(med_lvl2).fillna(med_lvl3)
    
    # 2. 일관성 검사 (중앙값 대비 상하한선 설정: 0.05배 ~ 20.0배)
    lower_limit = group_med * 0.05
    upper_limit = group_med * 20.0
    
    # mask 조건: 중앙값이 구해졌고(notnull), 지정된 한계선 안쪽에 위치한 경우
    mask = group_med.notnull() & (df['평균가격'] >= lower_limit) & (df['평균가격'] <= upper_limit)
    clean_df = df[mask].copy()
    
    return clean_df


if __name__ == "__main__":
    pass