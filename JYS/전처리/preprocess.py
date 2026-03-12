import pandas as pd
import numpy as np

# =============================================================================
# [설정 상수] - standardize_origin() 함수에서 사용됨
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
# [전처리 함수]
# =============================================================================

def clean_numeric_columns(df):
    """수치형 컬럼(총거래금액, 총거래물량, 평균가격)의 콤마 제거 및 숫자 변환"""
    target_cols = ['총거래금액', '총거래물량', '평균가격']
    for col in target_cols:
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
    
    valid_mask = (
        (df['총거래금액'] > 0) & 
        (df['총거래물량'] > 0) & 
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

def standardize_unit_capacity(df):
    """거래단위를 kg 기준으로 표준화하여 '총거래물량(kg)' 컬럼 생성"""
    extracted = df['거래단위'].str.extract(r'(\d+\.?\d*)\s*(.*)')
    extracted.columns = ['value', 'unit_part']
    extracted['value'] = pd.to_numeric(extracted['value'], errors='coerce')
    extracted['factor'] = 1.0
    # Case A: ton 또는 M/T (1,000배)
    is_ton = extracted['unit_part'].str.contains('ton|M/T', case=False, na=False, regex=True)
    extracted.loc[is_ton, 'factor'] = 1000.0

    # Case B: g (1/1000배) - kg는 제외하는 정규표현식 (?<!k)g 사용
    is_g = extracted['unit_part'].str.contains(r'(?<!k)g', case=False, na=False, regex=True)
    extracted.loc[is_g, 'factor'] = 0.001

    df['총거래물량(kg)'] = (extracted['value'] * extracted['factor']) * df['총거래물량']
    return df


# --- standardize_origin 모듈화 (세부 단계 분리) ---

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
    temp_country = df.loc[is_import, '광역시도'].replace('기타/미상', '수입(기타)')
    df.loc[is_import, '산지'] = "수입(" + temp_country + ")"
    
    return df

def standardize_origin(df):
    """산지 정보를 정제하는 메인 모듈 (세부 로직을 순차적으로 실행)"""
    df = _classify_origin_type(df)
    df = _clean_sigungu_names(df)
    df = _format_final_origin(df)
    return df


# =============================================================================
# [메인 파이프라인]
# =============================================================================

def preprocess(df):
    """모든 전처리 단계를 실행하는 메인 함수"""
    df = clean_numeric_columns(df)
    df = filter_invalid_rows(df)
    df = standardize_unit_capacity(df)
    df = standardize_origin(df)
    return df

if __name__ == "__main__":
    pass