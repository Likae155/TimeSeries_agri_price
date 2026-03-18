# EST 15기 시계열데이터 프로젝트 - 농산물 팀

데이콘(https://dacon.io/competitions/official/235801/overview/description)
![alt text](https://raw.githubusercontent.com/Likae155/TimeSeries_agri_price/main/image/image.png?raw=true)

#### 팀원: 박창민, 안도겸, 정영석

GitHub: https://github.com/Likae155/TimeSeries_agri_price

# 주제: 21년도 농산물 가격 예측

## 대회 설명
![한국농수산식품유통공사](https://raw.githubusercontent.com/Likae155/TimeSeries_agri_price/main/image/image-1.png?raw=true)                  ![데이콘](https://raw.githubusercontent.com/Likae155/TimeSeries_agri_price/main/image/image-3.png?raw=true)          
**한국농수산식품유통공사**가 주최하고 **데이콘**이 주관한 대회

2016년 부터 2021년 까지의 **전국 도매시장 거래정보 데이터**를 제공

![배추](https://raw.githubusercontent.com/Likae155/TimeSeries_agri_price/main/image/image-4.png?raw=true)    ![무](https://raw.githubusercontent.com/Likae155/TimeSeries_agri_price/main/image/image-6.png?raw=true) 배추나 무와 같은 21가지 농작물의 21년도 가격을 예측하는 AI 경진대회

t일에 t-1일까지의 데이터를 갖고 t+7, +14, +28의 데이터를 예측해야 함
![alt text](https://raw.githubusercontent.com/Likae155/TimeSeries_agri_price/main/image/image-20.png?raw=true)

<br>
<br>

## 참여 성과
26년도의 가격을 예측한 모델의 평가지표
![alt text](https://raw.githubusercontent.com/Likae155/TimeSeries_agri_price/main/image/image-19.png?raw=true)
![alt text](https://raw.githubusercontent.com/Likae155/TimeSeries_agri_price/main/image/image-7.png?raw=true)
<br>
<br>


# 데이터
## 대회에서 제공한 데이터
1. base line 데이터
- 각 작물의 거래량과 가격을 일 단위로 집계한 데이터
- 원본데이터를 가공하여 배포한 데이터셋
![alt text](https://raw.githubusercontent.com/Likae155/TimeSeries_agri_price/main/image/image-9.png?raw=true)

2. 원본 데이터
- 판매 일자, 시장, 법인, 품목, 품종, 단위, 포장, 산지 등등...
![alt text](https://raw.githubusercontent.com/Likae155/TimeSeries_agri_price/main/image/image-10.png?raw=true)

3. API를 통해 추가되는 데이터
- 현 시점에서 접근 불가

![alt text](https://raw.githubusercontent.com/Likae155/TimeSeries_agri_price/main/image/image-11.png?raw=true)

## 새로운 데이터
1. 전국 도매시장 경락 정보
- 전국 도매시장의 경매 낙찰 정보
- 각각의 거래의 일자, 품목, 물량, 가격, 산지 등의 정보
- 22년 1월 ~ 현재까지의 데이터
![alt text](https://raw.githubusercontent.com/Likae155/TimeSeries_agri_price/main/image/image-12.png?raw=true)

<br>
<br>

# 목표: 22년~25년의 데이터로 26년 1월의 가격을 예측

## 데이터 설명
### 거래 단위
- 총거래금액(원), 총거래물량(kg)
- '음수'로 표현되는 데이터가 존재: 취소거래
    - 거래일과 취소일이 동일하지 않을 경우 통계상 왜곡 발생 가능.
![alt text](https://raw.githubusercontent.com/Likae155/TimeSeries_agri_price/main/image/image-15.png?raw=true)

### **농작물**
- 총 **21가지**의 작물
    - 대상품목(16): 배추, 무, 양파, 건고추, 마늘, 대파, 얼갈이배추, 양배추, 깻잎, 시금치, 미나리, 당근, 파프리카, 새송이, 팽이버섯, 토마토
    - 대상품종(5): 청상추, 백다다기, 애호박, 캠벨얼리, 샤인마스캇
- 품목: 큰 단위. ex) 배추, 포도
- 품종: 품목 안의 작은 분류. ex) 고랭지 배추/ 김장 배추, 캠벨얼리/샤인마스캇

## 1. 계절성
배추, 무: 대표적인 계절성 작물.

    - 기온에 따른 재배지 이동(여름-고랭지, 겨울-남부지방)
    - '김장철'이라는 수요의 폭발이 있는 작물

이 데이터셋에는 계절성이 잘 나타나는가?
<br>
<br>

![alt text](https://raw.githubusercontent.com/Likae155/TimeSeries_agri_price/main/image/배추_03_monthly_trend.png?raw=true)
 - 예상과 달리 김장철의 거래물량이 적다
 - 김장배추는 도매시장을 통해 거래하지 않고 대형 마트, 농협 등으로 직거래하는 유통 구조
 - 9월의 고랭지 배추는 도매시장으로 유통되기 때문에 거래 물량이 많이 잡힌다.

 ### 상식적인 계절성은 아니여도, 이 데이터셋만의 고유한 계절성이 있다.

## 2. 추세
 - 단기적인 급락과 급등은 반복하지만, 장기적인 추세가 두드러지게 나타나지 않는다. <br>
![alt text](https://raw.githubusercontent.com/Likae155/TimeSeries_agri_price/main/image/image-14.png?raw=true)

## 3. 21가지 농작물의 차이
### 🥬 배추 vs 🍃 깻잎 vs 🍅 토마토 비교

| 구분 | 배추 (Cabbage) | 깻잎 (Perilla Leaf) | 토마토 (Tomato) |
| :--- | :--- | :--- | :--- |
| **품목 분류** | 노지 채소 (엽경채류) | 시설 채소 (엽채류) | 과채류 (열매채소) |
| **계절성** | **매우 강함** (산지 이동형) | **매우 낮음** (연중 일정함) | **중간** (봄~여름 피크) |
| **재배 방식** | 주로 노지(실외 밭) 재배 | 주로 비닐하우스(시설) 재배 | 노지 및 스마트팜 시설 재배 |
| **적정 기온** | **호냉성** ($15\sim20^{\circ}\text{C}$) | **호온성** ($20\sim30^{\circ}\text{C}$) | **호온성** ($20\sim27^{\circ}\text{C}$) |
| **거래 패턴** | **김장철(11월) 폭발적 증가** | 일 년 내내 꾸준한 유통량 | 기온 상승 시 급증 |
| **보관 특성** | 품종에 따라 장기 보관 가능 | 수확 후 수명이 매우 짧음 | 후숙 가능, 신선도 중요 |

## 하나의 모델로 모든 작물을 예측하기 어렵다.

# 데이터 전처리(EDA를 위한)
### 1. 이상 거래 제거
- 취소 거래(금액 또는 물량이 음수로 표현되는 데이터) 제거
- 거래물량이 1kg 미만인 데이터 제거(도매시장임을 감안하면 이상치로 간주 가능)

### 2. 산지 정보 표준화
- 시도 컬럼과 시군구 컬럼 통합
- '충북', '충청북도' 와 같이 중복된 이름 통합

### 3. 시간 데이터 생성

### 4. 품종, 등급 등의 데이터 정리
- 거래량을 중심으로 정리

### 5. 이상치
- 주단위 평균가격에 비해 지나치게 높거나 낮은 가격 제거(0.05배, 20배)
- 가격의 급등락이 심한 농작물의 특성상 많은 이상치를 제거하지 못함

![alt text](https://raw.githubusercontent.com/Likae155/TimeSeries_agri_price/main/image/image-16.png?raw=true) ![alt text](https://raw.githubusercontent.com/Likae155/TimeSeries_agri_price/main/image/배추_08_품종_trend.png?raw=true)


# 추가 전처리 및 파생변수 생성
- train, test 데이터 통합하여 전처리 진행
### 1. 결측치 보간작업
- 공휴일 등 거래가 없는 날
- 시장은 열려있지만 특정 작물의 거래가 없는 날
- ffill을 사용하여 직전 유효 데이터로 보간
- 날짜 관련 파생 변수 다시 생성
![alt text](https://raw.githubusercontent.com/Likae155/TimeSeries_agri_price/main/image/캠벨얼리_05_daily_trend.png?raw=true)

### 2. 숫자 컬럼의 로그 변환
- 농작물 별 무게의 차이로 kg당 가격이 상이
![alt text](https://raw.githubusercontent.com/Likae155/TimeSeries_agri_price/main/image/image-17.png?raw=true)

### 3. 명절 연휴 명시
- 명절 전후의 기간을 명시하여 모델이 명절 효과를 학습할 수 있도록 함
- 명절 전에는 물동량이 급증하고 명절 후에는 수요가 급감하는 일반적인 패턴이 있기 때문

### 4. Lag 변수, 이동평균(MA) 및 표준편차(STD)
- Lag변수: 과거 특정 시점의 가격. 주간 반복 패턴
- 이동평균(MA): 과거 n일간의 평균 가격. 단기적인 노이즈를 제거
- 표준편차(STD): 과거 n일간의 가격 표준편차. 최근 거래액의 변동성


### 4. 유가 데이터 추가
- 국제 유가를 추가하여 운송비, 난방비 등의 상승을 반영
- 국제 유가의 국내 기름값 반영 시차를 고려하여 28일 시차변수 생성

### 5. 기상 데이터 추가
- 산지에 따른 일평균, 최고, 최저 기온 및 강수량 데이터를 활용해 폭염, 한파, 호우, 가뭄 등의 극단적 기상 플래그를 도출
- 작물 생리 특성에 따라 엽채류(7일), 과채류(14일), 구근/저장류(30일)로 그룹화하여 시차 및 누적 기상 플래그를 차등 반영


![alt text](https://raw.githubusercontent.com/Likae155/TimeSeries_agri_price/main/image/image-18.png?raw=true)

### 6. 품목 정체성 임베딩 (PCA)
 - 품목 간의 상이한 데이터 분포를 모델이 인지할 수 있도록 평균 가격, 가격 변동성, 거래 빈도, 유가 상관계수, 기상 민감도를 통계량으로 추출
 - 유가 상관계수: 난방이 중요한 하우스 작물 등의 가중치
 - 기상 민감도: 노지에서 키워 기상 변화에 민감한 작물 등의 가중치
 - 추출된 통계량을 StandardScaler로 표준화한 뒤, 주성분 분석(PCA)을 통해 3차원 벡터(vec0, vec1, vec2)로 압축하여 각 품목의 임베딩 변수로 활용
 - '품목'을 임베딩 하지 않은 이유는 품목별 데이터 양의 차이로 적절한 학습이 이루어 지지 않을 수 있기 때문

# 모델 생성
### 1. ML 파트 (LightGBM)
- L1 손실 함수를 목적 함수로 설정하고 트리 개수를 150개로 적용
- 각 품목별 1주, 2주, 4주 후를 예측하는 독립적인 회귀 모델을 학습
- 21 x 3 = 63개의 모델 생성

### 2. DL 파트 (Chronos-2 Large)
- AutoGluon의 TimeSeriesPredictor를 활용하여 시계열 데이터를 추론
- 기상, 연휴, 유가 데이터를 공변량 설정

### 3. 모델 예측
- 28일 Rolling Window 예측을 수행
    - 오늘의 일자: t
    - 사용 데이터: t - 1일
    - 예측 일자: t + 1주, +2주, +4주
    - 2026년 1월 1일부터 28일간 수행

### 3. Optuna 기반 가중치 최적화
- LGBM과 Chronos 모델의 로그 예측값을 결합하는 과정에서 NMAE 지표를 최소화하기 위해 Optuna 라이브러리를 도입
- 품목 및 시점별 최적의 앙상블 가중치를 계산하여 결합한 후, np.expm1으로 역변환하여 최종 가격을 산출
```
📊 [토마토] NMAE: 0.1123 (데이터 84건 샘플)
📊 [건고추] NMAE: 0.1525 (데이터 84건 샘플)
📊 [청상추] NMAE: 0.1602 (데이터 84건 샘플)
📊 [백다다기] NMAE: 0.1012 (데이터 84건 샘플)
📊 [얼갈이배추] NMAE: 0.0978 (데이터 84건 샘플)
📊 [마늘] NMAE: 0.0475 (데이터 84건 샘플)
📊 [깻잎] NMAE: 0.1877 (데이터 84건 샘플)
📊 [양배추] NMAE: 0.1343 (데이터 84건 샘플)
📊 [당근] NMAE: 0.0572 (데이터 84건 샘플)
📊 [애호박] NMAE: 0.0956 (데이터 84건 샘플)
📊 [양파] NMAE: 0.0501 (데이터 84건 샘플)
📊 [미나리] NMAE: 0.1247 (데이터 84건 샘플)
📊 [샤인마스캇] NMAE: 0.1120 (데이터 84건 샘플)
📊 [대파] NMAE: 0.1116 (데이터 84건 샘플)
📊 [시금치] NMAE: 0.0845 (데이터 84건 샘플)
📊 [팽이버섯] NMAE: 0.1416 (데이터 84건 샘플)
📊 [파프리카] NMAE: 0.1708 (데이터 84건 샘플)
📊 [새송이] NMAE: 0.1160 (데이터 84건 샘플)
📊 [배추] NMAE: 0.0989 (데이터 84건 샘플)
📊 [무] NMAE: 0.1329 (데이터 84건 샘플)
📊 [캠벨얼리] NMAE: 1.0790 (데이터 84건 샘플)

🏆 최종 전체 품목 평균 NMAE: 0.1604
```

## 대안 및 비판적 검토
- 더 많은 데이터에 접근할 수 있었다면 더 좋은 결과를 기대할 수 있지 않았나 하는 아쉬움
예를 들어, '당근'의 경우 수입산의 비중이 커 관세청의 수입량 데이터를 확인했지만, 월별 통계 이상의 데이터 얻을 수 없어 일별 예측에 활용하지 못함
- 임베딩한 데이터가 적절히 사용되지 못한 한계. 현재는 앙상블 가중치에만 반영. 모델에 넣고 돌렸을 때는 오히려 더 안좋은 결과가 발생.

## 데이터 출처
- 농산물 데이터: 농넷 (https://www.nongnet.or.kr/front/M000000222/content/view.do)
- 유가 데이터: 오피넷 (https://www.opinet.co.kr/glopopdSelect.do)
- 기상 데이터: 기상청 (https://apihub.kma.go.kr/)

