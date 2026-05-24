# 🩺 유방암 진단 예측 모델 (Breast Cancer Prediction)

> **GitHub를 활용한 Open Source Software(OSS) 머신러닝 파이프라인 구축 실습**

## 📌 프로젝트 소개
scikit-learn의 **위스콘신 유방암 데이터셋(`load_breast_cancer`)**을 활용하여 종양의 악성(Malignant) 및 양성(Benign) 여부를 판별하는 이진 분류(Binary Classification) 머신러닝 프로젝트입니다. 
단순한 모델 구현을 넘어, 의료 데이터의 특성을 반영한 상세 평가지표(Precision, Recall)를 분석하고, 질병 예측에 가장 큰 영향을 미치는 주요 세포 특징(Feature Importance)을 도출하는 것을 목표로 합니다.

## 🛠 기술 스택 (Tech Stack)
* **Language:** Python 3.x
* **Library:** scikit-learn
* **VCS:** Git & GitHub

## 📂 파일 구성
* `oss_assignment.py`: 데이터 로드, 전처리, 모델 학습, 평가 및 특성 중요도 분석이 모두 포함된 소스 코드

## 🚀 핵심 구현 내용
1. **데이터 전처리:** 전체 데이터셋을 학습용(Train) 8, 테스트용(Test) 2의 비율로 분리 (`random_state=42` 적용으로 재현성 확보)
2. **모델 학습:** 과적합(Overfitting) 방지 및 높은 예측 성능을 위해 앙상블 기법인 **랜덤 포레스트 분류기(RandomForestClassifier)** 채택
    * Hyperparameter Tuning: `n_estimators=200`, `max_depth=5`
3. **모델 평가:** * 단순 정확도(Accuracy) 산출
    * `classification_report`를 활용한 정밀도(Precision), 재현율(Recall), F1-Score 다각도 분석
4. **특성 중요도(Feature Importance) 분석:** 악성 종양 판별에 기여한 상위 3개 핵심 인자 도출

## 📊 분석 결과 (Results)
* **테스트 데이터 정확도:** `0.9649` (약 96.5%)
* **주요 발병 예측 인자 Top 3:**
  1. `worst area` (종양의 최대 면적)
  2. `worst concave points` (종양의 최대 오목한 점의 수)
  3. `mean concave points` (종양의 평균 오목한 점의 수)

