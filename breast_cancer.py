from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. 유방암 데이터셋 로드
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# 2. Train/Test Split (8:2 비율)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 모델 학습 (기본 설정)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. 예측 및 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=== 유방암 진단 분류 Baseline 모델 ===")
print(f"테스트 정확도(Accuracy): {accuracy:.4f}")
