import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from scipy.stats import multivariate_normal as sciran


class BayesianClassifier:
    """
    베지안 분류기 클래스

    self.case: 공분산 정의 방식 -> 공통 공분산일 경우 common 아닐경우 그 외.
    self.mu1, self.mu2: 클래스 1, 2의 평균 벡터.
    self.sgm1, self.sgm2: 클래스 1, 2의 공분산.
    self.p1, self.p2: 클래스 사전 확률.
    Mahalanobis Distance 사용

    """

    def __init__(self, case="common"):
        # 'common' : LDA
        # 'general' : QDA
        self.case = case

    def fit(self, X1, X2):
        # 클래스 평균 계산
        self.mu1 = np.mean(X1, axis=0)
        self.mu2 = np.mean(X2, axis=0)

        # 공분산 계산 (케이스별)
        if self.case == "common":
            # 공통 공분산: 공분산들의 가중 평균
            # 각 클래스의 개수는 모두 N=100으로 동일하므로 행렬의 산술 평균을 구해준다
            self.sgm1 = (np.cov(X1.T) + np.cov(X2.T)) / 2
            self.sgm2 = self.sgm1
        else:
            # 개별 공분산
            self.sgm1 = np.cov(X1.T)
            self.sgm2 = np.cov(X2.T)

        # 사전 확률
        # 현재는 클래수의 개수가 같으므로 사전 확률은 모두 0.5이나
        # 표본의 크기가 다른 경우는 사전확률을 따로 계산해줘야 함.
        self.p1 = X1.shape[0] / (X1.shape[0] + X2.shape[0])
        self.p2 = X2.shape[0] / (X1.shape[0] + X2.shape[0])

    def discriminant_function(self, x, mu, sgm, p):

        inv_sgm = np.linalg.inv(sgm)

        if self.case == "common":
            # 공통 공분산 판별함수 : linear
            xval = x @ inv_sgm @ mu  # 선형 : μ^T Σ^{-1} x
            c = -0.5 * mu.T @ inv_sgm @ mu  # 상수 : -0.5 μ^T Σ^{-1} μ
            return xval + c + np.log(p)

        else:
            # 개별 공분산 판별함수 : non-linear
            squareval = -0.5 * (x - mu).T @ inv_sgm @ (x - mu)  # 2차, 
            logDet = -0.5 * np.log(np.linalg.det(sgm))  # 로그 det
            return squareval + logDet + np.log(p)

    def predict(self, X):
        scores1 = [
            self.discriminant_function(x, self.mu1, self.sgm1, self.p1) for x in X
        ]
        scores2 = [
            self.discriminant_function(x, self.mu2, self.sgm2, self.p2) for x in X
        ]
        return np.where(np.array(scores1) > np.array(scores2), 0, 1)


# 1. 조건에 맞는 랜덤 데이터 생성
# 클래스 2개, sample 각 100개 총 200개
N = 100
totals = N * 2
np.random.seed(0)

# 평균 vec
cls1_m = np.array([0, 0])
cls2_m = np.array([3, 5])

# cov vec
cls1_sgm = 4 * np.array([[4, 0], [0, 4]])
cls2_sgm = np.array([[3, 0], [0, 5]])

# random data gen.
# 단순 샘플링만 수행하기 위해 numpy 사용
cls1_x = multivariate_normal(cls1_m, cls1_sgm, N)
cls2_x = multivariate_normal(cls2_m, cls2_sgm, N)

# 새로운 데이터 포인트
x1 = np.array([1, 2])
x2 = np.array([0, -2])
newDatas = np.array([x1, x2])

# LDA로 분류
lda_res = BayesianClassifier(case="common")
lda_res.fit(cls1_x, cls2_x)
lda_predic = lda_res.predict(newDatas)

# QDA로 분류
qda_res = BayesianClassifier(case="general")
qda_res.fit(cls1_x, cls2_x)
qda_predic = qda_res.predict(newDatas)

# 결과 출력
print("LDA 분류 결과:")
print(f"x1 = [1, 2]는 클래스 C{lda_predic[0]}에 속합니다.")
print(f"x2 = [0, -2]는 클래스 C{lda_predic[1]}에 속합니다.")

print("\nQDA 분류 결과:")
print(f"x1 = [1, 2]는 클래스 C{qda_predic[0]}에 속합니다.")
print(f"x2 = [0, -2]는 클래스 C{qda_predic[1]}에 속합니다.")


# 1. 산점도를 그리고 
# 2. 산점도에 새로운 포인트와 결정 경계 표시하기
plt.figure(figsize=(20, 10))

# 첫 번째: 원본 데이터 시각화
plt.subplot(1, 2, 1)
plt.scatter(cls1_x[:, 0], cls1_x[:, 1], 
            c="pink", 
            marker="+", 
            s=30, 
            label="Class 1")
plt.scatter(cls2_x[:, 0], cls2_x[:, 1], 
            c="skyblue", 
            marker="x", 
            s=30, 
            label="Class 2")
plt.title("Original data")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend(loc = 'lower left')
plt.axis("equal")
plt.grid(True, linestyle="--", alpha=0.5)


# 두 번째: 원본 데이터 + 결정 경계와 새 데이터 시각화
plt.subplot(1, 2, 2)

# 원래 데이터 포인트 그리기
plt.scatter(cls1_x[:, 0], cls1_x[:, 1], 
            c="pink", 
            marker="+", 
            s=30, 
            label="Class 1")
plt.scatter(cls2_x[:, 0], cls2_x[:, 1], 
            c="skyblue", 
            marker="x", 
            s=30, 
            label="Class 2")

# 새로운 데이터 포인트 그리기
plt.scatter(x1[0], x1[1],
    c="red",
    marker="s",
    s=100,
    edgecolors="black",
    label=f"x1 = [1, 2], LDA: C{lda_predic[0]}, QDA: C{qda_predic[0]}",
)
plt.scatter(x2[0], x2[1],
    c="blue",
    marker="p",
    s=100,
    edgecolors="black",
    label=f"x2 = [0, -2], LDA: C{lda_predic[1]}, QDA: C{qda_predic[1]}",
)

# 결정 경계 시각화 (그리드 생성)
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()

# 축 범위 확장
x_min, x_max = min(x_min, -3), max(x_max, 7)
y_min, y_max = min(y_min, -3), max(y_max, 7)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
grid = np.c_[xx.ravel(), yy.ravel()]

'''
결정 경계를 그리는 과정
모든 격자점을 LDA/QDA 분류기로 분류한다. 그러면 각 점마다 0 또는 1 를 반환한다.
(because : return np.where(np.array(scores1) > np.array(scores2), 0, 1) 므로)
1또는 2를 반환하므로 분류 결과가 (z = 1.5) 인 지점을 등고선처럼 이어서 그린다.
z_lda, z_qda : x, y의 같은 높이를 갖는 점들.
'''

# LDA 결정 경계
z_lda = lda_res.predict(grid) # 모든 격자점을 LDA 분류기로 분류함, 그러면 (각 점마다 1 또는 2 값 반환)
z_lda = z_lda.reshape(xx.shape) # 1차원 분류 결과를 원래 격자 모양(500x500)으로 변환
# 1또는 2를 반환하므로 분류 결과가 (z = 1.5) 인 지점을 등고선처럼 이어서 그림
plt.contour(xx, yy, z_lda, levels=[0.5], colors="forestgreen", linewidths = 1.5)

# QDA 결정 경계
z_qda = qda_res.predict(grid) # 모든 격자점을 QDA 분류기로 분류함, 그러면 (각 점마다 1 또는 2 값 반환)
z_qda = z_qda.reshape(xx.shape) # 1차원 분류 결과를 원래 격자 모양(500x500)으로 변환
# 1또는 2를 반환하므로 분류 결과가 (z = 1.5) 인 지점을 등고선처럼 이어서 그림
plt.contour(xx, yy, z_qda, levels=[0.5], colors="orange", linewidths= 1.5)

# 시각화
plt.title("Bayes Classifier Decision Boundary")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True, linestyle="--", alpha=0.5)
plt.axis("equal")

plt.plot([], [], color="forestgreen", linestyle="--", label="LDA Boundary")
plt.plot([], [], color="orange", linestyle="--", label="QDA Boundary")
plt.legend(loc='lower left')

plt.tight_layout()
plt.show()
