import h5py  # MATLAB 7.3 이상 형식(HDF5 기반) 파일 불러오기
from scipy.io import loadmat  # MATLAB 7.2 이하 버전 파일 불러오기
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def readmatFile(path="Data.mat"):
    with h5py.File(path, "r") as f:
        print("h5py로 파일 열기")
        for key in f.keys():
            data = f[key][:]
            print(f"\n'{key}' 변수 정보:")
            print(f"  - shape: {data.shape}")
            print(f"  - dtype: {data.dtype}")
            # data shape 가 (2, 800) 므로 transpose 안 함.
        X = data[0, :]
        Y = data[1, :]
    return X, Y

# [1] 데이터 불러오기 및 산점도 그리기
def Drawplot(X, Y, title="Data.mat"):
    plt.figure(figsize=(10, 8))
    plt.scatter(X, Y, s=15)
    plt.title(title, fontsize=16)
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def print_data_statistics(X, Y):
    print(f"\n데이터 통계:")
    print(f"포인트 개수: {len(X)}")
    print(f"X 범위: [{np.min(X):.3f}, {np.max(X):.3f}]")
    print(f"Y 범위: [{np.min(Y):.3f}, {np.max(Y):.3f}]")
    print(f"X 평균: {np.mean(X):.3f}, 표준편차: {np.std(X):.3f}")
    print(f"Y 평균: {np.mean(Y):.3f}, 표준편차: {np.std(Y):.3f}")

class GMM:
    def __init__(self, n_comp=2, max_itr = 10, tolerance = 0.00001, randomSeed = None):
        self.n_comp = n_comp
        self.max_itr = max_itr
        self.tolerance = tolerance
        self.randomSeed = randomSeed

        self.mean = None
        self.cov = None
        self.weights = None

        self.log_likelihoood =[] # 로그 우도 저장

    
    def Initialize(self, X):
        n_samples, n_features = X.shape

        # 랜덤 시드 설정
        if self.randomSeed is not None:
            np.random.seed(self.randomSeed)

        # K-means++ 방식으로 초기 평균 선택
        self.means = self.kmeansInit(X)
                
        # 공분산 행렬 초기화 (전체 데이터의 공분산 사용)
        self.cov = np.array([np.cov(X.T) for _ in range(self.n_comp)])
        
        # 가중치 초기화 (균등 분포)
        self.weights = np.ones(self.n_comp) / self.n_comp

    def kmeansInit(self, X):
        '''
        k-means++ 방식으로 초기화
        1. 최초 중심점은 랜덤으로 선택.
        2. 2번 째 포인트 부터 각 데이터 포인트와 가장 가까운 기존 중심점과의 
        거리 제곱에 비례하여 선택. (멀리 떨어져 있을 수록 선택될 확률이 높음)
        '''
        n_samples = X.shape[0]
        centers = np.zeros((self.n_comp, X.shape[1]))

        # 첫번째 중심 랜덤 선택
        centers[0] = X[np.random.choice(n_samples)]

        for i in range(1, self.n_comp):
            # 각 점에서 가장 가까운 중심까지의 거리 계산
            distances = np.array([min([np.linalg.norm(x - c)**2 for c in centers[:i]]) 
                                for x in X])
            
            # 거리에 비례하는 확률로 다음 중심 선택(멀리 떨어질 수록 선택 확률 높음)
            prob = distances / distances.sum()
            centers[i] = X[np.random.choice(n_samples, p=prob)]
        
        return centers
    
    # EM-Algorithm : e-step, m-step
    def eStep(self, X):
        # e-step 계산
        n_samples = X.shape[0]
        responsMat = np.zeros((n_samples, self.n_comp)) # 데이터 포인트가 각 가우시안 컴포넌트에 속할 확률
        
        for k in range(self.n_comp):
            # 다변량 정규분포의 확률밀도함수 계산
            rv = multivariate_normal(self.means[k], self.cov[k])
            responsMat[:, k] = self.weights[k] * rv.pdf(X) # 가중치 곱
        
        # 정규화 (각 행의 합이 1이 되도록)
        respons_sum = responsMat.sum(axis=1)[:, np.newaxis] # 샘플 별 책임도 합
        responsMat = responsMat / (respons_sum + self.tolerance)  # 0으로 나누기 방지
        
        return responsMat

    def mStep(self, X, responsMat):
        # m-step : e-stpe 단계에서 계산했던 여러가지 param들을 update
        n_samples = X.shape[0]
        
        # 각 가우시안에 대한 유효 데이터 포인트 수
        Nk = responsMat.sum(axis=0)
        
        # 가중치 업데이트
        self.weights = Nk / n_samples
        
        # 평균 업데이트
        self.means = np.zeros((self.n_comp, X.shape[1]))
        for k in range(self.n_comp):
            self.means[k] = (responsMat[:, k:k+1] * X).sum(axis=0) / (Nk[k] + 1e-10)
        
        # 공분산 업데이트
        self.cov = np.zeros((self.n_comp, X.shape[1], X.shape[1]))
        for k in range(self.n_comp):
            diff = X - self.means[k]
            weighted_diff = responsMat[:, k:k+1] * diff
            self.cov[k] = (weighted_diff.T @ diff) / (Nk[k] + self.tolerance)
            # singular matrix 계산을 방지하기 위해
            self.cov[k] += np.eye(X.shape[1]) * self.tolerance

    def logLikelihood(self, X):
        # 로그 우도 계산
        n_samples = X.shape[0]
        likelihood = np.zeros((n_samples, self.n_comp))
        
        for k in range(self.n_comp):
            rv = multivariate_normal(self.means[k], self.cov[k])
            likelihood[:, k] = self.weights[k] * rv.pdf(X)
        
        return np.log(likelihood.sum(axis=1) + self.tolerance).sum()

    def fit(self, X):
        # EM 알고리즘을 사용하여 모델 학습
        # 파라미터 초기화
        self.Initialize(X)
        
        prevllhood = -np.inf
        
        for itr in range(self.max_itr):

            # e-step
            resposMat = self.eStep(X)
            
            # m-step
            self.mStep(X, resposMat)
            
            # 로그 우도 계산
            llhood = self.logLikelihood(X)
            self.log_likelihoood.append(llhood)
            
            # 값이 변하지 않으면 종료 -> 수렴
            if abs(llhood - prevllhood) < self.tolerance:
                print(f"완료 (반복: {itr + 1})")
                break
            
            prevllhood = llhood
            
            if (itr + 1) % 10 == 0:
                print(f"반복 {itr + 1}, 로그 우도: {llhood:.4f}")
        
        return self
    
    def predict(self, X):
        # 각 포인트가 포함된 클러스터 예측
        return np.argmax(self.eStep(X), axis =1)
    
    def predictProb(self, X):
        # 각 포인트가 포함된 소속 클러스터 확률 예측
        return self.eStep(X)

def DrawGMM(X, Y, gmm, title="Gaussian Mixture Model"):
    """GMM 시각화"""
    data = np.column_stack((X, Y))
    labels = gmm.predict(data)
    
    # 색상 맵
    colors = plt.cm.rainbow(np.linspace(0, 1, gmm.n_comp))
    
    plt.figure(figsize=(12, 8))
    
    # 데이터 포인트 그리기
    for k in range(gmm.n_comp):
        cluster_data = data[labels == k]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                   color=colors[k], alpha=0.6, 
                   label=f'Cluster {k+1}', s=30)
    
    # 가우시안 중심 그리기
    plt.scatter(gmm.means[:, 0], gmm.means[:, 1], 
               color='black', marker='x', s=200, linewidth=3,
               label='Centroids')
    
    # 가우시안 등고선 그리기
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    positions = np.array([xx.ravel(), yy.ravel()]).T
    
    # 각 성분별 가장 큰 등고선만 표시
    for k in range(gmm.n_comp):
        rv = multivariate_normal(gmm.means[k], gmm.cov[k])
        max_density = rv.pdf(gmm.means[k])  # 최대 밀도 값
        level = 0.1 * max_density  # 최대값의 10% 수준
        
        # 단일 등고선 그리기
        Z = rv.pdf(positions).reshape(xx.shape)
        plt.contour(xx, yy, Z, levels=[level], 
                   colors=colors[k], alpha=0.5, linewidths=2)


    plt.title(f'{title} (K={gmm.n_comp})', fontsize=16)
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# 메인 실행 코드
if __name__ == "__main__":
    # load data
    X, Y = readmatFile()
    
    if X is not None and Y is not None:
        # 데이터 통계 출력
        
        # [1] 산점도 그리기
        Drawplot(X, Y)
        
        # 데이터를 하나의 배열로 결합 (GMM 분석을 위해)
        data = np.column_stack((X, Y))

        Kval = [2, 6, 10]
        
        for k in Kval:
            # GMM 모델 생성 및 학습
            max_itr = 50
            gmm = GMM(n_comp=k, max_itr=max_itr, randomSeed=0)
            gmm.fit(data)

            # 결과 분석
            final_log_likelihood = gmm.log_likelihoood[-1]
            n_samples, n_features = data.shape

            print(f"수렴까지 반복 횟수: {len(gmm.log_likelihoood)}")
            print(f"최종 로그 우도: {final_log_likelihood:.2f}")
            print(f"클러스터 가중치: {gmm.weights}")

            # [2] GMM시각화
            DrawGMM(X, Y, gmm, title=f"GMM {max_itr}")

    else:
        print("올바른 데이터가 없음.")
