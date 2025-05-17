import h5py  # COIL20.mat 파일 불러오기
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg


def readmatFile(path="COIL20.mat"):
    with h5py.File(path, "r") as f:
        # 학습 데이터만 사용
        X = np.array(f["X"])
        Y = np.array(f["Y"])

        # MATLAB은 열 우선 순서 이고 python은 행 우선 순서 => Transpose 해줘야 함.
        X = X.T
        Y = Y.T
        Y = Y.ravel()  # flatten 작업

        print(f"data : {X.shape}, label : {Y.shape}")
        print(f"number of class : {len(np.unique(Y))}")

    return X, Y


def PCA(datas, m_comp=2):
    """
    datas: 입력 데이터 배열, 형태 ->(N, feature_cnt)
    m_comp: 추출할 주성분 개수 => 현재는 2차원

    Returns:
    X_pca: 변환된 데이터, 형태 (N, m_comp)
    comp_vec: 주성분 벡터, 형태 (m_comp, feature_cnt)
    exp_var: explained variance
    exp_ratio: explained variance ratio
    """
    x_center = datas - np.mean(datas, axis=0)  # 중앙화
    # m_comp = 2

    # cov mat
    # (1) 문제에서 사용했던 PCA 함수 사용.
    cov_mat = np.cov(x_center, rowvar=False)

    # cov mat => eigen val, eigen vec
    # egn_val, egn_vec = np.linalg.eigh(cov_mat) # 실수값만 반환
    egn_val, egn_vec = linalg.eig(cov_mat)

    # eigen val 내림차순 정렬
    idx = np.argsort(egn_val)[::-1]
    egn_val = egn_val[idx]
    egn_vec = egn_vec[:, idx]

    """
    # # eigenval -> max 2개 선택 (2차원 projection 하기 위해)
    # # np.linalg.eigh 는 오름차순 정렬이므로 맨 뒤의 max 두개 선택
    # comp_vec = egn_vec[:, -m_comp:]
    
    # # explained var
    # exp_var = egn_val[-m_comp:]
    # total_var = np.sum(egn_val)
    # exp_ratio = exp_var / total_var
    
    오름차순이라 맨 뒤의 두개 선택시 내림차순 정렬 후 선택하는 값과 vector 순서가 달라지므로
    결과가 원점 대칭된 형태로 나타나기 때문에 통일성을 위해 내림차순 정렬로 변경.
    """

    # projection dim 수만큼 주성분 벡터 선택
    comp_vec = egn_vec[:, :m_comp]

    # projection
    x_pca = np.dot(x_center, comp_vec)

    return x_pca


def PCA95(datas):
    """
    95% 보존 PCA
    datas: 입력 데이터 배열, 형태 ->(N, feature_cnt)

    Returns:
    X_pca: 변환된 데이터, 형태 (N, m_comp)
    comp_vec: 주성분 벡터, 형태 (m_comp, feature_cnt)
    exp_var: explained variance
    exp_ratio: explained variance ratio
    """
    # x_center = datas - np.mean(datas, axis=0)
    x_center = (datas - np.mean(datas, axis=0)) / np.std(datas, axis=0) # 표준화
    # Min-Max 정규화 (0-1 스케일링)
    # min_vals = np.min(datas, axis=0)
    # max_vals = np.max(datas, axis=0)
    # x_center = (datas - min_vals) / (max_vals - min_vals + 1e-10)  # 0으로 나누기 방지

    # cov
    cov_mat = np.cov(x_center, rowvar=False)

    # eigen value, eigen vec
    # egn_val, egn_vec = np.linalg.eigh(cov_mat)
    egn_val, egn_vec = linalg.eigh(cov_mat) #실수값만 채택

    # eigen val 내림차순 정렬
    idx = np.argsort(egn_val)[::-1]
    egn_val = egn_val[idx]
    egn_vec = egn_vec[:, idx]

    # explained var / ratio
    total_var = np.sum(egn_val)
    exp_ratio = egn_val / total_var
    cum_sum = np.cumsum(exp_ratio)  # 누적 분산

    # 95% 이상으로 주성분 개수 선택
    comp_95 = np.where(cum_sum >= 0.95)[0][0] + 1
    print(f"95% 분산 유지에 필요한 주성분 수: {comp_95}")

    # 선택된 주성분으로 데이터 변환
    comp_vec = egn_vec[:, :comp_95]
    # x_pca = x_center.dot(comp_vec)
    x_pca = np.dot(x_center, comp_vec)

    return x_pca, comp_vec


def LDA(X, y, m_comp=2, alpha=0.01):
    """
    X: 입력 데이터
    y: 클래스 레이블
    m_comp : 추출할 주 성분 수
    alpha: 정규화 파라미터 (0~1 사이 값)
    """
    # 클래스 개수
    classes = np.unique(y)
    classes_cnt = len(classes)

    # 전체 평균
    average = np.mean(X, axis=0)

    # 클래스 간/내 분산 초기화
    Sgm_W = np.zeros((X.shape[1], X.shape[1]))
    Sgm_B = np.zeros((X.shape[1], X.shape[1]))

    for c in classes:
        X_cls = X[y == c]
        X_cls_cnt = X_cls.shape[0]

        # 클래스 평균
        avg = np.mean(X_cls, axis=0)

        # 클래스 내 분산
        cent = X_cls - avg
        Sgm_W += np.dot(cent.T, cent)

        # 클래스 간 분산
        avgDiff = (avg - average).reshape(-1, 1)
        Sgm_B += X_cls_cnt * np.dot(avgDiff, avgDiff.T)

    # 파라미터 조정 : 정규화 강도를 조절
    # alpha값이 클수록 강한 정규화
    Sgm_W_reg = (1 - alpha) * Sgm_W + alpha * np.eye(Sgm_W.shape[0]) * np.trace(
        Sgm_W
    ) / Sgm_W.shape[0]

    # 역행렬 계산
    try:
        Sgm_W_inv = linalg.inv(Sgm_W_reg)
    except linalg.LinAlgError:
        # 역행렬 계산 실패 시 의사역행렬 사용
        Sgm_W_inv = linalg.pinv(Sgm_W_reg)

    # 나머지 부분은 동일...
    matrix = np.dot(Sgm_W_inv, Sgm_B)

    # eigen value, eigen vec
    egn_val, egn_vec = linalg.eig(matrix)

    # 복소수가 나올 수 있으므로 실수 값만 선택
    egnRvalues = np.real(egn_val)
    egnRvectors = np.real(egn_vec)

    # eigen value 내림차순 정렬
    idx = np.argsort(egnRvalues)[::-1]
    egnRvalues = egnRvalues[idx]
    egnRvectors = egnRvectors[:, idx]

    # eigen vector m_comp(= dim)개 선택
    # m_comp = min(m_comp, classes_cnt-1)
    lda_comp = egnRvectors[:, :m_comp]

    # 데이터 변환
    X_lda = np.dot(X, lda_comp)
    
    return X_lda, lda_comp


# PCA 후 LDA 적용
def PCA_PLUS_LDA(X, y, alpha = 0.01):
    # PCA로 95% 정보보존
    # X_pca, pca_comp = PCA95(X)
    X_pca = PCA(X)
    x_pca95, x_comp_95 = PCA95(X)
    
    # LDA 적용
    # X_lda, lda_comp = LDA(X_pca, y, m_comp=2)
    X_lda, lda_comp = LDA(x_pca95, y, alpha=alpha)
    
    return X_pca, X_lda

alpha = 0.095
X, Y = readmatFile('COIL20.mat')
x_pca, x_lda = PCA_PLUS_LDA(X, Y, alpha= alpha)

# 시각화 (PCA와 LDA 결과를 함께 표시)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))

unique_classes = np.unique(Y)
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))

# PCA 결과 시각화
for i, cls in enumerate(unique_classes):
    idx = Y == cls
    ax1.scatter(x_pca[idx, 0], x_pca[idx, 1], color=colors[i], s=10, label=f'data{int(cls)}')

ax1.set_title('COIL20 PCA 2-Dim', fontsize=16)
ax1.grid(True)
ax1.legend(bbox_to_anchor=(1.0, 1), loc='upper left', fontsize=8)

# LDA 결과 시각화
for i, cls in enumerate(unique_classes):
    idx = Y == cls
    ax2.scatter(x_lda[idx, 0], x_lda[idx, 1], color=colors[i], s=10, label=f'data{int(cls)}')

ax2.set_title(f'COIL20 PCA 95% + LDA 2-Dim, alpha = {alpha}', fontsize=16)
ax2.grid(True)
ax2.legend(bbox_to_anchor=(1.0, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.show()