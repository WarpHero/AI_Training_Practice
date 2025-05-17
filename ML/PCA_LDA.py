import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal

# 클래스 2개, sample 각 100개 총 200개
N = 100
totals = N * 2

# 평균 vec
cls1_m = np.array([0, 0])
cls2_m = np.array([0, 5])

# cov vec
cls1_sgm = np.array([[10, 2], [2, 1]])

# random data gen.
cls1_x = multivariate_normal(cls1_m, cls1_sgm, N)
cls2_x = multivariate_normal(cls2_m, cls1_sgm, N)

datas = np.vstack((cls1_x, cls2_x))


# (1)
def Pca_lda_vec_visualize(PCA=False, LDA=False, titl="Sample Data"):
    plt.figure(figsize=(10, 10))

    # 클래스 1 데이터 (red)
    plt.scatter(cls1_x[:, 0], cls1_x[:, 1], c="red", marker="o", s=50, label="Class 1")

    # 클래스 2 데이터 (blue)
    plt.scatter(cls2_x[:, 0], cls2_x[:, 1], c="blue", marker="x", s=50, label="Class 2")

    # 데이터 평균 계산
    mean_X = np.mean(datas, axis=0)

    # 주성분 벡터 스케일링
    scale = 10

    # PCA 표시
    if PCA:
        pca_line = scale * pca_vec
        plt.plot(
            [mean_X[0], mean_X[0] + pca_line[0]],
            [mean_X[1], mean_X[1] + pca_line[1]],
            color="green",
            linewidth=2,
            linestyle="--",
            label="PCA",
        )

    # LDA 벡터 표시
    if LDA:
        lda_line = scale * lda_vec
        plt.plot(
            [mean_X[0], mean_X[0] + lda_line[0]],
            [mean_X[1], mean_X[1] + lda_line[1]],
            color="violet",
            linewidth=2,
            linestyle="--",
            label="LDA",
        )

    # visualize
    plt.xlim(-10, 10)
    plt.ylim(-5, 10)
    plt.title(titl, fontsize=10)
    plt.legend(fontsize=8)
    plt.show()


# (2) PCA LDA
def PCA():
    x_center = datas - np.mean(datas, axis=0)

    # cov
    cov_mat = np.cov(x_center, rowvar=False)

    # eigen value, eigen vec
    egn_val, egn_vec = np.linalg.eigh(cov_mat)

    # eigenval -> max
    max_egn_val = egn_val[-1]
    max_egn_vec = egn_vec[:, -1]

    # 데이터 projection
    x_pca = np.dot(x_center, max_egn_vec)

    return x_pca, max_egn_val, max_egn_vec


def LDA():

    x_center = datas - np.mean(datas, axis=0)

    # 클래스별 평균
    mean1 = np.mean(cls1_x, axis=0)
    mean2 = np.mean(cls2_x, axis=0)

    # 클래스 내 분산
    Sgm1 = np.cov(cls1_x, rowvar=False)
    Sgm2 = np.cov(cls2_x, rowvar=False)

    Sgm = Sgm1 + Sgm2

    # LDA 방향 벡터
    lda_vector = np.linalg.inv(Sgm).dot(mean2 - mean1)

    # normalize
    lda_vector = lda_vector / np.linalg.norm(lda_vector)

    # projection
    x_lda = np.dot(datas, lda_vector)

    return x_lda, lda_vector


def Pca_lda_proj_visualize():
    cls1_idx = np.arange(0, N)
    cls2_idx = np.arange(N, N * 2)

    # 비교 그래프
    fig, (graph1, graph2) = plt.subplots(1, 2, figsize=(18, 9))

    # 1. graph1 : PCA projection
    graph1.scatter(
        x_pca[cls1_idx],
        np.zeros_like(x_pca[cls1_idx]),
        color="red",
        marker="o",
        s=50,
        label="Class 1",
        alpha=0.7,
    )
    graph1.scatter(
        x_pca[cls2_idx],
        np.zeros_like(x_pca[cls2_idx]),
        color="blue",
        marker="x",
        s=50,
        label="Class 2",
        alpha=0.7,
    )

    # PCA 클래스별 평균
    pca_mean1 = np.mean(x_pca[cls1_idx])
    pca_mean2 = np.mean(x_pca[cls2_idx])
    graph1.scatter(pca_mean1, 0, color="darkred", s=100, marker="*", label="Mean Class 1")
    graph1.scatter(pca_mean2, 0, color="darkblue", s=100, marker="*", label="Mean Class 2")

    graph1.set_title("PCA Projection", fontsize=10)
    graph1.set_xlabel("Projection Value", fontsize=8)
    graph1.set_yticks([])
    graph1.legend(fontsize=6)
    graph1.grid(True)

    # 2. graph2 : LDA projection
    graph2.scatter(
        x_lda[cls1_idx],
        np.zeros_like(x_lda[cls1_idx]),
        color="red",
        marker="o",
        s=50,
        label="Class 1",
        alpha=0.7,
    )
    graph2.scatter(
        x_lda[cls2_idx],
        np.zeros_like(x_lda[cls2_idx]),
        color="blue",
        marker="x",
        s=50,
        label="Class 2",
        alpha=0.7,
    )

    # LDA 클래스별 평균
    lda_mean1 = np.mean(x_lda[cls1_idx])
    lda_mean2 = np.mean(x_lda[cls2_idx])
    graph2.scatter(lda_mean1, 0, color="darkred", s=100, marker="*", label="Mean Class 1")
    graph2.scatter(lda_mean2, 0, color="darkblue", s=100, marker="*", label="Mean Class 2")

    # LDA 결정 경계
    lda_threshold = (lda_mean1 + lda_mean2) / 2
    graph2.axvline(
        x=lda_threshold, color="black", linestyle=":", label="Decision Boundary"
    )

    graph2.set_title("LDA Projection", fontsize=10)
    graph2.set_xlabel("Projection Value", fontsize=8)
    graph2.set_yticks([])
    graph2.legend(fontsize=6)
    graph2.grid(True)

    plt.suptitle("Comparison of PCA and LDA Projections", fontsize=12)
    plt.tight_layout()
    plt.show()


x_pca, pca_val, pca_vec = PCA()
x_lda, lda_vec = LDA()

# 산점도와 pca, lda 주 성분 vector 그리기 함수.
Pca_lda_vec_visualize(True, True, "PCA and LDA")

# pca와 lda 주성분 projection 그리기 비교 함수.
Pca_lda_proj_visualize()