import numpy as np
import matplotlib.pyplot as plt

# 각 데이터의 수 N = 100
N = 100
np.random.seed(0)

# 평균 벡터
mu1 = np.array([0, 4])
mu2 = np.array([4, 4])
mu3 = np.array([2, 0])

# 공분산 행렬 (모든 클래스가 동일)
cov = np.eye(2)

def CreateRandomDatas(m, cov, samples):
    return np.random.multivariate_normal(m, cov, samples)

def Calc_Mean(datas):
    # total = len(datas)
    total = N # 현재는 모두 N=100 으로 동일하므로
    sumX = sumY = 0

    for p in datas:
        sumX += p[0]
        sumY += p[1]
    
    return [sumX/total, sumY/total]

def euclidean_distance(p1, p2):
    dist = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return np.sqrt(dist)

# cluster 새로 할당
def assign_clusters(data, centroids):
    # 클러스터 별 리스트 생성 후 인덱스 저장
    clusters = [[] for _ in range(K)]
    cluster_indices = []
    
    '''
    현재 포인트들과 모든 중심점 간의 distance 계산 후 가장 가까운 중심점 인덱스 찾음
    포인트를 가까운 클러스터에 추가하고
    포인트가 속한 클러스터의 가장 가까운 인덱스 추가
    '''
    for point in data:
        distances = [euclidean_distance(point, c) for c in centroids]
        min_index = distances.index(min(distances))
        clusters[min_index].append(point)
        cluster_indices.append(min_index)
    
    return clusters, cluster_indices

# 중심점 재계산 함수
def calculate_centroids(clusters, data, current_centroids):
    new_centroids = []
    
    # 클러스터 크기 계산
    cluster_sizes = [len(cluster) for cluster in clusters]
    
    for i, cluster in enumerate(clusters):
        if len(cluster) == 0:  # 빈 군집 처리
            # 빈 클러스터가 아닌 것 중에서 가장 큰 클러스터 찾기
            valid_clusters = [j for j, size in enumerate(cluster_sizes) if size > 0]
            
            # 모든 클러스터가 비어있는 경우 예외처리
            if not valid_clusters:
                # 데이터에서 무작위 포인트 선택
                random_idx = np.random.randint(len(data))
                new_centroids.append(data[random_idx].tolist())
                continue
                
            largest_cluster_idx = max(valid_clusters, key=lambda j: cluster_sizes[j])
            largest_cluster = clusters[largest_cluster_idx]
            largest_centroid = current_centroids[largest_cluster_idx]
            
            # 가장 큰 클러스터에서 중심점과 가장 먼 포인트 찾기
            max_distance = -1
            farthest_point = None
            
            for point in largest_cluster:
                dist = euclidean_distance(point, largest_centroid)
                if dist > max_distance:
                    max_distance = dist
                    farthest_point = point
            
            new_centroids.append(farthest_point)

        else:
            # 정상적인 군집 중심 계산
            sum_x = sum([p[0] for p in cluster])
            sum_y = sum([p[1] for p in cluster])
            new_centroids.append([sum_x / len(cluster), sum_y / len(cluster)])
        
    return new_centroids

# [1] 클래스 1, 2, 3 데이터 생성
class1 = CreateRandomDatas(mu1, cov, N)
class2 = CreateRandomDatas(mu2, cov, N)
class3 = CreateRandomDatas(mu3, cov, N)

# 모든 데이터 합치기
data = np.vstack((class1, class2, class3))

# 원래 클래스 레이블 (나중에 비교를 위해)
true_labels = np.concatenate([np.zeros(N), np.ones(N), 2*np.ones(N)])

# [2] 각 클래스의 평균 계산
mean1 = Calc_Mean(class1)
mean2 = Calc_Mean(class2)
mean3 = Calc_Mean(class3)

# [3] k-means clustering

# 결과 기록용 리스트
centroids_history = [] # 중심점 변화 표시를 위한 리스트
labels_history = []
max_iters = 10
K = 3

# 초기 중심점을 랜덤하게 선택 - 3개 (수정된 부분)
random_indices = np.random.choice(len(data), K, replace=False)
centroids = [data[i].tolist() for i in random_indices]
centroids_history.append(centroids.copy())

# 각 반복마다 Clustering
for iteration in range(max_iters):
    # 군집 할당
    clusters, cluster_indices = assign_clusters(data, centroids)
    labels_history.append(cluster_indices.copy())
    
    # 중심점 재계산 (수정된 부분)
    new_centroids = calculate_centroids(clusters, data, centroids)
    
    # 중심점 변화 확인 (종료 조건, 필요 시)
    if new_centroids == centroids:
        print(f"{iteration+1}번째 centroid 변화 없음.")
        break
    
    centroids = new_centroids
    centroids_history.append(centroids.copy())

# 최종 반복에서의 클러스터 결과
final_clusters, final_indices = assign_clusters(data, centroids)

# 데이터 시각화
# [1]
plt.figure(figsize=(10, 8))
plt.scatter(class1[:, 0], class1[:, 1], c='lightpink', label='Class 1')
plt.scatter(class2[:, 0], class2[:, 1], c='lightgreen', label='Class 2')
plt.scatter(class3[:, 0], class3[:, 1], c='skyblue', label='Class 3')

# [2]
plt.scatter(mean1[0], mean1[1], c='deeppink', marker='s', s=100, label='Class 1 Mean')
plt.scatter(mean2[0], mean2[1], c='forestgreen', marker='s', s=100, label='Class 2 Mean')
plt.scatter(mean3[0], mean3[1], c='dodgerblue', marker='s', s=100, label='Class 3 Mean')

# [3] K-means 결과 중심점 추가
# colors = ['lightgray', 'silver', 'darkgray', 'gray', 'dimgray', 'black']

centroid_paths = [[] for _ in range(K)]
for history in centroids_history:
    for i in range(K):
        centroid_paths[i].append(history[i])

# 중심점 이동 라인으로 표시
colors = ['red', 'green', 'blue']
for i in range(K):
    x = [p[0] for p in centroid_paths[i]]
    y = [p[1] for p in centroid_paths[i]]
    # print(x, y)
    plt.scatter(x, y, c=colors[i], marker='o', s=5, 
                label=f'Cluster {i+1} Centroid')
    plt.plot(x, y, '-', linewidth=1, markersize=8, color=colors[i],
             label=f'Cluster {i+1} Path')
    
    plt.scatter(x[-1], y[-1], s=150, c='black', zorder=10,  marker='x')

plt.title('k-means clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='lower right')
plt.grid(True)
plt.axis('equal')
plt.show()

# 결과 분석
print("\nK-means 알고리즘 최종 결과 분석:")
print("\n최종 군집 중심:")
for k in range(K):
    print(f"클러스터 {k+1} 중심: [{centroids[k][0]:.4f}, {centroids[k][1]:.4f}]")

print("\n실제 데이터 중심:")
print(f"클래스 1 중심: [{mean1[0]:.4f}, {mean1[1]:.4f}]")
print(f"클래스 2 중심: [{mean2[0]:.4f}, {mean2[1]:.4f}]")
print(f"클래스 3 중심: [{mean3[0]:.4f}, {mean3[1]:.4f}]")

print("\n실제 데이터와 중심 거리 차이:")
print(f"클래스 1 차이: {euclidean_distance(centroids[0],mean1):.4f}")
print(f"클래스 2 차이: {euclidean_distance(centroids[1],mean2):.4f}")
print(f"클래스 3 차이: {euclidean_distance(centroids[2],mean3):.4f}")

print("\n최종 군집 크기:")
for k in range(K):
    print(f"클러스터 {k+1}: {final_indices.count(k)}")

# 알고리즘 반복 횟수
print(f"\n총 반복 횟수: {len(centroids_history)-1}")