import h5py  # MATLAB 7.3 이상 형식(HDF5 기반) 파일 불러오기
from scipy.io import loadmat  # MATLAB 7.2 이하 버전 파일 불러오기
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.linalg as linalg
from collections import Counter
from sklearn.model_selection import train_test_split


def readmatFile(path="iris_shuffled.mat"):
    try:
        with h5py.File(path, "r") as f:
            print("h5py로 파일 열기 성공!")

            print("파일 키")
            for key in f.keys():
                print(f"- {key} : shape = {f[key].shape}")

            # 학습 데이터만 사용
            X = np.array(f["iris_data"])
            Y = np.array(f["iris_class"])

            print("\n=== 원본 shape ===")
            print(f"X shape (원본): {X.shape}")
            print(f"Y shape (원본): {Y.shape}")

            # MATLAB은 열 우선 순서 이고 python은 행 우선 순서 => Transpose 해줘야 함.
            X = X.T
            Y = Y.T
            Y = Y.ravel()  # flatten 작업

        return X, Y

    # except(OSError, KeyError) as e:
    except Exception as e:
        # h5py로 열 수 없으면 scipy로 열기 시도 (MATLAB 7.2 이하 형식)
        print(f"h5py로 파일을 열 수 없습니다. 오류: {e}")
        print("scipy로 파일을 열어보겠습니다...")

        try:
            # scipy로 파일 열기
            data = loadmat(path)
            print("scipy로 파일 열기 성공!")

            # .mat 파일 내부에 포함된 모든 키 출력
            print("파일에 포함된 키들:", data.keys())

            """
            dict_keys(['__header__', '__version__', '__globals__', 'iris_data', 'iris_class'])
            므로 iris_data, iris_class를 사용한다.

            === origin shape ===
            X shape (origin): (150, 4)
            Y shape (origin): (150, 1)

            === after trs shape ===
            X shape after trs: (4, 150)
            Y shape after trs: (150,)
            
            므로 (150, 4) 형태를 가기도록 X는 transpose 시키지 않아야 함.

            X 데이터가 object 형식으로 되어 있어서 데이터 처리를 해야 함.
            """

            # 'X'와 'Y'가 있는지 확인
            if "iris_data" in data and "iris_class" in data:
                X_origin = data["iris_data"]
                Y_origin = data["iris_class"]

                # X 데이터 변환 
                X = np.zeros((X_origin.shape[0], X_origin.shape[1]), dtype=float)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        # 문자열 값을 추출하고 float으로 변환
                        X[i, j] = float(X_origin[i][j][0])
                        
                # Y 데이터 변환
                Y = np.zeros(Y_origin.shape[0], dtype=int)
                for i in range(Y_origin.shape[0]):
                    Y[i] = int(float(Y_origin[i][0][0]) if isinstance(Y_origin[i][0], np.ndarray) else Y[i][0])


                # 데이터 shape 출력
                # print("\n=== origin shape ===")
                # print(f"X shape (origin): {X.shape}")
                # print(f"Y shape (origin): {Y.shape}")


                # X = X.T
                Y = Y.T
                Y = Y.ravel()  # flatten

                return X, Y

            else:
                print("파일에 'X' 또는 'Y' 키가 없습니다.")
                return None, None

        except Exception as e:
            print(f"scipy로 파일을 열 수 없습니다. 오류: {e}")
            return None, None


def EDistance(x1, x2):
    """Euclidean distance"""
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """학습 데이터 저장"""
        self.X_train = X
        self.y_train = y

    def predict(self, X, include = False):
        """테스트 데이터에 대한 예측 수행"""
        if(include):
            predic = [self.predict_sol(x, True) for x in X]
            return np.array(predic)
        else:
            predic = [self.predict_sol(x, False) for x in X]
            return np.array(predic)

    def predict_sol(self, x, include = False):
        """
        단일 데이터 포인트에 대한 예측
        단, 자기 자신은 제외한다.

        모든 학습 데이터와의 거리 계산 및 자기 자신 식별한다.

        ** 단 비교를 위해 include 버전을 추가하여 계산산.
        """

        nbd_k = [] # 근접 k들

        if(include):
            # 자기자신을 근접 포인트에 포함시킬 경우

            dist = [EDistance(x, x_train) for x_train in self.X_train]

            # 가장 가까운 k개의 근접한 포인트 선별
            nbd_k = np.argsort(dist)[: self.k]
        
        else:
            # 자기자신을 근접 포인트에 포함시키지 않을 경우

            dist = []
            indices = []
            
            for i, x_train in enumerate(self.X_train):
                # 정확히 동일한 데이터 포인트인지 확인 (배열 비교)
                if np.array_equal(x, x_train):
                    # 자기 자신인 경우 제외
                    continue
                else:
                    distance = EDistance(x, x_train)
                    dist.append(distance)
                    indices.append(i)
            
            # 거리와 인덱스를 numpy 배열로 변환
            dist = np.array(dist)
            indices = np.array(indices)
            
            # K값보다 적은 샘플이 남는 경우 샘플 개수를 K로 둔다.
            k = min(self.k, len(dist))
            
            if k == 0:
                '''
                만약 모든 샘플이 자기 자신과 동일한 경우
                => 모든 특성이 동일, 전체 데이터셋에서 가장 흔한 클래스를 반환
                '''
                return Counter(self.y_train).most_common(1)[0][0]
            
            # 가장 가까운 k개의 근접한 포인트 선별
            nearest_indices = np.argsort(dist)[:k]
            nbd_k = indices[nearest_indices]
        
        # 가장 근접한 포인트들의 label들 확인
        labels = [self.y_train[i] for i in nbd_k]
        
        # 가장 많은 레이블을 예측값으로 반환 (다수결)
        most_common = Counter(labels).most_common()

        # 만약 빈도가 같은 클래스들이 여러개면 랜덤으로 선택
        
        # 최대 빈도 클래스들 찾기
        tied_cls = [cls for cls, count in most_common if count == most_common[0][1]]
    
        if len(tied_cls) > 1:
            # 최대 빈도가 여러개인 경우 랜덤으로 선택
            return random.choice(tied_cls)
        else:
            return tied_cls[0]

 

# 오분류율 계산 함수
def calculate_misclassification_rate(y_true, y_pred):
    """오분류율 계산"""
    return np.sum(y_true != y_pred) / len(y_true)


X, Y = readmatFile()

# 데이터 분할 (학습: 120개, 테스트: 30개)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=30/150, random_state=0)

Kval = [5, 10, 15, 20, 25, 30]

# 각 K값에 대한 오분류율 계산 dic
misclassification_rates_includes = {}
misclassification_rates_excludes = {}
misclassification_rates_subset = {}


# 각 K 값에 대해 KNN 모델 학습 및 오분류율 계산
for k in Kval:
    # KNN 모델 생성
    knn = KNNClassifier(k=k)

    # 모델 학습
    knn.fit(X, Y)

    # 학습 데이터에 대한 예측
    predictions_include = knn.predict(X, True)
    predictions_exclude = knn.predict(X, False)

    # 오분류율 계산
    # 자기자신이 포함된 결과의 오분류율
    misclassification_rates_include = calculate_misclassification_rate(Y, predictions_include)
    misclassification_rates_includes[k] = misclassification_rates_include

    # 자기자신이 포함되지 않은 결과의 오분류율
    misclassification_rates_exclude = calculate_misclassification_rate(Y, predictions_exclude)
    misclassification_rates_excludes[k] = misclassification_rates_exclude

    print(f"K = {k}, 오분류율 = {misclassification_rates_include:.3f} | {misclassification_rates_exclude:.3f}")


for k in Kval:

    '''
    데이터 4:1 분할을 이용한 knn 적용
    과정은 위와 동일
    '''
    knn = KNNClassifier(k=k)
    
    # 모델 학습
    knn.fit(X_train, Y_train)
    
    # 테스트 데이터에 대한 예측
    # 테스트 데이터는 자기자신이 포함되어 있지 않으므로 하나만 테스트 한다. 
    predictions = knn.predict(X_test, True)
    
    # 오분류율 계산
    misclassification_rate = calculate_misclassification_rate(Y_test, predictions)
    misclassification_rates_subset[k] = misclassification_rate
    
    print(f"K = {k}, 4:1 분할 오분류율 = {misclassification_rate:.3f}")


# 결과 표 출력
data = [(k, 
         round(misclassification_rates_includes[k], 3), 
         round(misclassification_rates_excludes[k], 3)) for k in Kval]

columns = ["K", "Including", "Excluding"]

datas = [(k, 
         round(misclassification_rates_includes[k], 3), 
         round(misclassification_rates_excludes[k], 3),
         round(misclassification_rates_subset[k], 3)) for k in Kval]

columnss = ["K", "Including", "Excluding","Test_set"]


# 표 그리기
# 두 개의 서브플롯 생성
fig, axes = plt.subplots(nrows=2, figsize=(5, 6))

# 첫 번째 테이블
axes[0].axis('off')
table1 = axes[0].table(cellText=data, colLabels=columns, loc="center", cellLoc="center", colColours=['#f2f2f2']*3)

# 두 번째 테이블
axes[1].axis('off')
table2 = axes[1].table(cellText=datas, colLabels=columnss, loc="center", cellLoc="center", colColours=['#f2f2f2']*4)

# 표시
plt.tight_layout()
plt.show()