import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# data.mat 파일 불러오기
data = loadmat("data_python.mat")

# 데이터 추출
X = data["X"]  # 특성 데이터 (400x2)
T = data["T"]  # 클래스 레이블 (400x1)

# 클래스별로 데이터 분리
class1_idx = T[:, 0] == 1
class_negative_idx = T[:, 0] == -1

class1 = X[class1_idx]
class_negative = X[class_negative_idx]

# 산점도 그리기
plt.figure(figsize=(10, 8))
plt.scatter(
    class1[:, 0], class1[:, 1], c="lightpink", marker="o", label="Class 1", alpha=0.7
)
plt.scatter(
    class_negative[:, 0],
    class_negative[:, 1],
    c="skyblue",
    marker="x",
    label="Class -1",
    alpha=0.7,
)

plt.title("data.mat")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# ==================== Multi Layer Perceptron ====================
"""
input_size : 입력 뉴런 2개
hidden_size : 은닉 뉴런 5개
output_size : 출력 뉴런 1개
learning_rate : 0.1
"""
class MultiLayerPerceptron:
    def __init__(self, input_size=2, hidden_size=5, output_size=1, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        """
        # 가중치 초기화 (작은 랜덤 값으로 초기화)
        # W1, b1: 입력층 -> 은닉층 weight, bias
        # W2, b2: 은닉층 -> 출력층 weight, bias
        """
        np.random.seed(0)
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.5
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.5
        self.b2 = np.zeros((1, self.output_size))

        # leakyrelu 사용 시 He 초기화 부분
        # self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2 / self.input_size)
        # self.b1 = np.zeros((1, self.hidden_size))
        # leakyReLu 사용시 가중치 미세 조정
        # self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.1
        # self.b2 = np.zeros((1, self.output_size))

        # 학습 과정 기록용
        self.loss_history = []
        self.accuracy_history = []
        self.error_rate_history = []
        self.early_stop_training = False
        self.stopping_epoch = None
        self.stopping_reason = ""

    # sigmoid 함수는 사용하지 않음
    # def sigmoid(self, x):
    #     # sigmoid activation function
    #     x = np.clip(x, -500, 500)  # 오버플로우 방지
    #     return 1 / (1 + np.exp(-x))

    # def sigmoid_derivative(self, x):
    #     # sigmoid derivative
    #     return x * (1 - x)

    def tanh(self, x):
        # tanh 활성화 함수
        return np.tanh(x)

    def tanh_derivative(self, x):
        # tanh 함수의 도함수
        return 1 - x**2

    def relu(self, x):
        # ReLU 활성화 함수
        return np.maximum(0, x)

    def relu_derivative(self, x):
        # ReLU 도함수 (x > 0이면 1, 아니면 0)
        return (x > 0).astype(float)

    def leaky_relu(self, x, alpha=0.01):
        # Leaky ReLU 활성화 함수
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        # Leaky ReLU 함수의 도함수
        return np.where(x > 0, 1, alpha)

    def forward(self, X):
        """
        # forward propagation
        z1 : 은닉층 입력
        z2 : 출력층 입력
        a1 : 은닉층 활성화
        a2 : 출력층 활성화
        출력층 활성함수는 1, -1을 출력해야 하므로 tanh사용
        """

        # 입력층 -> 은닉층
        self.z1 = np.dot(X, self.W1) + self.b1
        # self.a1 = self.leaky_relu(self.z1)  # 은닉층 leakyReLu 사용
        self.a1 = self.tanh(self.z1)  # 은닉층 tanh 사용

        # 은닉층 -> 출력층
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.tanh(self.z2)  # 출력층 tanh 사용 (-1, 1 출력을 위해)

        return self.a2

    def forward_ReLu(self, X):
        """
        # forward propagation using Leaky ReLU
        z1 : 은닉층 입력
        z2 : 출력층 입력
        a1 : 은닉층 활성화
        a2 : 출력층 활성화
        출력층 활성함수는 1, -1을 출력해야 하므로 tanh사용
        """

        # 입력층 -> 은닉층
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.leaky_relu(self.z1)  # 은닉층 leakyReLu 사용

        # 은닉층 -> 출력층
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.tanh(self.z2)  # 출력층 tanh 사용 (-1, 1 출력을 위해)

        return self.a2

    def backward(self, X, y, output):
        """
        # back propagation
        dz2 : 출력 오차
        dW1, dW2 : 각 층 가중치 기울기기
        """
        m = X.shape[0]  # 샘플 수

        # 출력층 오차
        dz2 = output - y
        dW2 = (1 / m) * np.dot(self.a1.T, dz2)
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)

        # 은닉층 오차
        da1 = np.dot(dz2, self.W2.T)
        # dz1 = da1 * self.leaky_relu_derivative(self.z1) # leakyReLu 사용시
        dz1 = da1 * self.tanh_derivative(self.a1)
        dW1 = (1 / m) * np.dot(X.T, dz1)
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)

        # 가중치 업데이트(경사 하강법 사용)
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def backward_ReLu(self, X, y, output):
        """
        # back propagation using leaky ReLu
        dz2 : 출력 오차
        dW1, dW2 : 각 층 가중치 기울기기
        """
        m = X.shape[0]  # 샘플 수

        # 출력층 오차
        dz2 = output - y
        dW2 = (1 / m) * np.dot(self.a1.T, dz2)
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)

        # 은닉층 오차
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.leaky_relu_derivative(self.z1)  # leakyReLu 사용시
        dW1 = (1 / m) * np.dot(X.T, dz1)
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)

        # 가중치 업데이트(경사 하강법 사용)
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def compute_loss(self, y_true, y_pred):
        # MSE 계산
        return np.mean((y_pred - y_true) ** 2)

    def compute_accuracy(self, y_true, y_pred):
        # 정확도 계산
        predictions = np.sign(y_pred)  # -1 또는 1로 변환
        # 실제 값과 비교
        return np.mean(predictions == y_true)

    def train(
        self,
        X,
        y,
        epochs=1000,
        print_every=100,
        error_threshold=0.05,
        early_stopping=False,
        training_model="tanh",
    ):
        """
        # 모델 학습
        # forward -> backward

        early stopping
        loss <= error_threshold

        training model : tanh, leaky ReLu
        epoch마다 if조건 검색하는게 비효율적이라 외부에서 조건 처리리
        """

        if training_model == "tanh":

            for epoch in range(epochs):

                # forward propagation
                output = self.forward(X)
                # back propagation
                self.backward(X, y, output)
                    
                # epoch마다 loss, accuracy 계산
                loss = self.compute_loss(y, output)
                accuracy = self.compute_accuracy(y, output)
                error_rate = 1 - accuracy  # 오류율

                self.loss_history.append(loss)
                self.accuracy_history.append(accuracy)
                self.error_rate_history.append(error_rate)

                if (epoch + 1) % print_every == 0:
                    print(
                        f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
                    )

                # early stopping option
                if early_stopping and error_rate <= error_threshold:
                    self.early_stop_training = True
                    self.stopping_epoch = epoch + 1
                    print(f"\n=== Early Stop ===")
                    print(
                        f"에포크 {self.stopping_epoch}에서 오류율 {error_rate:.4f} < {error_threshold}"
                    )
                    break

        elif training_model == "ReLu":
            for epoch in range(epochs):
                # forward propagation using leaky ReLU
                output = self.forward_ReLu(X)

                # back propagation using leaky ReLU
                self.backward_ReLu(X, y, output)
                # epoch마다 loss, accuracy 계산

                loss = self.compute_loss(y, output)
                accuracy = self.compute_accuracy(y, output)
                error_rate = 1 - accuracy  # 오류율

                self.loss_history.append(loss)
                self.accuracy_history.append(accuracy)
                self.error_rate_history.append(error_rate)
            
                if (epoch + 1) % print_every == 0:
                    print(
                        f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
                    )

                # early stopping option
                if early_stopping and error_rate <= error_threshold:
                    self.early_stop_training = True
                    self.stopping_epoch = epoch + 1
                    print(f"\n=== Early Stop ===")
                    print(
                        f"에포크 {self.stopping_epoch}에서 오류율 {error_rate:.4f} < {error_threshold}"
                    )
                    break

        if not self.early_stop_training:
            self.stopping_epoch = epochs
            if early_stopping:
                print(f"{epochs} epoch 오류율 {error_rate}")
            else:
                print(f"학습 완료")

    def predict(self, X):
        # 예측 함수
        output = self.forward(X)
        return output  # 연속적인 값으로 변환

    def predict_class(self, X):
        # 클래스 예측
        # forward 후 결과를 1, -1 값으로 출력
        output = self.forward(X)
        return np.sign(output)  # -1 또는 1로 변환

    def training_summary(self):
        final_error_rate = (
            self.error_rate_history[-1] if self.error_rate_history else 1.0
        )
        return {
            "stopped_early": self.early_stop_training,
            "stopping_epoch": self.stopping_epoch,
            "final_error_rate": final_error_rate,
            "final_accuracy": (
                self.accuracy_history[-1] if self.accuracy_history else 0.0
            ),
            "final_loss": self.loss_history[-1] if self.loss_history else float("inf"),
        }


# ==================== Draw Result Data ====================
def Draw_Training_Result(mlp: MultiLayerPerceptron, subject = ""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # loss
    ax1.plot(mlp.loss_history)
    ax1.set_title(f"Training_Loss {subject}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    # accuracy
    ax2.plot(mlp.accuracy_history)
    ax2.set_title(f"Training_Accuracy {subject}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def Draw_Result(mlp: MultiLayerPerceptron, predictions, subject = ""):
    plt.figure(figsize=(12, 5))

    # origin data
    plt.subplot(1, 2, 1)
    plt.scatter(
        class1[:, 0], class1[:, 1], c="skyblue", marker="o", label="Class 1", alpha=0.7
    )
    plt.scatter(
        class_negative[:, 0],
        class_negative[:, 1],
        c="lightpink",
        marker="x",
        label="Class -1",
        alpha=0.7,
    )
    plt.title(f"Original Data {subject}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)

    # predic results
    plt.subplot(1, 2, 2)
    pred_class_1_indices = predictions[:, 0] == 1
    pred_class_minus_1_indices = predictions[:, 0] == -1
    plt.scatter(
        X[pred_class_1_indices, 0],
        X[pred_class_1_indices, 1],
        c="dodgerblue",
        marker="o",
        label="Predicted Class 1",
        alpha=0.7,
    )
    plt.scatter(
        X[pred_class_minus_1_indices, 0],
        X[pred_class_minus_1_indices, 1],
        c="deeppink",
        marker="x",
        label="Predicted Class -1",
        alpha=0.7,
    )
    plt.title(f"Predict results {subject}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ==================== Draw Dicision BD function ====================
def Draw_Decision_Boundary(
    mlp,
    X,
    T,
    X_mean,
    X_std,
    test_inputs=[],
    test_outputs=[],
    test_labels=[],
    test_classes=[],
    subject =''
):

    # meshgrid ([x, y] = meshgrid([-4:0.1:10], [-4:0.1:10]))
    x_min, x_max = -4, 10
    y_min, y_max = -4, 10
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # meshgrid normalize
    m_points = np.c_[xx.ravel(), yy.ravel()]
    m_points_normalized = (m_points - X_mean) / X_std

    # 각 point에 대한 predict
    Z = mlp.predict(m_points_normalized)
    Z = Z.reshape(xx.shape)

    # draw graph
    plt.figure(figsize=(12, 10))

    # draw decision bd
    plt.contourf(
        xx, yy, Z, levels=2, alpha=0.3, colors=["#808080", "#C0C0C0"], extend="both"
    )
    contours = plt.contour(
        xx, yy, Z, levels=[0], colors="darkviolet", linewidths=3, linestyles="--"
    )

    # origin data
    class1 = T[:, 0] == 1
    class_negative = T[:, 0] == -1

    plt.scatter(
        X[class1, 0],
        X[class1_idx, 1],
        c="lightpink",
        marker="o",
        label="Class 1",
        alpha=0.7,
    )
    plt.scatter(
        X[class_negative, 0],
        X[class_negative_idx, 1],
        c="skyblue",
        marker="x",
        label="Class -1",
        alpha=0.7,
    )

    if len(test_inputs) != 0:
        test_col = ["red", "brown", "orangered", "tomato", "coral"]
        for i, (point, output, label, predict_cls) in enumerate(
            zip(test_inputs, test_outputs, test_labels, test_classes)
        ):
            plt.scatter(
                point[0],
                point[1],
                c=test_col[i],
                marker="s",
                s=100,
            )

            plt.annotate(
                f"{int(predict_cls.item())}",
                (point[0], point[1]),
                xytext=(10, 0),
                textcoords="offset points",
                # bbox=dict(boxstyle='round,pad=0.3', facecolor=test_col[i], alpha=0.7),
                fontsize=10,
                fontweight="bold",
            )

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("X")
    plt.ylabel("Y")
    if len(test_inputs) != 0:
        plt.title(f"Decision Boundary of MLP with Test_input {subject}")
    else:
        plt.title(f"Decision Boundary of MLP {subject}")

    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.colorbar(label='Output Value')
    plt.show()

# ==================== Print Summary ====================
def Print_Summary(mlp, final_accuracy, subject = "mlp"):
    print(f"\n=== {subject} Summary ===")
    print(f"Network Structure: {mlp.input_size}-{mlp.hidden_size}-{mlp.output_size}")
    print(f"learning rate: {mlp.learning_rate}")
    print(f"final loss: {mlp.loss_history[-1]:.4f}")
    print(f"final acc: {final_accuracy:.4f}")
    print(f"total training epoch: {len(mlp.loss_history)}")

# ==================== Data Postprocessing ====================
# Normalize
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_normalized = (X - X_mean) / X_std


# ==================== Model Training ====================
"""
# 다층 퍼셉트론 생성 및 학습
# leakyReLu 쓸 경우 leraning rate = 0.01로 줄이기
# input : 입력
# hidden : 은닉
# output : 출력
"""
mlp = MultiLayerPerceptron(input_size=2, hidden_size=5, output_size=1, learning_rate=0.1)
mlp.train(X_normalized, T, epochs=2000, print_every=200)

# 최종 예측 및 성능 평가
predictions = mlp.predict_class(X_normalized)
final_accuracy = mlp.compute_accuracy(T, predictions)
print(f"\n최종 정확도: {final_accuracy:.4f}")

# hidden layer = 3 경우
# Activation function : tanh
mlp_tanh = MultiLayerPerceptron(
    input_size=2, hidden_size=3, output_size=1, learning_rate=0.1
)
mlp_tanh.train(X_normalized, T, epochs=2000, print_every=200, early_stopping= True)
predic_tanh = mlp_tanh.predict_class(X_normalized)
final_accuracy_tanh = mlp_tanh.compute_accuracy(T, predic_tanh)

# Activation function : Leaky ReLu
mlp_relu = MultiLayerPerceptron(
    input_size=2, hidden_size=3, output_size=1, learning_rate=0.1
)
mlp_relu.train(X_normalized, T, training_model="ReLu", epochs=2000, print_every=200, early_stopping= True)
predic_relu = mlp_relu.predict_class(X_normalized)
final_accuracy_relu = mlp_relu.compute_accuracy(T, predic_relu)


# ==================== 4-4 input data ====================
input_points = np.array(
    [
        [0, 0],  # 데이터 1
        [6, 6],  # 데이터 2
        [0, 6],  # 데이터 3
        [6, 0],  # 데이터 4
        [3, 2],  # 데이터 5
    ]
)

input_labels = ["(0,0)", "(6,6)", "(0,6)", "(6,0)", "(3,2)"]

# input data normalize
input_points_normalized = (input_points - X_mean) / X_std

# input datas results
output_points = mlp.predict(input_points_normalized)
output_classes = mlp.predict_class(input_points_normalized)

output_points_tanh = mlp_tanh.predict(input_points_normalized)
output_classes_tanh = mlp_tanh.predict_class(input_points_normalized)

output_points_relu = mlp_relu.predict(input_points_normalized)
output_classes_relu = mlp_relu.predict_class(input_points_normalized)

# ==================== Training Results ====================
Draw_Training_Result(mlp)
Draw_Training_Result(mlp_tanh, subject="mlp_tanh")
Draw_Training_Result(mlp_relu, subject="mlp_leaky ReLu")

# ==================== Result Visualize ====================
Draw_Result(mlp, predictions=predictions)

# tanh
Draw_Result(mlp_tanh, predictions=predic_tanh, subject="tanh")

# leakyReLu
Draw_Result(mlp_relu, predictions=predic_relu, subject="leaky ReLu")

# ==================== Draw Dicision BD (with Test Inputs)====================
Draw_Decision_Boundary(mlp, X, T, X_mean, X_std)

# 4-4 데이터 포함
Draw_Decision_Boundary(
    mlp, X, T, X_mean, X_std, input_points, output_points, input_labels, output_classes
)

# tanh
Draw_Decision_Boundary(
    mlp_tanh, X, T, X_mean, X_std, input_points, output_points_tanh, input_labels, output_classes_tanh, subject="tanh"
)

# leakyReLU
Draw_Decision_Boundary(
    mlp_relu, X, T, X_mean, X_std, input_points, output_points_relu, input_labels, output_classes_relu, subject="leaky ReLu"
)

# ==================== Summary ====================
Print_Summary(mlp, final_accuracy=final_accuracy)
Print_Summary(mlp_tanh, final_accuracy=final_accuracy_tanh, subject="mlp_tanh")
Print_Summary(mlp_relu, final_accuracy=final_accuracy_relu, subject="mlp_leakyReLu")

# # ==================== Input Points Summary ====================
# 4-4 뉴런에서의 출력값 계산
print("\n=== 뉴런에서의 출력값 계산 ===")
print("=" * 10)
for i, (point, output, predicted_class, label) in enumerate(zip(input_points, output_points, output_classes, input_labels)):
    print(f"데이터 {i+1}: 입력 {label}")
    print(f"출력 뉴런 값: {output[0]:.6f}")
    print(f"예측 클래스: {int(predicted_class[0])}")
    print(f"클래스 해석: {'Class 1' if predicted_class[0] > 0 else 'Class -1'}")
    print("-" * 10)

