import numpy as np


class Readout:
    """
    MLP 16 → hidden_dim → 3.
    Вход: 80 нейронов коннектома (проекция до 16).
    Выход: 3 логита (turn_left, straight, turn_right).
    """

    def __init__(self, input_dim=80, hidden_dim=12, output_dim=3, seed=42):
        np.random.seed(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Фиксированная проекция 80 -> 16
        self.proj = np.random.randn(input_dim, 16) * 0.1

        # Обучаемые параметры
        self.W1 = np.random.randn(16, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)

        self.param_dim = (
            16 * hidden_dim + hidden_dim +
            hidden_dim * output_dim + output_dim
        )
        print(f"Readout: {self.param_dim} параметров")

    def forward(self, neural_activity):
        x = neural_activity @ self.proj
        h = np.tanh(x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return logits

    def get_params(self):
        return np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten(),
        ])

    def set_params(self, params):
        idx = 0
        w1_size = 16 * self.hidden_dim
        self.W1 = params[idx:idx + w1_size].reshape(16, self.hidden_dim)
        idx += w1_size

        self.b1 = params[idx:idx + self.hidden_dim]
        idx += self.hidden_dim

        w2_size = self.hidden_dim * self.output_dim
        self.W2 = params[idx:idx + w2_size].reshape(self.hidden_dim, self.output_dim)
        idx += w2_size

        self.b2 = params[idx:idx + self.output_dim]

    @staticmethod
    def softmax(logits):
        z = logits - np.max(logits)
        e = np.exp(z)
        return e / (e.sum() + 1e-12)

    def get_action(self, neural_activity, epsilon=0.0):
        logits = self.forward(neural_activity)
        probs = self.softmax(logits)

        if np.random.random() < epsilon:
            action = np.random.randint(0, self.output_dim)
        else:
            action = int(np.argmax(logits))

        return action, probs