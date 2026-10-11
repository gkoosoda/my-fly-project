import numpy as np
import pandas as pd
from scipy import sparse


class FlyBrain:
    """
    Коннектом дрозофилы: 80 нейронов, фиксированные связи.
    Leaky tanh динамика. НЕ обучается.
    """

    def __init__(self, weights_file, annotations_file,
                 n_neurons=80, n_inputs=12, seed=42):
        np.random.seed(seed)
        print("Загрузка аннотаций...")
        all_annotations = pd.read_feather(annotations_file)
        print("Загрузка весов...")
        all_weights = pd.read_feather(weights_file)

        # --- 1. Ищем зрительные нейроны (сенсорный вход) ---
        visual_mask = all_annotations['type'].str.contains(
            'LC|L1|L2|L3|Mi|visual|optic', case=False, na=False
        )
        visual_ids = set(all_annotations[visual_mask]['bodyId'])

        # Берём n_inputs нейронов с наибольшим числом связей
        weights_count = all_weights[
            all_weights['body_pre'].isin(visual_ids)
        ]['body_pre'].value_counts()
        top_visual_ids = list(weights_count.index[:n_inputs])

        if len(top_visual_ids) < n_inputs:
            remaining = [b for b in visual_ids if b not in set(top_visual_ids)]
            top_visual_ids.extend(remaining[:n_inputs - len(top_visual_ids)])

        # --- 2. Остальные нейроны ---
        selected_body_ids = list(top_visual_ids)
        other_ids = [b for b in all_annotations['bodyId']
                     if b not in set(selected_body_ids)]
        selected_body_ids.extend(other_ids[:n_neurons - len(selected_body_ids)])

        self.annotations = all_annotations[
            all_annotations['bodyId'].isin(set(selected_body_ids))
        ].reset_index(drop=True)
        self.num_neurons = len(self.annotations)
        self.body_to_idx = {
            body_id: idx for idx, body_id in enumerate(self.annotations['bodyId'])
        }

        # Индексы входных нейронов
        self.input_neurons = [
            self.body_to_idx[b] for b in top_visual_ids if b in self.body_to_idx
        ]
        self.n_inputs = len(self.input_neurons)

        # --- 3. Матрица весов ---
        valid_body_ids = set(self.annotations['bodyId'])
        self.weights = all_weights[
            all_weights['body_pre'].isin(valid_body_ids) &
            all_weights['body_post'].isin(valid_body_ids)
        ].copy()

        rows = self.weights['body_pre'].map(self.body_to_idx).values.astype(int)
        cols = self.weights['body_post'].map(self.body_to_idx).values.astype(int)
        data = self.weights['weight'].values.astype(float)

        coo = sparse.coo_matrix(
            (data, (rows, cols)), shape=(self.num_neurons, self.num_neurons)
        )
        dense_weights = coo.toarray().astype(float)

        # Спектральная норма -> стабильная динамика
        norm = np.linalg.norm(dense_weights, ord=2)
        if norm > 0:
            self.weight_matrix = (dense_weights / norm) * 1.0
        else:
            self.weight_matrix = dense_weights

        # --- 4. Состояние ---
        self.activity = np.zeros(self.num_neurons)
        self.leak = 0.7

        print(f"Коннектом загружен! Нейронов: {self.num_neurons}, "
              f"входных: {self.n_inputs}")

    def reset(self):
        self.activity.fill(0.0)

    def step(self, stimuli):
        """stimuli: (n_inputs,) -> возвращает (num_neurons,)"""
        n = min(len(stimuli), self.n_inputs)
        in_idx = self.input_neurons[:n]

        self.activity[in_idx] = stimuli[:n]

        raw_next = self.activity @ self.weight_matrix
        new_activity = (1.0 - self.leak) * np.tanh(raw_next) + self.leak * self.activity
        new_activity[in_idx] = stimuli[:n]
        self.activity = new_activity

        return self.activity.copy()

    def get_activity(self):
        return self.activity.copy()