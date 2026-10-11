"""
Экспорт эпизода FlySnake в формат fly-connectome-template.
Запуск: python export_for_viewer.py
"""

import json
import numpy as np
from pathlib import Path
from brain import FlyBrain
from readout import Readout
from snake_env import SnakeEnv


# --- Пути ---
ROOT = Path(__file__).parent.resolve()
ATLAS = ROOT.parent / "public" / "data" / "brain-atlas"
OUTPUT = ROOT.parent / "public" / "examples" / "flysnake-replay.json"


def load_visible_ids():
    ids = np.fromfile(ATLAS / "ids.bin", dtype="<u4")
    groups = np.fromfile(ATLAS / "groups.bin", dtype="u1")
    return set(ids[groups < 3].tolist())


def find_file(pattern):
    for p in ROOT.rglob(pattern):
        return p
    for p in ATLAS.rglob(pattern):
        return p
    raise FileNotFoundError(f"Не найден {pattern}")


def record_episode(brain, readout, env, visible_ids, max_frames=2000):
    """
    Прогоняет эпизод и собирает активность видимых нейронов.
    Возвращает frames: [{"time": t, "values": [[bodyId, v], ...]}, ...]
    """
    env.reset()
    brain.reset()
    frames = []
    t = 0

    body_ids = brain.annotations['bodyId'].tolist()

    while True:
        stimuli = env.get_stimuli()
        activity = brain.step(stimuli)

        # Выбираем действие
        action, _ = readout.get_action(activity, epsilon=0.0)

        # --- Собираем активность видимых нейронов ---
        values = []
        for idx, body_id in enumerate(body_ids):
            if body_id not in visible_ids:
                continue
            # Нормализация: активность в [-1, 1] → [0, 1]
            v = float(np.clip((activity[idx] + 1.0) / 2.0, 0.0, 1.0))
            if v > 0.01:  # пропускаем почти-нули для компактности
                values.append([int(body_id), round(v, 4)])

        frames.append({"time": float(t), "values": values})
        t += 1

        # Шаг игры
        _, _, done, _ = env.step(action)

        if done or t >= max_frames:
            break

    return frames


def main():
    print("=" * 60)
    print("Экспорт FlySnake в формат fly-connectome-template")
    print("=" * 60)

    # --- Загружаем видимые ID ---
    visible_ids = load_visible_ids()
    print(f"Видимых ID в атласе: {len(visible_ids)}")

    # --- Загружаем мозг ---
    annotations_path = find_file("body-annotations*.feather")
    weights_path = find_file("connectome-weights*.feather")
    print(f"Аннотации: {annotations_path}")
    print(f"Веса: {weights_path}")

    brain = FlyBrain(
        weights_file=str(weights_path),
        annotations_file=str(annotations_path),
        n_neurons=80,
        n_inputs=12,
    )

    # --- Проверка пересечения ---
    your_ids = set(brain.annotations['bodyId'].tolist())
    overlap = your_ids & visible_ids
    print(f"Твоих нейронов видно: {len(overlap)} из {len(your_ids)}")

    # --- Загружаем модель ---
    model_path = None
    for candidate in ["fly_snake_model.npz", "fly_snake_best.npz",
                      "fly_snake_checkpoint.npz"]:
        p = ROOT / candidate
        if p.exists():
            model_path = p
            break

    if model_path is None:
        raise FileNotFoundError(
            "Нет модели! Запусти train.py сначала."
        )

    print(f"Модель: {model_path}")
    data = np.load(model_path)

    readout = Readout(input_dim=brain.num_neurons, hidden_dim=24, output_dim=3)
    readout.set_params(data['readout_params'])

    # --- Прогоняем эпизод ---
    env = SnakeEnv(grid_size=20, max_steps=200)
    print("\nЗапись эпизода...")
    frames = record_episode(brain, readout, env, visible_ids)
    print(f"Записано кадров: {len(frames)}")
    print(f"Очки: {env.score}")

    # --- Формируем JSON ---
    replay = {
        "version": 1,
        "dataset": "male-cns:v1.0",
        "source": {
            "kind": "predicted",
            "name": "FlySnake CEM — hidden=24",
            "normalization": "(tanh_activity + 1) / 2, clamped to [0, 1]",
        },
        "frames": frames,
    }

    # --- Сохраняем ---
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(replay, f, separators=(",", ":"))

    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"\n✅ Сохранено: {OUTPUT}")
    print(f"   Размер: {size_mb:.2f} MB")

    if size_mb > 10:
        print("⚠️ Файл больше 10 MB! Viewer отклонит.")
        print("   Решение: уменьшить max_frames или поднять порог v > 0.01")
    else:
        print(f"   Viewer примет файл (лимит 10 MB)")

    # --- Статистика ---
    total_values = sum(len(f["values"]) for f in frames)
    avg_values = total_values / len(frames)
    print(f"   Среднее число активных нейронов в кадре: {avg_values:.1f}")


if __name__ == "__main__":
    main()