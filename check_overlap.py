import numpy as np
import pandas as pd
from pathlib import Path

# --- 1. Загружаем видимые ID ---
root = Path("public/data/brain-atlas")
ids = np.fromfile(root / "ids.bin", dtype="<u4")
groups = np.fromfile(root / "groups.bin", dtype="u1")
visible_ids = set(ids[groups < 3].tolist())
print(f"Видимых ID: {len(visible_ids)}")

# --- 2. Загружаем аннотации ---
annotations_path = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
if not Path(annotations_path).exists():
    # ищем рядом
    for p in Path(".").rglob("body-annotations*.feather"):
        annotations_path = str(p)
        break

print(f"Аннотации: {annotations_path}")
annotations = pd.read_feather(annotations_path)
print(f"Всего аннотаций: {len(annotations)}")

# --- 3. Твои визуальные нейроны (как в brain.py) ---
visual_mask = annotations['type'].str.contains(
    'LC|L1|L2|L3|Mi|visual|optic', case=False, na=False
)
your_visual_ids = set(annotations[visual_mask]['bodyId'].tolist())
print(f"Визуальных нейронов: {len(your_visual_ids)}")

# --- 4. Пересечение с видимыми ---
overlap = your_visual_ids & visible_ids
print(f"\n=== РЕЗУЛЬТАТ ===")
print(f"Визуальных ∩ видимых: {len(overlap)} из {len(your_visual_ids)}")

if len(overlap) == 0:
    print("\n🔴 ПРОБЛЕМА: ни один твой визуальный нейрон не виден!")
    print("   Нужно переобучить модель с нейронами из visible_ids.")
elif len(overlap) < 10:
    print(f"\n⚠️ Мало видимых нейронов: {len(overlap)}")
    print("   Viewer покажет только их, остальные проигнорирует.")
else:
    print(f"\n✅ Хорошо! Viewer покажет {len(overlap)} из твоих нейронов.")

# --- 5. Проверка: сколько из 80 твоих выбранных нейронов видно ---
# Симулируем отбор как в brain.py
weights_path = None
for p in Path(".").rglob("connectome-weights*.feather"):
    weights_path = str(p)
    break

if weights_path:
    print(f"\nВеса: {weights_path}")
    weights = pd.read_feather(weights_path)

    weights_count = weights[
        weights['body_pre'].isin(your_visual_ids)
    ]['body_pre'].value_counts()
    top_visual = list(weights_count.index[:12])

    # Добираем если меньше 12
    if len(top_visual) < 12:
        remaining = [b for b in your_visual_ids if b not in set(top_visual)]
        top_visual.extend(remaining[:12 - len(top_visual)])

    visible_top = [b for b in top_visual if b in visible_ids]
    print(f"Из 12 входных нейронов видно: {len(visible_top)}")

    # Проверяем "остальные" нейроны (не визуальные)
    selected = list(top_visual)
    other = [b for b in annotations['bodyId'] if b not in set(selected)]
    selected.extend(other[:80 - len(selected)])

    selected_visible = [b for b in selected if b in visible_ids]
    print(f"Из всех 80 нейронов модели видно: {len(selected_visible)}")