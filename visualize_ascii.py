"""
ASCII-визуализация змейки + активность мозга в терминале.
Запуск: python visualize_ascii.py
"""
import time
import numpy as np
from pathlib import Path
from brain import FlyBrain
from readout import Readout
from snake_env import SnakeEnv


ROOT = Path(__file__).parent.resolve()


def clear_screen():
    print("\033[H\033[J", end="")


def render_ascii(env, activity, visible_count):
    grid = [['·' for _ in range(env.GS)] for _ in range(env.GS)]
    for x, y in env.body:
        grid[y][x] = 'o'
    hx, hy = env.body[0]
    grid[hy][hx] = '@'
    grid[env.food[1]][env.food[0]] = '*'

    lines = []
    lines.append(f"Score: {env.score} | Steps: {env.total_steps}/{env.MAX_STEPS}")
    lines.append(f"Brain activity: mean={activity.mean():.3f} "
                 f"max={activity.max():.3f} | active neurons: {visible_count}")
    lines.append('┌' + '─' * (env.GS * 2 - 1) + '┐')
    for row in grid:
        lines.append('│' + ' '.join(row) + '│')
    lines.append('└' + '─' * (env.GS * 2 - 1) + '┘')

    # Полоска активности (гистограмма)
    n_bins = 20
    hist, _ = np.histogram(activity, bins=n_bins, range=(-1, 1))
    max_h = max(hist.max(), 1)
    lines.append("\nBrain activity histogram:")
    for i in range(n_bins - 1, -1, -1):
        bar = '█' * int(hist[i] / max_h * 40)
        lo = -1 + i * 0.1
        lines.append(f"{lo:+.1f} │{bar}")

    clear_screen()
    print('\n'.join(lines))


def main():
    def find_file(pattern):
        # Ищем сначала в snake_project, потом во всём проекте
        for p in ROOT.rglob(pattern):
            return p
        for p in ROOT.parent.rglob(pattern):
            return p
        raise FileNotFoundError(f"Не найден {pattern}")

    annotations = find_file("body-annotations*.feather")
    weights = find_file("connectome-weights*.feather")
    print(f"Аннотации: {annotations}")
    print(f"Веса: {weights}")

    brain = FlyBrain(
        weights_file=str(weights),
        annotations_file=str(annotations),
        n_neurons=80, n_inputs=12,
    )

    model = np.load(ROOT / "fly_snake_model.npz")
    readout = Readout(input_dim=80, hidden_dim=24, output_dim=3)
    readout.set_params(model['readout_params'])

    env = SnakeEnv(grid_size=20, max_steps=200)

    for ep in range(5):
        env.reset()
        brain.reset()

        while True:
            stimuli = env.get_stimuli()
            activity = brain.step(stimuli)
            action, _ = readout.get_action(activity, epsilon=0.0)

            visible = int(np.sum(np.abs(activity) > 0.1))
            render_ascii(env, activity, visible)
            time.sleep(0.05)

            _, _, done, _ = env.step(action)
            if done:
                break

        print(f"\nЭпизод {ep + 1}: score={env.score}, "
              f"steps={env.total_steps}")
        time.sleep(2)


if __name__ == "__main__":
    main()