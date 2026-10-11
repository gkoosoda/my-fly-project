import pygame
import numpy as np
from brain import FlyBrain
from readout import Readout
from snake_env import SnakeEnv

W, H = 500, 500
GS = 20
CELL = W // GS
MAX_STEPS = 200   # <-- как в train.py

BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 150, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)

pygame.init()
screen = pygame.display.set_mode((W, H))
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 16)
pygame.display.set_caption("Fly Snake — Trained Model")


from pathlib import Path

def load_model():
    # Путь к public/data/brain-atlas/ относительно этого скрипта
    ROOT = Path(__file__).parent.resolve()          # snake_project/
    ATLAS = ROOT.parent / "public" / "data" / "brain-atlas"
    
    annotations = ATLAS / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    weights = ATLAS / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    
    print(f"Аннотации: {annotations}")
    print(f"Веса: {weights}")
    
    assert annotations.exists(), f"Не найдено: {annotations}"
    assert weights.exists(), f"Не найдено: {weights}"
    
    brain = FlyBrain(
        weights_file=str(weights),
        annotations_file=str(annotations),
        n_neurons=80,
        n_inputs=12,
    )
    
    # Модель — рядом со скриптом
    model_path = None
    for name in ["fly_snake_checkpoint.npz", "fly_snake_best.npz",
                 "fly_snake_model.npz"]:
        p = ROOT / name
        if p.exists():
            model_path = p
            break
    
    if model_path is None:
        raise FileNotFoundError(
            f"Нет ни одного .npz в {ROOT}\n"
            f"Искал: fly_snake_model.npz, fly_snake_best.npz, "
            f"fly_snake_checkpoint.npz"
        )
    
    print(f"Модель: {model_path}")
    data = np.load(model_path)
    
    hidden_dim = int(data['readout_hidden_dim']) if 'readout_hidden_dim' in data.files else 24
    
    readout = Readout(
        input_dim=brain.num_neurons,
        hidden_dim=hidden_dim,
        output_dim=3,
    )
    readout.set_params(data['readout_params'])
    
    return brain, readout


def draw(env, episode, total_score):
    screen.fill(BLACK)
    for x in range(0, W, CELL):
        pygame.draw.line(screen, GRAY, (x, 0), (x, H), 1)
    for y in range(0, H, CELL):
        pygame.draw.line(screen, GRAY, (0, y), (W, y), 1)

    for i, (x, y) in enumerate(env.body):
        color = GREEN if i == 0 else DARK_GREEN
        rect = pygame.Rect(x * CELL + 1, y * CELL + 1, CELL - 2, CELL - 2)
        pygame.draw.rect(screen, color, rect)

    fx, fy = env.food
    rect = pygame.Rect(fx * CELL + 2, fy * CELL + 2, CELL - 4, CELL - 4)
    pygame.draw.rect(screen, RED, rect)

    info = font.render(
        f"Ep {episode} | Score {env.score} | Total {total_score} | "
        f"Steps {env.total_steps}/{env.MAX_STEPS}",
        True, WHITE
    )
    screen.blit(info, (10, 10))
    pygame.display.flip()


def main():
    print("Загрузка модели...")
    brain, readout = load_model()
    env = SnakeEnv(grid_size=GS, max_steps=MAX_STEPS)

    # Проверка: прогони 5 эпизодов и напечатай score
    print("\nПроверка модели (5 эпизодов, epsilon=0):")
    for i in range(5):
        brain.reset()
        reward, steps = env.run_episode(brain, readout, epsilon=0.0)
        print(f"  Эп {i+1}: score={env.score}, steps={steps}, reward={reward:.2f}")

    episodes = 5
    delay_ms = 80
    paused = False
    total_score = 0

    for ep in range(1, episodes + 1):
        env.reset()
        brain.reset()
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return

            if paused:
                clock.tick(30)
                continue

            stimuli = env.get_stimuli()
            brain.step(stimuli)
            neural_activity = brain.get_activity()
            action, _ = readout.get_action(neural_activity, epsilon=0.0)

            _, _, done, _ = env.step(action)

            draw(env, ep, total_score)
            clock.tick(1000 // delay_ms)

        total_score += env.score
        print(f"Эпизод {ep}: score={env.score}, steps={env.total_steps}")

    print(f"\nИтого за {episodes} эпизодов: {total_score}")

    screen.fill(BLACK)
    end_text = font.render(
        f"Done! Total score: {total_score}. ESC to exit.",
        True, WHITE
    )
    screen.blit(end_text, (50, H // 2))
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                waiting = False
    pygame.quit()


if __name__ == "__main__":
    main()