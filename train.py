import numpy as np
from brain import FlyBrain
from readout import Readout
from cem import CEMTrainer
from snake_env import SnakeEnv


def evaluate_model(brain, readout, env, n_episodes=100, epsilon=0.0):
    """Честная оценка на N эпизодах."""
    scores = []
    steps_list = []
    rewards = []

    for _ in range(n_episodes):
        reward, steps = env.run_episode(brain, readout, epsilon=epsilon)
        scores.append(env.score)
        steps_list.append(steps)
        rewards.append(reward)

    return {
        "mean_score": float(np.mean(scores)),
        "max_score": int(np.max(scores)),
        "min_score": int(np.min(scores)),
        "std_score": float(np.std(scores)),
        "mean_steps": float(np.mean(steps_list)),
        "mean_reward": float(np.mean(rewards)),
        "scores": scores,
    }


def main():
    print("=" * 60)
    print("Fly Snake CEM — архитектура Fly Dino")
    print("=" * 60)

    # --- 1. Инициализация ---
    brain = FlyBrain(
        weights_file='connectome-weights-male-cns-v1.0-minconf-0.5.feather',
        annotations_file='body-annotations-male-cns-v1.0-minconf-0.5.feather',
        n_neurons=80,
        n_inputs=12,
    )

    readout = Readout(
        input_dim=brain.num_neurons,
        hidden_dim=24,        # было 12, увеличено для лучшей ёмкости
        output_dim=3,
    )

    env = SnakeEnv(grid_size=20, max_steps=200)

    # --- 2. CEM обучение ---
    print("\nНачало CEM обучения...")
    trainer = CEMTrainer(
        readout,
        pop_size=64,
        n_elite=8,
        n_gen=100,            # было 150
        init_sigma=0.7,
        patience = 25,
    )

    trainer.train(env, brain, verbose=True)

    # --- 3. Честная оценка на 100 эпизодах ---
    print("\n" + "=" * 60)
    print("ОЦЕНКА (100 эпизодов, epsilon=0)")
    print("=" * 60)

    results = evaluate_model(brain, readout, env, n_episodes=100, epsilon=0.0)

    print(f"  Средний score:   {results['mean_score']:.2f} ± {results['std_score']:.2f}")
    print(f"  Максимум:        {results['max_score']}")
    print(f"  Минимум:         {results['min_score']}")
    print(f"  Средние шаги:    {results['mean_steps']:.1f}")
    print(f"  Средний reward:  {results['mean_reward']:.2f}")

    # Распределение score
    from collections import Counter
    dist = Counter(results['scores'])
    print("\n  Распределение score:")
    for score in sorted(dist.keys()):
        bar = "█" * dist[score]
        print(f"    {score:2d}: {bar} ({dist[score]})")

    # --- 4. Сохранение ---
    np.savez("fly_snake_model.npz",
             readout_params=readout.get_params(),
             brain_n_neurons=brain.num_neurons,
             brain_n_inputs=brain.n_inputs,
             readout_hidden_dim=readout.hidden_dim,
             readout_input_dim=readout.input_dim,
             readout_output_dim=readout.output_dim)
    print("\nМодель сохранена: fly_snake_model.npz")


if __name__ == "__main__":
    main()