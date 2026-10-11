import numpy as np


class CEMTrainer:
    """
    Cross-Entropy Method с ранним остановом и автосохранением.
    """

    def __init__(self, readout, pop_size=64, n_elite=8,
                 n_gen=300, init_sigma=0.7, seed=42,
                 patience=25, checkpoint_path="fly_snake_checkpoint.npz"):
        np.random.seed(seed)
        self.readout = readout
        self.pop_size = pop_size
        self.n_elite = n_elite
        self.n_gen = n_gen

        self.param_dim = readout.param_dim
        self.mu = np.zeros(self.param_dim)
        self.sigma = np.ones(self.param_dim) * init_sigma

        self.best_params = None
        self.best_score = -np.inf
        self.history = []

        self.patience = patience
        self.no_improve = 0
        self.checkpoint_path = checkpoint_path

    def evaluate(self, params, env, brain, n_episodes=5):
        self.readout.set_params(params)
        total = 0.0
        for _ in range(n_episodes):
            brain.reset()
            reward, _ = env.run_episode(brain, self.readout, epsilon=0.0)
            total += reward
        return total / n_episodes

    def train(self, env, brain, verbose=True):
        for gen in range(self.n_gen):
            # 1. Сэмплируем кандидатов
            candidates = [
                self.mu + self.sigma * np.random.randn(self.param_dim)
                for _ in range(self.pop_size)
            ]

            # 2. Оцениваем
            scores = [self.evaluate(c, env, brain) for c in candidates]

            # 3. Элита
            elite_idx = np.argsort(scores)[-self.n_elite:]
            elite = [candidates[i] for i in elite_idx]
            elite_scores = [scores[i] for i in elite_idx]

            # 4. Обновляем mu и sigma
            self.mu = np.mean(elite, axis=0)
            self.sigma = np.std(elite, axis=0) + 1e-6

            # 5. Best + early stopping
            best_idx = np.argmax(scores)
            improved = False
            if scores[best_idx] > self.best_score:
                self.best_score = scores[best_idx]
                self.best_params = candidates[best_idx].copy()
                self.no_improve = 0
                improved = True
            else:
                self.no_improve += 1

            self.history.append(max(scores))

            if verbose and (gen % 10 == 0 or gen == self.n_gen - 1):
                marker = " *" if improved else ""
                print(f"  Gen {gen:3d} | best: {max(scores):7.2f} | "
                      f"elite avg: {np.mean(elite_scores):7.2f} | "
                      f"sigma: {self.sigma.mean():.4f} | "
                      f"mu |W|: {np.abs(self.mu).mean():.4f}{marker}")

            # 6. Автосохранение каждые 10 поколений
            if self.best_params is not None and gen % 10 == 0:
                np.savez(self.checkpoint_path,
                         readout_params=self.best_params,
                         gen=gen,
                         best_score=self.best_score)
                if verbose:
                    print(f"           [checkpoint saved: {self.checkpoint_path}]")

            # 7. Ранний останов
            if self.no_improve >= self.patience:
                print(f"\nРанний останов на поколении {gen} "
                      f"(нет улучшения {self.patience} поколений)")
                break

        self.readout.set_params(self.best_params)
        print(f"CEM завершён. Best score: {self.best_score:.2f}")
        return self.best_params