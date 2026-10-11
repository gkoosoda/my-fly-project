import random
import numpy as np


class SnakeEnv:
    """
    Змейка с 12-мерным структурированным входом.
    Действия: 0 = turn_left, 1 = straight, 2 = turn_right.
    Честный счётчик шагов (не сбрасывается при еде).
    """

    def __init__(self, grid_size=20, max_steps=200):
        self.GS = grid_size
        self.MAX_STEPS = max_steps
        self.reset()

    def reset(self):
        self.body = [(self.GS // 2, self.GS // 2)]
        self.direction = (1, 0)
        self.grow = False
        self.food = (0, 0)
        self.score = 0
        self.steps = 0                 # для таймаута (сбрасывается при еде)
        self.total_steps = 0           # честный счётчик (никогда не сбрасывается)
        self._randomize_food()

    def _randomize_food(self):
        while True:
            self.food = (random.randint(0, self.GS - 1),
                         random.randint(0, self.GS - 1))
            if self.food not in self.body:
                break

    def get_stimuli(self):
        """
        12-мерный вход:
          [0] danger_left
          [1] danger_straight
          [2] danger_right
          [3] forward_norm   — проекция еды на направление движения
          [4] side_norm      — проекция еды на боковое направление
          [5] food_left
          [6] food_right
          [7] food_up
          [8] food_down
          [9] bias = 1.0
          [10] padding
          [11] padding
        """
        head = self.body[0]
        dx, dy = self.direction
        left_dir = (-dy, dx)
        right_dir = (dy, -dx)

        def is_danger(direction):
            nx, ny = head[0] + direction[0], head[1] + direction[1]
            if nx < 0 or nx >= self.GS or ny < 0 or ny >= self.GS:
                return 1.0
            if (nx, ny) in self.body:
                return 1.0
            return 0.0

        danger_left = is_danger(left_dir)
        danger_straight = is_danger(self.direction)
        danger_right = is_danger(right_dir)

        fx, fy = self.food
        rel_x = fx - head[0]
        rel_y = fy - head[1]

        forward = rel_x * dx + rel_y * dy
        side = rel_x * (-dy) + rel_y * dx

        forward_norm = np.clip(forward / self.GS, -1.0, 1.0)
        side_norm = np.clip(side / self.GS, -1.0, 1.0)

        food_left = 1.0 if fx < head[0] else 0.0
        food_right = 1.0 if fx > head[0] else 0.0
        food_up = 1.0 if fy < head[1] else 0.0
        food_down = 1.0 if fy > head[1] else 0.0

        return np.array([
            danger_left, danger_straight, danger_right,
            forward_norm, side_norm,
            food_left, food_right, food_up, food_down,
            1.0, 0.0, 0.0,
        ], dtype=float)

    def step(self, action):
        dx, dy = self.direction
        if action == 0:
            self.direction = (-dy, dx)
        elif action == 2:
            self.direction = (dy, -dx)

        head = self.body[0]
        new_head = (head[0] + self.direction[0],
                    head[1] + self.direction[1])

        alive = True
        if (new_head in self.body or
                new_head[0] < 0 or new_head[0] >= self.GS or
                new_head[1] < 0 or new_head[1] >= self.GS):
            alive = False

        if not alive:
            return self.get_stimuli(), -1.0, True, {"score": self.score}

        self.body.insert(0, new_head)
        if not self.grow:
            self.body.pop()
        self.grow = False
        self.steps += 1
        self.total_steps += 1

        reward = 0.0
        if new_head == self.food:
            self.grow = True
            reward = 1.0
            self.score += 1
            self.steps = 0                     # таймаут сбрасывается
            self._randomize_food()
        else:
            old_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
            new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            if new_dist < old_dist:
                reward += 0.1
            else:
                reward -= 0.1

        done = self.steps >= self.MAX_STEPS
        return self.get_stimuli(), reward, done, {"score": self.score}

    def run_episode(self, brain, readout, epsilon=0.0):
        """
        Возвращает (total_reward, total_steps) — честное число шагов.
        """
        self.reset()
        brain.reset()
        total_reward = 0.0

        while True:
            stimuli = self.get_stimuli()
            brain.step(stimuli)
            neural_activity = brain.get_activity()

            action, _ = readout.get_action(neural_activity, epsilon)
            _, reward, done, info = self.step(action)
            total_reward += reward

            if done:
                return total_reward, self.total_steps