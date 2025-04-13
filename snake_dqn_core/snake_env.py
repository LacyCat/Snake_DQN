import pygame
import random

class SnakeGameEnv:
    def __init__(self, width=10, height=10, render_mode=False):
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.cell_size = 20

        if render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width * self.cell_size, self.height * self.cell_size))
            pygame.display.set_caption("Snake AI")
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = (0, -1)
        self.spawn_food()
        self.done = False
        self.score = 0
        return self.get_state()

    def spawn_food(self):
        while True:
            self.food = (
                random.randint(0, self.width - 1),
                random.randint(0, self.height - 1)
            )
            if self.food not in self.snake:
                break

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, {}

        old_direction = self.direction
        self.change_direction(action)

        head_x, head_y = self.snake[0]
        dir_x, dir_y = self.direction
        new_head = (head_x + dir_x, head_y + dir_y)

        old_distance = abs(self.food[0] - head_x) + abs(self.food[1] - head_y)
        reward = -0.01  # reward shaping: step penalty
        if action != 0 and self.direction == (old_direction[0], old_direction[1]):
            reward -= 0.05
        if (new_head in self.snake) or not (0 <= new_head[0] < self.width) or not (0 <= new_head[1] < self.height):
            self.done = True
            reward = -1
            return self.get_state(), reward, self.done, {}

        self.snake.insert(0, new_head)
        new_distance = abs(self.food[0] - new_head[0]) + abs(self.food[1] - new_head[1])
        reward += (old_distance - new_distance) * 0.05
        if new_head == self.food:
            reward = 1
            self.score += 1
            self.spawn_food()
        else:
            self.snake.pop()

        if len(self.snake) == self.width * self.height:
            self.done = True
            reward += 100  # win bonus

        return self.get_state(), reward, self.done, {}

    def change_direction(self, action):
        dir_x, dir_y = self.direction
        if action == 1:
            self.direction = (-dir_y, dir_x)
        elif action == 2:
            self.direction = (dir_y, -dir_x)

    def get_state(self):
        head_x, head_y = self.snake[0]
        dir_x, dir_y = self.direction

        left_dir = (-dir_y, dir_x)
        right_dir = (dir_y, -dir_x)
        forward_dir = self.direction

        def danger_in_direction(dx, dy):
            next_x, next_y = head_x + dx, head_y + dy
            return (
                next_x < 0 or next_x >= self.width or
                next_y < 0 or next_y >= self.height or
                (next_x, next_y) in self.snake
            )

        danger_left = int(danger_in_direction(*left_dir))
        danger_right = int(danger_in_direction(*right_dir))
        danger_forward = int(danger_in_direction(*forward_dir))

        food_dx = self.food[0] - head_x
        food_dy = self.food[1] - head_y

        return [
            head_x, head_y,
            danger_left, danger_forward, danger_right,
            dir_x, dir_y,
            food_dx, food_dy
        ]

    def render(self):
        if not self.render_mode:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill((0, 0, 0))

        for x, y in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0), (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))

        food_x, food_y = self.food
        pygame.draw.rect(self.screen, (255, 0, 0), (food_x * self.cell_size, food_y * self.cell_size, self.cell_size, self.cell_size))

        pygame.display.flip()
        self.clock.tick(10)