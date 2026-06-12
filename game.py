# pyrefly: ignore [missing-import]
import pygame
import random
from settings import *
from sprites import Dino, Obstacle, Ground

class DinoGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Dino Runner")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("comicsans", 30)
        self.reset()

    def reset(self):
        self.dino = Dino()
        self.obstacles = []
        self.ground = Ground()
        self.score = 0
        self.speed = BASE_SPEED
        self.game_over = False
        return self.get_state()

    def get_state(self):
        # Useful for Neural Network input later
        return {
            "dino_y": self.dino.y,
            "dino_vel_y": self.dino.vel_y,
            "obstacles": [(obs.x, obs.y, obs.width, obs.height) for obs in self.obstacles],
            "speed": self.speed
        }

    def spawn_obstacle(self):
        if len(self.obstacles) == 0:
            self.obstacles.append(Obstacle(SCREEN_WIDTH + random.randint(100, 300)))
        else:
            last_obs = self.obstacles[-1]
            # Ensure minimum distance between obstacles
            if SCREEN_WIDTH - last_obs.x > random.randint(OBSTACLE_SPAWN_MIN_DISTANCE, OBSTACLE_SPAWN_MAX_DISTANCE):
                self.obstacles.append(Obstacle(SCREEN_WIDTH + random.randint(50, 150)))

    def step(self, action):
        """
        action: 0=do nothing, 1=jump, 2=duck
        returns: (state, reward, done)
        """
        if self.game_over:
            return self.get_state(), 0, True

        # Process action
        if action == 1:
            self.dino.jump()
        elif action == 2:
            self.dino.duck()
        else:
            self.dino.unduck()

        # Update dino
        self.dino.update()

        # Update obstacles and check collision
        reward = 0.1 # Survival reward per frame
        for obs in self.obstacles:
            obs.update(self.speed)
            if self.dino.rect.colliderect(obs.rect):
                self.game_over = True
                reward = -10
                break

        # Remove off-screen obstacles and increase score
        for obs in self.obstacles[:]:
            if obs.x + obs.width < 0:
                self.obstacles.remove(obs)
                self.score += 10 # 10 points per obstacle cleared
                reward += 1

        self.spawn_obstacle()

        # Gradually increase game speed
        if self.speed < MAX_SPEED:
            self.speed += SPEED_INCREMENT

        return self.get_state(), reward, self.game_over

    def draw(self):
        self.screen.fill(WHITE)
        self.ground.draw(self.screen)
        
        for obs in self.obstacles:
            obs.draw(self.screen)
            
        self.dino.draw(self.screen)

        # Draw Score
        score_text = self.font.render(f"Score: {int(self.score)}", True, BLACK)
        self.screen.blit(score_text, (SCREEN_WIDTH - 150, 20))

        if self.game_over:
            go_text = self.font.render("Game Over! Press R to Restart", True, BLACK)
            self.screen.blit(go_text, (SCREEN_WIDTH//2 - go_text.get_width()//2, SCREEN_HEIGHT//2))

        pygame.display.flip()
