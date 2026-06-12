# pyrefly: ignore [missing-import]
import pygame
import random
from settings import *

class Dino:
    def __init__(self):
        self.width = DINO_WIDTH
        self.height = DINO_HEIGHT
        self.x = 50
        self.y = FLOOR_Y - self.height
        self.vel_y = 0
        self.is_jumping = False
        self.is_ducking = False
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def jump(self):
        if not self.is_jumping:
            self.vel_y = JUMP_VELOCITY
            self.is_jumping = True
            # Unduck if jumping
            if self.is_ducking:
                self.unduck()

    def duck(self):
        if not self.is_jumping and not self.is_ducking:
            self.is_ducking = True
            self.height = DINO_DUCK_HEIGHT
            self.y = FLOOR_Y - self.height

    def unduck(self):
        if self.is_ducking:
            self.is_ducking = False
            self.height = DINO_HEIGHT
            self.y = FLOOR_Y - self.height

    def update(self):
        # Apply gravity
        self.vel_y += GRAVITY
        self.y += self.vel_y

        # Check collision with floor
        if self.y >= FLOOR_Y - self.height:
            self.y = FLOOR_Y - self.height
            self.vel_y = 0
            self.is_jumping = False

        self.rect.update(self.x, self.y, self.width, self.height)

    def draw(self, surface):
        pygame.draw.rect(surface, DINO_COLOR, self.rect)


class Obstacle:
    def __init__(self, x):
        self.is_bird = random.random() < 0.25 # 25% chance of bird
        
        if self.is_bird:
            self.width = 40
            self.height = 25
            # Fly at ducking height or jump height
            self.y = FLOOR_Y - DINO_HEIGHT - random.choice([0, 20])
        else:
            self.width = random.choice([20, 30, 40])
            self.height = random.choice([35, 45, 55])
            self.y = FLOOR_Y - self.height
            
        self.x = x
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def update(self, speed):
        self.x -= speed
        self.rect.x = self.x

    def draw(self, surface):
        color = (150, 50, 50) if self.is_bird else CACTUS_COLOR
        pygame.draw.rect(surface, color, self.rect)


class Ground:
    def __init__(self):
        self.x = 0
        self.y = FLOOR_Y
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT - FLOOR_Y
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        
    def draw(self, surface):
        pygame.draw.line(surface, GROUND_COLOR, (0, self.y), (SCREEN_WIDTH, self.y), 2)
