import pygame
import random
from src.settings import *


class Dino:
    def __init__(self):
        self.width = DINO_WIDTH
        self.height = DINO_HEIGHT

        self.x = 50
        self.y = FLOOR_Y - self.height

        self.vel_y = 0

        self.is_jumping = False
        self.is_ducking = False

        self.rect = pygame.Rect(
            self.x,
            self.y,
            self.width,
            self.height
        )

    def jump(self):
        if not self.is_jumping:

            self.vel_y = JUMP_VELOCITY
            self.is_jumping = True

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

        # Gravity only really affects us while jumping,
        # but your original implementation is also valid.
        self.vel_y += GRAVITY
        self.y += self.vel_y

        if self.y >= FLOOR_Y - self.height:

            self.y = FLOOR_Y - self.height

            self.vel_y = 0
            self.is_jumping = False

        self.rect.update(
            self.x,
            self.y,
            self.width,
            self.height
        )

    def draw(self, surface):
        pygame.draw.rect(
            surface,
            DINO_COLOR,
            self.rect
        )


class Obstacle:
    def __init__(self, x):

        # NEW:
        # Used to ensure the AI gets the obstacle-clear
        # reward only once.
        self.passed = False

        # 25% chance of a bird
        self.is_bird = random.random() < 0.25

        if self.is_bird:

            self.width = 40
            self.height = 25

            # Bird at one of two heights
            self.y = (
                FLOOR_Y
                - DINO_HEIGHT
                - random.choice([0, 20])
            )

        else:

            self.width = random.choice(
                [20, 30, 40]
            )

            self.height = random.choice(
                [35, 45, 55]
            )

            self.y = FLOOR_Y - self.height

        self.x = x

        self.rect = pygame.Rect(
            self.x,
            self.y,
            self.width,
            self.height
        )

    def update(self, speed):

        self.x -= speed

        self.rect.x = int(self.x)

    def draw(self, surface):

        if self.is_bird:
            color = (150, 50, 50)
        else:
            color = CACTUS_COLOR

        pygame.draw.rect(
            surface,
            color,
            self.rect
        )


class Ground:
    def __init__(self):

        self.x = 0
        self.y = FLOOR_Y

        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT - FLOOR_Y

        self.rect = pygame.Rect(
            self.x,
            self.y,
            self.width,
            self.height
        )

    def draw(self, surface):

        pygame.draw.line(
            surface,
            GROUND_COLOR,
            (0, self.y),
            (SCREEN_WIDTH, self.y),
            2
        )