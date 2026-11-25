import pygame
import random

from src.settings import *
from src.sprites import Dino, Obstacle, Ground


class DinoGame:

    def __init__(self):

        pygame.init()

        self.screen = pygame.display.set_mode(
            (SCREEN_WIDTH, SCREEN_HEIGHT)
        )

        pygame.display.set_caption(
            "Dino Runner"
        )

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(
            "comicsans",
            30
        )

        self.reset()


    # ========================================================
    # RESET GAME
    # ========================================================

    def reset(self):

        self.dino = Dino()

        self.obstacles = []

        self.ground = Ground()

        self.score = 0

        self.speed = BASE_SPEED

        self.game_over = False

        return self.get_state()


    # ========================================================
    # RETURN STATE FOR AI
    # ========================================================

    def get_state(self):

        return {

            "dino_x": self.dino.x,

            "dino_y": self.dino.y,

            "dino_vel_y": self.dino.vel_y,

            "is_ducking": self.dino.is_ducking,

            "obstacles": [
                (
                    obs.x,
                    obs.y,
                    obs.width,
                    obs.height
                )
                for obs in self.obstacles
            ],

            "speed": self.speed
        }


    # ========================================================
    # SPAWN OBSTACLE
    # ========================================================

    def spawn_obstacle(self):

        if len(self.obstacles) == 0:

            self.obstacles.append(
                Obstacle(
                    SCREEN_WIDTH
                    + random.randint(100, 300)
                )
            )

            return


        last_obs = self.obstacles[-1]

        distance_from_spawn = (
            SCREEN_WIDTH - last_obs.x
        )


        required_distance = random.randint(
            OBSTACLE_SPAWN_MIN_DISTANCE,
            OBSTACLE_SPAWN_MAX_DISTANCE
        )


        if distance_from_spawn > required_distance:

            self.obstacles.append(
                Obstacle(
                    SCREEN_WIDTH
                    + random.randint(50, 150)
                )
            )


    # ========================================================
    # GAME STEP
    # ========================================================

    def step(self, action):

        """
        action:

        0 = Run / do nothing
        1 = Jump
        2 = Duck

        Returns:

        state, reward, done
        """

        if self.game_over:

            return (
                self.get_state(),
                0,
                True
            )


        # ----------------------------------------------------
        # PROCESS ACTION
        # ----------------------------------------------------

        if action == 1:

            self.dino.jump()

        elif action == 2:

            self.dino.duck()

        else:

            self.dino.unduck()


        # ----------------------------------------------------
        # UPDATE DINO
        # ----------------------------------------------------

        self.dino.update()


        # ----------------------------------------------------
        # SMALL SURVIVAL REWARD
        # ----------------------------------------------------

        reward = 0.01


        # ----------------------------------------------------
        # UPDATE OBSTACLES
        # ----------------------------------------------------

        for obs in self.obstacles:

            obs.update(
                self.speed
            )


            # ------------------------------------------------
            # COLLISION
            # ------------------------------------------------

            if self.dino.rect.colliderect(
                obs.rect
            ):

                self.game_over = True

                # Strong punishment for crashing
                reward = -10.0

                # IMPORTANT:
                # Stop processing immediately.
                return (
                    self.get_state(),
                    reward,
                    True
                )


            # ------------------------------------------------
            # SUCCESSFULLY PASSED OBSTACLE
            # ------------------------------------------------
            #
            # As soon as the obstacle moves behind the Dino,
            # give the reward.
            #
            # Don't wait until it leaves the entire screen.
            # ------------------------------------------------

            if (
                not obs.passed
                and
                obs.x + obs.width < self.dino.x
            ):

                obs.passed = True

                self.score += 10

                # Strong positive feedback
                reward += 2.0


        # ----------------------------------------------------
        # REMOVE OLD OBSTACLES
        # ----------------------------------------------------

        self.obstacles = [

            obs

            for obs in self.obstacles

            if obs.x + obs.width > 0

        ]


        # ----------------------------------------------------
        # SPAWN NEW OBSTACLE
        # ----------------------------------------------------

        self.spawn_obstacle()


        # ----------------------------------------------------
        # GRADUALLY INCREASE SPEED
        # ----------------------------------------------------

        if self.speed < MAX_SPEED:

            self.speed = min(
                MAX_SPEED,
                self.speed + SPEED_INCREMENT
            )


        return (
            self.get_state(),
            reward,
            False
        )


    # ========================================================
    # DRAW
    # ========================================================

    def draw(self):

        self.screen.fill(
            WHITE
        )


        self.ground.draw(
            self.screen
        )


        for obs in self.obstacles:

            obs.draw(
                self.screen
            )


        self.dino.draw(
            self.screen
        )


        score_text = self.font.render(
            f"Score: {int(self.score)}",
            True,
            BLACK
        )


        self.screen.blit(
            score_text,
            (
                SCREEN_WIDTH - 150,
                20
            )
        )


        if self.game_over:

            go_text = self.font.render(
                "Game Over!",
                True,
                BLACK
            )


            self.screen.blit(
                go_text,
                (
                    SCREEN_WIDTH // 2
                    - go_text.get_width() // 2,

                    SCREEN_HEIGHT // 2
                )
            )


        pygame.display.flip()