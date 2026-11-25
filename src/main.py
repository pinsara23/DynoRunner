# pyrefly: ignore [missing-import]
import pygame
import sys
from src.game import DinoGame
from src.settings import FPS

def main():
    game = DinoGame()
    
    running = True
    while running:
        action = 0 # 0=do nothing, 1=jump, 2=duck
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                    if game.game_over:
                        game.reset()
                    else:
                        action = 1
                elif event.key == pygame.K_DOWN:
                    action = 2
                elif event.key == pygame.K_r and game.game_over:
                    game.reset()

        # Check continuous key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_DOWN] and not game.game_over:
            action = 2
        elif (keys[pygame.K_SPACE] or keys[pygame.K_UP]) and not game.game_over:
            # Only trigger jump if not already jumping to avoid holding jump down to fly
            if not game.dino.is_jumping:
                action = 1
            
        state, reward, done = game.step(action)
        game.draw()
        
        game.clock.tick(FPS)

if __name__ == "__main__":
    main()
