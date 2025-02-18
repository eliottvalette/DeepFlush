# renderer.py
import pygame

def handle_final_rendering(env):
    env._draw()
    pygame.display.flip()
    
    if env.pygame_winner_info:
        current_time = pygame.time.get_ticks()
        while current_time - env.pygame_winner_display_start < env.pygame_winner_display_duration:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            env._draw()
            pygame.display.flip()
            env.clock.tick(3)  # Vous pouvez utiliser FPS si besoin
            current_time = pygame.time.get_ticks()

def handle_rendering(env, rendering, episode, render_every):
    if not (rendering and episode % render_every == 0):
        return
        
    env._draw()
    pygame.display.flip()
    env.clock.tick(3)
    
    if env.pygame_winner_info:
        current_time = pygame.time.get_ticks()
        while current_time - env.pygame_winner_display_start < env.pygame_winner_display_duration:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            env._draw()
            pygame.display.flip()
            env.clock.tick(3)
            current_time = pygame.time.get_ticks()
