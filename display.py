import pygame


class Display(object):
    def __init__(self, W, H):
        pygame.init()

        self.screen = pygame.display.set_mode((W, H))
        self.W, self.H = W, H

    def draw(self, img):
        background = pygame.surfarray.make_surface(img)
        self.screen.blit(background, (0, 0))
        self.screen.convert()
        pygame.display.flip()