import pygame
import math

pygame.init()
pantalla = pygame.display.set_mode((400, 400))
# Fuente para mostrar valores en pantalla
font = pygame.font.SysFont(None, 24)
BLANCO = (255, 255, 255)
ROJO = (255, 0, 0)

cx, cy = 200, 200
r = 80

# definir aquí para evitar variables posiblemente desvinculadas
start, stop, step = 0, 180, 4

corriendo = True
while corriendo:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            corriendo = False

    pantalla.fill((0, 0, 50))

    # Dibujar círculo con ecuación paramétrica
    pygame.draw.circle(pantalla, BLANCO, (cx, cy), 80, 2)

    keys = pygame.key.get_pressed()
    # Increment: y, u, i
    if keys[pygame.K_y]:
        start += 1
    if keys[pygame.K_u]:
        stop += 1
    if keys[pygame.K_i]:
        step += 1
    # Decrement: h, j, k
    if keys[pygame.K_h]:
        start -= 1
    if keys[pygame.K_j]:
        stop -= 1
    if keys[pygame.K_k]:
        step -= 1

    # avoid zero step and clamp values
    if step == 0:
        step = 1
    start = max(-360, min(360, start))
    stop = max(-360, min(360, stop))

    # Renderizar valores en pantalla
    info_text = f"start: {start}  stop: {stop}  step: {step}"
    info_surf = font.render(info_text, True, BLANCO)
    pantalla.blit(info_surf, (10, 10))

    # puntos a la derecha e izquierda del círculo
    pygame.draw.circle(pantalla, ROJO, (int(cx + r), int(cy)), 5)
    pygame.draw.circle(pantalla, BLANCO, (int(cx - r), int(cy)), 5)

    for grados in range(start, stop, step):
        t = math.radians(grados)
        x = cx + r * math.cos(t)
        y = cy + r * math.sin(t)
        pygame.draw.circle(pantalla, ROJO, (int(x), int(y)), 3)
    pygame.display.flip()

pygame.quit()