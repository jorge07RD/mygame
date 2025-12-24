import pygame
import math

pygame.init()

ANCHO, ALTO = 800, 600
pantalla = pygame.display.set_mode((ANCHO, ALTO))
clock = pygame.time.Clock()

BLANCO = (255, 255, 255)
ROJO = (255, 0, 0)
Azul = (0, 0, 255)

NEGRO = (0, 0, 0)

# Distancias entre cada punto consecutivo
distancias = [100,100]

# Inicializar posiciones de todos los puntos de la cadena
puntos_cadena = []
pos_actual = [ANCHO // 2, ALTO // 2]
for dist in distancias:
    pos_actual = [pos_actual[0] + dist, pos_actual[1]]
    puntos_cadena.append(list(pos_actual))

def restringir_distancia(ancla, punto, distancia):
    # Vector del ancla al punto
    dx = punto[0] - ancla[0]
    dy = punto[1] - ancla[1]
    
    # Distancia actual
    dist_actual = math.sqrt(dx * dx + dy * dy)
    
    if dist_actual == 0:
        return [ancla[0] + distancia, ancla[1]]
    
    # Normalizar y escalar
    dx_norm = dx / dist_actual
    dy_norm = dy / dist_actual
    
    # Nueva posición a distancia fija
    nuevo_x = ancla[0] + dx_norm * distancia
    nuevo_y = ancla[1] + dy_norm * distancia
    
    return [int(nuevo_x), int(nuevo_y)]

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # El ancla sigue al mouse
    ancla_pos = list(pygame.mouse.get_pos())

    pantalla.fill(NEGRO)

    # Dibujar el ancla (mouse)
    pygame.draw.circle(pantalla, BLANCO, (int(ancla_pos[0]), int(ancla_pos[1])), 10)

    # Actualizar posiciones de la cadena de puntos
    # El primer punto persigue al ancla (mouse)
    puntos_cadena[0] = restringir_distancia(ancla_pos, puntos_cadena[0], distancias[0])

    # Cada punto siguiente persigue al anterior
    for i in range(1, len(puntos_cadena)):
        puntos_cadena[i] = restringir_distancia(puntos_cadena[i-1], puntos_cadena[i], distancias[i])

    # Dibujar la cadena
    referencia = ancla_pos
    for i, punto in enumerate(puntos_cadena):
        pygame.draw.aaline(pantalla, BLANCO, referencia, punto, 2)
        pygame.draw.circle(pantalla, ROJO, punto, 10)
        referencia = punto

    pygame.display.flip()
    # clock.tick(10)
    clock.tick(140)
pygame.quit()