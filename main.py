import pygame
import math

pygame.init()
ANCHO, ALTO = 800, 600
pantalla = pygame.display.set_mode((ANCHO, ALTO))
clock = pygame.time.Clock()

BLANCO = (255, 255, 255)
ROJO = (255, 0, 0)
NEGRO = (0, 0, 0)
distancias = [75] * 2 + [50] * 8  + [25] * 10 + [10] * 10# Cadena de 10 puntos con distancia fija de 50 píxeles
# distancias = [100] * 2 # Cadena de 10 puntos con distancia fija de 50 píxeles

ANGULO_MAX = math.radians(30)  # Límite de 30 grados entre segmentos

puntos_cadena = []
pos_actual = [ANCHO // 2, ALTO // 2]
for dist in distancias:
    pos_actual = [pos_actual[0] + dist, pos_actual[1]]
    puntos_cadena.append(list(pos_actual))

def restringir_distancia(ancla, punto, distancia):
    dx = punto[0] - ancla[0]
    dy = punto[1] - ancla[1]
    dist_actual = math.sqrt(dx * dx + dy * dy)
    if dist_actual == 0:
        return [ancla[0] + distancia, ancla[1]], 0
    dx_norm = dx / dist_actual
    dy_norm = dy / dist_actual
    nuevo_x = ancla[0] + dx_norm * distancia
    nuevo_y = ancla[1] + dy_norm * distancia
    angulo = math.atan2(dy_norm, dx_norm)
    return [nuevo_x, nuevo_y], angulo

def limitar_angulo(angulo_actual, angulo_anterior, limite):
    # Calcular diferencia de ángulo
    diff = angulo_actual - angulo_anterior
    
    # Normalizar a rango [-π, π]
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    
    # Limitar el ángulo
    if diff > limite:
        return angulo_anterior + limite
    elif diff < -limite:
        return angulo_anterior - limite
    return angulo_actual

def aplicar_restricciones(ancla, punto, distancia, angulo_anterior):
    # Primero calculamos el ángulo natural
    dx = punto[0] - ancla[0]
    dy = punto[1] - ancla[1]
    angulo_actual = math.atan2(dy, dx)
    
    # Limitar el ángulo
    angulo_limitado = limitar_angulo(angulo_actual, angulo_anterior, ANGULO_MAX)
    
    # Calcular nueva posición con ángulo limitado
    nuevo_x = ancla[0] + distancia * math.cos(angulo_limitado)
    nuevo_y = ancla[1] + distancia * math.sin(angulo_limitado)
    
    return [nuevo_x, nuevo_y], angulo_limitado

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ancla_pos = list(pygame.mouse.get_pos())
    pantalla.fill(NEGRO)

    pygame.draw.circle(pantalla, BLANCO, (int(ancla_pos[0]), int(ancla_pos[1])), 10)

    # Primer punto sigue al ancla sin límite de ángulo
    puntos_cadena[0], angulo_prev = restringir_distancia(ancla_pos, puntos_cadena[0], distancias[0])
    
    # Los demás puntos tienen límite de ángulo
    for i in range(1, len(puntos_cadena)):
        puntos_cadena[i], angulo_prev = aplicar_restricciones(
            puntos_cadena[i-1], 
            puntos_cadena[i], 
            distancias[i], 
            angulo_prev
        )

    # Guardar puntos rojos
    puntos_izq = []
    puntos_der = []

    referencia = ancla_pos
    for i, punto in enumerate(puntos_cadena):
        punto_int = (int(punto[0]), int(punto[1]))
        pygame.draw.circle(pantalla, BLANCO, punto_int, 5)
        pygame.draw.circle(pantalla, BLANCO, punto_int, distancias[i], 2)

        dx = punto[0] - referencia[0]
        dy = punto[1] - referencia[1]
        angulo = math.atan2(dy, dx)

        x_der = punto[0] + distancias[i] * math.cos(angulo + math.pi/2)
        y_der = punto[1] + distancias[i] * math.sin(angulo + math.pi/2)
        puntos_der.append((int(x_der), int(y_der)))
        pygame.draw.circle(pantalla, ROJO, (int(x_der), int(y_der)), 5)

        x_izq = punto[0] + distancias[i] * math.cos(angulo - math.pi/2)
        y_izq = punto[1] + distancias[i] * math.sin(angulo - math.pi/2)
        puntos_izq.append((int(x_izq), int(y_izq)))
        pygame.draw.circle(pantalla, ROJO, (int(x_izq), int(y_izq)), 5)

        referencia = punto

    # Conectar puntos rojos
    for i in range(len(puntos_izq) - 1):
        pygame.draw.line(pantalla, ROJO, puntos_izq[i], puntos_izq[i+1], 2)
        pygame.draw.line(pantalla, ROJO, puntos_der[i], puntos_der[i+1], 2)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()