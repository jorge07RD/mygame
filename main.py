import pygame
import math

pygame.init()
ANCHO, ALTO = 2560, 1440
pantalla = pygame.display.set_mode((ANCHO, ALTO))
clock = pygame.time.Clock()

BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
VERDE_OSCURO = (34, 120, 50)
VERDE_CLARO = (50, 180, 70)
AMARILLO = (220, 200, 50)
ROJO = (200, 50, 50)
GRIS = (100, 100, 100)

# Más segmentos para una serpiente suave
NUM_SEGMENTOS = 50
distancias = [30] * NUM_SEGMENTOS
ANGULO_MAX = math.radians(25)

puntos_cadena = []
pos_actual = [ANCHO // 2, ALTO // 2]
for dist in distancias:
    pos_actual = [pos_actual[0] + dist, pos_actual[1]]
    puntos_cadena.append(list(pos_actual))

# Control
usar_mouse = True
ancla_pos = [ANCHO // 2, ALTO // 2]
velocidad = 5

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
    diff = angulo_actual - angulo_anterior
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    if diff > limite:
        return angulo_anterior + limite
    elif diff < -limite:
        return angulo_anterior - limite
    return angulo_actual

def aplicar_restricciones(ancla, punto, distancia, angulo_anterior):
    dx = punto[0] - ancla[0]
    dy = punto[1] - ancla[1]
    angulo_actual = math.atan2(dy, dx)
    angulo_limitado = limitar_angulo(angulo_actual, angulo_anterior, ANGULO_MAX)
    nuevo_x = ancla[0] + distancia * math.cos(angulo_limitado)
    nuevo_y = ancla[1] + distancia * math.sin(angulo_limitado)
    return [nuevo_x, nuevo_y], angulo_limitado

def obtener_grosor(indice, total):
    progreso = indice / total
    if progreso < 0.2:
        return 25 - (progreso * 30)
    elif progreso < 0.7:
        return 18
    else:
        return 18 - ((progreso - 0.7) * 50)

def dibujar_boton(pantalla, texto, x, y, ancho, alto, activo):
    color = VERDE_CLARO if activo else GRIS
    pygame.draw.rect(pantalla, color, (x, y, ancho, alto), border_radius=5)
    pygame.draw.rect(pantalla, BLANCO, (x, y, ancho, alto), 2, border_radius=5)
    
    fuente = pygame.font.Font(None, 24)
    texto_render = fuente.render(texto, True, BLANCO)
    texto_rect = texto_render.get_rect(center=(x + ancho//2, y + alto//2))
    pantalla.blit(texto_render, texto_rect)
    
    return pygame.Rect(x, y, ancho, alto)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        # Cambiar modo con ESPACIO o click en botón
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                usar_mouse = not usar_mouse
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if boton_rect.collidepoint(event.pos):
                usar_mouse = not usar_mouse

    # Control con flechas
    if not usar_mouse:
        teclas = pygame.key.get_pressed()
        if teclas[pygame.K_LEFT]:
            ancla_pos[0] -= velocidad
        if teclas[pygame.K_RIGHT]:
            ancla_pos[0] += velocidad
        if teclas[pygame.K_UP]:
            ancla_pos[1] -= velocidad
        if teclas[pygame.K_DOWN]:
            ancla_pos[1] += velocidad
        
        # Mantener dentro de la pantalla
        ancla_pos[0] = max(0, min(ANCHO, ancla_pos[0]))
        ancla_pos[1] = max(0, min(ALTO, ancla_pos[1]))
    else:
        ancla_pos = list(pygame.mouse.get_pos())

    pantalla.fill(NEGRO)

    # Actualizar cadena
    puntos_cadena[0], angulo_prev = restringir_distancia(ancla_pos, puntos_cadena[0], distancias[0])
    for i in range(1, len(puntos_cadena)):
        puntos_cadena[i], angulo_prev = aplicar_restricciones(
            puntos_cadena[i-1], 
            puntos_cadena[i], 
            distancias[i], 
            angulo_prev
        )

    # Calcular puntos del manto
    puntos_izq = []
    puntos_der = []
    puntos_centro = [ancla_pos]

    dx = puntos_cadena[0][0] - ancla_pos[0]
    dy = puntos_cadena[0][1] - ancla_pos[1]
    angulo_cabeza = math.atan2(dy, dx)
    grosor_cabeza = 22
    
    puntos_izq.append((
        int(ancla_pos[0] + grosor_cabeza * math.cos(angulo_cabeza + math.pi/2)),
        int(ancla_pos[1] + grosor_cabeza * math.sin(angulo_cabeza + math.pi/2))
    ))
    puntos_der.append((
        int(ancla_pos[0] + grosor_cabeza * math.cos(angulo_cabeza - math.pi/2)),
        int(ancla_pos[1] + grosor_cabeza * math.sin(angulo_cabeza - math.pi/2))
    ))

    referencia = ancla_pos
    for i, punto in enumerate(puntos_cadena):
        puntos_centro.append(punto)
        dx = punto[0] - referencia[0]
        dy = punto[1] - referencia[1]
        angulo = math.atan2(dy, dx)
        
        grosor = obtener_grosor(i, len(puntos_cadena))

        x_der = punto[0] + grosor * math.cos(angulo + math.pi/2)
        y_der = punto[1] + grosor * math.sin(angulo + math.pi/2)
        puntos_der.append((int(x_der), int(y_der)))

        x_izq = punto[0] + grosor * math.cos(angulo - math.pi/2)
        y_izq = punto[1] + grosor * math.sin(angulo - math.pi/2)
        puntos_izq.append((int(x_izq), int(y_izq)))

        referencia = punto

    ultimo = puntos_cadena[-1]
    puntos_cola = (int(ultimo[0] + 15 * math.cos(angulo)), 
                   int(ultimo[1] + 15 * math.sin(angulo)))

    manto = puntos_izq + [puntos_cola] + puntos_der[::-1]

    if len(manto) > 2:
        pygame.draw.polygon(pantalla, VERDE_OSCURO, manto)
        pygame.draw.polygon(pantalla, VERDE_CLARO, manto, 3)

    for i in range(len(puntos_izq) - 1):
        if i % 2 == 0:
            pygame.draw.line(pantalla, VERDE_CLARO, puntos_izq[i], puntos_der[i], 1)

    for i in range(len(puntos_centro) - 1):
        pygame.draw.line(pantalla, AMARILLO, 
                        (int(puntos_centro[i][0]), int(puntos_centro[i][1])),
                        (int(puntos_centro[i+1][0]), int(puntos_centro[i+1][1])), 4)

    pygame.draw.circle(pantalla, VERDE_OSCURO, (int(ancla_pos[0]), int(ancla_pos[1])), 24)
    pygame.draw.circle(pantalla, VERDE_CLARO, (int(ancla_pos[0]), int(ancla_pos[1])), 24, 2)

    ojo_offset = 12
    ojo_angulo_offset = 0.6
    
    ojo_izq = (
        int(ancla_pos[0] + ojo_offset * math.cos(angulo_cabeza + math.pi + ojo_angulo_offset)),
        int(ancla_pos[1] + ojo_offset * math.sin(angulo_cabeza + math.pi + ojo_angulo_offset))
    )
    ojo_der = (
        int(ancla_pos[0] + ojo_offset * math.cos(angulo_cabeza + math.pi - ojo_angulo_offset)),
        int(ancla_pos[1] + ojo_offset * math.sin(angulo_cabeza + math.pi - ojo_angulo_offset))
    )
    
    pygame.draw.circle(pantalla, AMARILLO, ojo_izq, 6)
    pygame.draw.circle(pantalla, AMARILLO, ojo_der, 6)
    pygame.draw.circle(pantalla, NEGRO, ojo_izq, 3)
    pygame.draw.circle(pantalla, NEGRO, ojo_der, 3)

    lengua_base = (
        int(ancla_pos[0] + 20 * math.cos(angulo_cabeza + math.pi)),
        int(ancla_pos[1] + 20 * math.sin(angulo_cabeza + math.pi))
    )
    lengua_punta1 = (
        int(lengua_base[0] + 15 * math.cos(angulo_cabeza + math.pi + 0.3)),
        int(lengua_base[1] + 15 * math.sin(angulo_cabeza + math.pi + 0.3))
    )
    lengua_punta2 = (
        int(lengua_base[0] + 15 * math.cos(angulo_cabeza + math.pi - 0.3)),
        int(lengua_base[1] + 15 * math.sin(angulo_cabeza + math.pi - 0.3))
    )
    
    pygame.draw.line(pantalla, ROJO, (int(ancla_pos[0]), int(ancla_pos[1])), lengua_base, 3)
    pygame.draw.line(pantalla, ROJO, lengua_base, lengua_punta1, 2)
    pygame.draw.line(pantalla, ROJO, lengua_base, lengua_punta2, 2)

    # Dibujar botón y texto de control
    modo_texto = "MOUSE" if usar_mouse else "FLECHAS"
    boton_rect = dibujar_boton(pantalla, f"Modo: {modo_texto}", 10, 10, 150, 40, True)
    
    # Instrucciones
    fuente = pygame.font.Font(None, 20)
    instrucciones = fuente.render("ESPACIO o click para cambiar modo", True, GRIS)
    pantalla.blit(instrucciones, (10, 55))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()