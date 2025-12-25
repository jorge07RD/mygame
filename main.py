"""
Simulación de serpiente con física de cadena cinemática.
Implementa una serpiente que sigue al mouse o se controla con las flechas del teclado,
con restricciones de distancia y ángulo entre segmentos para movimiento realista.
"""

from typing import List, Tuple
import pygame
import math

# ============================================================================
# CONFIGURACIÓN E INICIALIZACIÓN
# ============================================================================

pygame.init()

# Dimensiones de la ventana
ANCHO, ALTO = 2560, 1440
pantalla = pygame.display.set_mode((ANCHO, ALTO))
clock = pygame.time.Clock()

# Paleta de colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
VERDE_OSCURO = (34, 120, 50)
VERDE_CLARO = (50, 180, 70)
AMARILLO = (220, 200, 50)
ROJO = (200, 50, 50)
GRIS = (100, 100, 100)

# Física de la serpiente
NUM_SEGMENTOS = 50  # Más segmentos para una serpiente suave
DISTANCIA_SEGMENTO = 30  # Píxeles entre segmentos
ANGULO_MAX = math.radians(25)  # Máximo cambio de ángulo entre segmentos

# Constantes de apariencia
GROSOR_CABEZA = 22
GROSOR_CUERPO_GRUESO = 25
GROSOR_CUERPO_MEDIO = 18
RADIO_CABEZA = 24
RADIO_OJO = 6
RADIO_PUPILA = 3
OFFSET_OJO = 12  # Distancia del centro de la cabeza a los ojos
ANGULO_SEPARACION_OJOS = 0.6  # Radianes entre los ojos
LONGITUD_BASE_LENGUA = 20
LONGITUD_PUNTA_LENGUA = 15
ANGULO_BIFURCACION_LENGUA = 0.3
EXTENSION_COLA = 15

# Control de velocidad
VELOCIDAD_TECLADO = 5
FPS = 60

distancias = [DISTANCIA_SEGMENTO] * NUM_SEGMENTOS

# ============================================================================
# INICIALIZACIÓN DE LA CADENA
# ============================================================================

# Crear la cadena de segmentos inicialmente en línea recta horizontal
puntos_cadena: List[List[float]] = []
pos_actual: List[float] = [float(ANCHO // 2), float(ALTO // 2)]
for dist in distancias:
    pos_actual = [pos_actual[0] + dist, pos_actual[1]]
    puntos_cadena.append(list(pos_actual))

# ============================================================================
# VARIABLES DE CONTROL
# ============================================================================

usar_mouse = True  # True = sigue el mouse, False = control con flechas
ancla_pos: List[float] = [
    float(ANCHO // 2),
    float(ALTO // 2),
]  # Posición de la cabeza de la serpiente

# ============================================================================
# FUNCIONES DE FÍSICA
# ============================================================================


def restringir_distancia(
    ancla: List[float], punto: List[float], distancia: float
) -> Tuple[List[float], float]:
    """
    Restringe un punto a una distancia fija desde un ancla.

    Calcula la nueva posición de un punto de manera que esté exactamente
    a la distancia especificada del punto de anclaje, manteniendo la
    dirección original.

    Args:
        ancla: Coordenadas [x, y] del punto de anclaje.
        punto: Coordenadas [x, y] del punto a restringir.
        distancia: Distancia deseada en píxeles desde el ancla.

    Returns:
        Tupla conteniendo:
        - Lista [x, y] con la nueva posición del punto restringido.
        - Ángulo en radianes desde el ancla al nuevo punto.
    """
    dx = punto[0] - ancla[0]
    dy = punto[1] - ancla[1]
    dist_actual = math.sqrt(dx * dx + dy * dy)

    # Caso especial: si la distancia es cero, colocar a la derecha del ancla
    if dist_actual == 0:
        return [ancla[0] + distancia, ancla[1]], 0

    # Normalizar el vector de dirección
    dx_norm = dx / dist_actual
    dy_norm = dy / dist_actual

    # Calcular nueva posición a la distancia exacta
    nuevo_x = ancla[0] + dx_norm * distancia
    nuevo_y = ancla[1] + dy_norm * distancia

    # Calcular el ángulo resultante
    angulo = math.atan2(dy_norm, dx_norm)

    return [nuevo_x, nuevo_y], angulo


def limitar_angulo(
    angulo_actual: float, angulo_anterior: float, limite: float
) -> float:
    """
    Limita el cambio angular entre dos segmentos consecutivos.

    Esta función evita que la serpiente se doble demasiado bruscamente
    al restringir cuánto puede cambiar el ángulo entre un segmento y el siguiente.
    Maneja correctamente el wrapping de ángulos para evitar saltos en la transición
    entre -π y π.

    Args:
        angulo_actual: Ángulo en radianes que tendría el segmento sin restricciones.
        angulo_anterior: Ángulo en radianes del segmento previo.
        limite: Máximo cambio permitido en radianes.

    Returns:
        Ángulo ajustado en radianes que respeta el límite.
    """
    # Calcular la diferencia angular
    diff = angulo_actual - angulo_anterior

    # Normalizar la diferencia al rango [-π, π] para manejar el wrapping
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi

    # Aplicar el límite angular
    if diff > limite:
        return angulo_anterior + limite
    elif diff < -limite:
        return angulo_anterior - limite

    return angulo_actual


def aplicar_restricciones(
    ancla: List[float], punto: List[float], distancia: float, angulo_anterior: float
) -> Tuple[List[float], float]:
    """
    Aplica restricciones de distancia y ángulo a un segmento de la cadena.

    Combina la restricción de distancia fija con la limitación angular para
    crear un movimiento realista de la serpiente que respeta las leyes físicas
    de una cadena articulada.

    Args:
        ancla: Coordenadas [x, y] del punto de anclaje (segmento anterior).
        punto: Coordenadas [x, y] del punto a restringir (posición deseada).
        distancia: Distancia fija que debe mantenerse.
        angulo_anterior: Ángulo en radianes del segmento previo.

    Returns:
        Tupla conteniendo:
        - Lista [x, y] con la nueva posición que respeta ambas restricciones.
        - Ángulo resultante en radianes.
    """
    # Calcular el ángulo hacia la posición deseada
    dx = punto[0] - ancla[0]
    dy = punto[1] - ancla[1]
    angulo_actual = math.atan2(dy, dx)

    # Limitar el cambio angular
    angulo_limitado = limitar_angulo(angulo_actual, angulo_anterior, ANGULO_MAX)

    # Calcular la nueva posición respetando ambas restricciones
    nuevo_x = ancla[0] + distancia * math.cos(angulo_limitado)
    nuevo_y = ancla[1] + distancia * math.sin(angulo_limitado)

    return [nuevo_x, nuevo_y], angulo_limitado


# ============================================================================
# FUNCIONES DE RENDERIZADO
# ============================================================================


def obtener_grosor(indice: int, total: int) -> float:
    """
    Calcula el grosor del cuerpo de la serpiente en un punto específico.

    El grosor varía a lo largo del cuerpo para dar apariencia realista:
    - Sección frontal (0-20%): Adelgazamiento desde la cabeza
    - Sección media (20-70%): Grosor constante
    - Sección trasera (70-100%): Adelgazamiento hacia la cola

    Args:
        indice: Posición del segmento en la cadena (0 = más cercano a la cabeza).
        total: Número total de segmentos.

    Returns:
        Grosor en píxeles para este segmento.
    """
    progreso = indice / total

    # Sección frontal: adelgazamiento desde la cabeza
    if progreso < 0.2:
        return GROSOR_CUERPO_GRUESO - (progreso * 30)

    # Sección media: grosor constante
    elif progreso < 0.7:
        return GROSOR_CUERPO_MEDIO

    # Sección trasera: adelgazamiento hacia la cola
    else:
        return GROSOR_CUERPO_MEDIO - ((progreso - 0.7) * 50)


def dibujar_boton(
    pantalla: pygame.Surface,
    texto: str,
    x: int,
    y: int,
    ancho: int,
    alto: int,
    activo: bool,
) -> pygame.Rect:
    """
    Dibuja un botón con texto centrado.

    Args:
        pantalla: Superficie de Pygame donde dibujar.
        texto: Texto a mostrar en el botón.
        x: Coordenada x de la esquina superior izquierda.
        y: Coordenada y de la esquina superior izquierda.
        ancho: Ancho del botón en píxeles.
        alto: Alto del botón en píxeles.
        activo: Si True, usa color verde; si False, usa gris.

    Returns:
        Objeto pygame.Rect que representa el área del botón.
    """
    color = VERDE_CLARO if activo else GRIS

    # Dibujar fondo del botón con esquinas redondeadas
    pygame.draw.rect(pantalla, color, (x, y, ancho, alto), border_radius=5)
    # Dibujar borde
    pygame.draw.rect(pantalla, BLANCO, (x, y, ancho, alto), 2, border_radius=5)

    # Renderizar y centrar texto
    fuente = pygame.font.Font(None, 24)
    texto_render = fuente.render(texto, True, BLANCO)
    texto_rect = texto_render.get_rect(center=(x + ancho // 2, y + alto // 2))
    pantalla.blit(texto_render, texto_rect)

    return pygame.Rect(x, y, ancho, alto)


# ============================================================================
# LOOP PRINCIPAL DEL JUEGO
# ============================================================================

running = True
boton_rect = pygame.Rect(10, 10, 150, 40)  # Inicializar antes del loop

while running:
    # ========================================================================
    # MANEJO DE EVENTOS
    # ========================================================================

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

    # ========================================================================
    # CONTROL DE ENTRADA
    # ========================================================================

    if not usar_mouse:
        # Modo teclado: mover con flechas direccionales
        teclas = pygame.key.get_pressed()
        if teclas[pygame.K_LEFT]:
            ancla_pos[0] -= VELOCIDAD_TECLADO
        if teclas[pygame.K_RIGHT]:
            ancla_pos[0] += VELOCIDAD_TECLADO
        if teclas[pygame.K_UP]:
            ancla_pos[1] -= VELOCIDAD_TECLADO
        if teclas[pygame.K_DOWN]:
            ancla_pos[1] += VELOCIDAD_TECLADO

        # Mantener la cabeza dentro de la pantalla
        ancla_pos[0] = max(0.0, min(float(ANCHO), ancla_pos[0]))
        ancla_pos[1] = max(0.0, min(float(ALTO), ancla_pos[1]))
    else:
        # Modo mouse: seguir la posición del cursor
        mouse_pos = pygame.mouse.get_pos()
        ancla_pos = [float(mouse_pos[0]), float(mouse_pos[1])]

    # ========================================================================
    # RENDERIZADO
    # ========================================================================

    pantalla.fill(NEGRO)

    # ========================================================================
    # ACTUALIZACIÓN DE LA FÍSICA
    # ========================================================================

    # Actualizar posiciones de la cadena de segmentos
    # Primer segmento: mantener distancia fija desde la cabeza
    puntos_cadena[0], angulo_prev = restringir_distancia(
        ancla_pos, puntos_cadena[0], distancias[0]
    )

    # Segmentos subsecuentes: aplicar restricciones de distancia y ángulo
    for i in range(1, len(puntos_cadena)):
        puntos_cadena[i], angulo_prev = aplicar_restricciones(
            puntos_cadena[i - 1], puntos_cadena[i], distancias[i], angulo_prev
        )

    # ========================================================================
    # CALCULAR GEOMETRÍA DEL CUERPO
    # ========================================================================

    # Listas para almacenar los puntos del contorno del cuerpo
    puntos_izq = []  # Lado izquierdo del cuerpo
    puntos_der = []  # Lado derecho del cuerpo
    puntos_centro = [ancla_pos]  # Línea central (esqueleto)

    # Calcular ángulo de la cabeza
    dx = puntos_cadena[0][0] - ancla_pos[0]
    dy = puntos_cadena[0][1] - ancla_pos[1]
    angulo_cabeza = math.atan2(dy, dx)

    # Calcular puntos iniciales del contorno en la cabeza
    puntos_izq.append(
        (
            int(ancla_pos[0] + GROSOR_CABEZA * math.cos(angulo_cabeza + math.pi / 2)),
            int(ancla_pos[1] + GROSOR_CABEZA * math.sin(angulo_cabeza + math.pi / 2)),
        )
    )
    puntos_der.append(
        (
            int(ancla_pos[0] + GROSOR_CABEZA * math.cos(angulo_cabeza - math.pi / 2)),
            int(ancla_pos[1] + GROSOR_CABEZA * math.sin(angulo_cabeza - math.pi / 2)),
        )
    )

    # Calcular puntos del contorno para cada segmento del cuerpo
    referencia = ancla_pos
    angulo = angulo_cabeza  # Inicializar con el ángulo de la cabeza

    for i, punto in enumerate(puntos_cadena):
        puntos_centro.append(punto)

        # Calcular ángulo desde el segmento anterior
        dx = punto[0] - referencia[0]
        dy = punto[1] - referencia[1]
        angulo = math.atan2(dy, dx)

        # Obtener grosor variable según la posición en el cuerpo
        grosor = obtener_grosor(i, len(puntos_cadena))

        # Calcular puntos perpendiculares al ángulo (lados del cuerpo)
        x_der = punto[0] + grosor * math.cos(angulo + math.pi / 2)
        y_der = punto[1] + grosor * math.sin(angulo + math.pi / 2)
        puntos_der.append((int(x_der), int(y_der)))

        x_izq = punto[0] + grosor * math.cos(angulo - math.pi / 2)
        y_izq = punto[1] + grosor * math.sin(angulo - math.pi / 2)
        puntos_izq.append((int(x_izq), int(y_izq)))

        referencia = punto

    # Calcular punta de la cola usando el último ángulo calculado
    ultimo = puntos_cadena[-1]
    puntos_cola = (
        int(ultimo[0] + EXTENSION_COLA * math.cos(angulo)),
        int(ultimo[1] + EXTENSION_COLA * math.sin(angulo)),
    )

    # Construir el polígono completo del manto (contorno del cuerpo)
    # Lado izquierdo + punta de cola + lado derecho invertido
    manto = puntos_izq + [puntos_cola] + puntos_der[::-1]

    # ========================================================================
    # DIBUJAR CUERPO DE LA SERPIENTE
    # ========================================================================

    # Dibujar el manto (relleno y borde)
    if len(manto) > 2:
        pygame.draw.polygon(pantalla, VERDE_OSCURO, manto)  # Relleno
        pygame.draw.polygon(pantalla, VERDE_CLARO, manto, 3)  # Borde

    # Dibujar líneas decorativas transversales (cada 2 segmentos)
    for i in range(len(puntos_izq) - 1):
        if i % 2 == 0:
            pygame.draw.line(pantalla, VERDE_CLARO, puntos_izq[i], puntos_der[i], 1)

    # Dibujar esqueleto central (línea amarilla)
    for i in range(len(puntos_centro) - 1):
        pygame.draw.line(
            pantalla,
            AMARILLO,
            (int(puntos_centro[i][0]), int(puntos_centro[i][1])),
            (int(puntos_centro[i + 1][0]), int(puntos_centro[i + 1][1])),
            4,
        )

    # ========================================================================
    # DIBUJAR CABEZA DE LA SERPIENTE
    # ========================================================================

    # Círculo de la cabeza (relleno y borde)
    pygame.draw.circle(
        pantalla, VERDE_OSCURO, (int(ancla_pos[0]), int(ancla_pos[1])), RADIO_CABEZA
    )
    pygame.draw.circle(
        pantalla, VERDE_CLARO, (int(ancla_pos[0]), int(ancla_pos[1])), RADIO_CABEZA, 2
    )

    # Calcular posiciones de los ojos
    ojo_izq = (
        int(
            ancla_pos[0]
            + OFFSET_OJO * math.cos(angulo_cabeza + math.pi + ANGULO_SEPARACION_OJOS)
        ),
        int(
            ancla_pos[1]
            + OFFSET_OJO * math.sin(angulo_cabeza + math.pi + ANGULO_SEPARACION_OJOS)
        ),
    )
    ojo_der = (
        int(
            ancla_pos[0]
            + OFFSET_OJO * math.cos(angulo_cabeza + math.pi - ANGULO_SEPARACION_OJOS)
        ),
        int(
            ancla_pos[1]
            + OFFSET_OJO * math.sin(angulo_cabeza + math.pi - ANGULO_SEPARACION_OJOS)
        ),
    )

    # Dibujar ojos (esclerótica amarilla y pupila negra)
    pygame.draw.circle(pantalla, AMARILLO, ojo_izq, RADIO_OJO)
    pygame.draw.circle(pantalla, AMARILLO, ojo_der, RADIO_OJO)
    pygame.draw.circle(pantalla, NEGRO, ojo_izq, RADIO_PUPILA)
    pygame.draw.circle(pantalla, NEGRO, ojo_der, RADIO_PUPILA)

    # ========================================================================
    # DIBUJAR LENGUA BÍFIDA
    # ========================================================================

    # Calcular posición base de la lengua (frente de la cabeza)
    lengua_base = (
        int(ancla_pos[0] + LONGITUD_BASE_LENGUA * math.cos(angulo_cabeza + math.pi)),
        int(ancla_pos[1] + LONGITUD_BASE_LENGUA * math.sin(angulo_cabeza + math.pi)),
    )

    # Calcular puntas bifurcadas de la lengua
    lengua_punta1 = (
        int(
            lengua_base[0]
            + LONGITUD_PUNTA_LENGUA
            * math.cos(angulo_cabeza + math.pi + ANGULO_BIFURCACION_LENGUA)
        ),
        int(
            lengua_base[1]
            + LONGITUD_PUNTA_LENGUA
            * math.sin(angulo_cabeza + math.pi + ANGULO_BIFURCACION_LENGUA)
        ),
    )
    lengua_punta2 = (
        int(
            lengua_base[0]
            + LONGITUD_PUNTA_LENGUA
            * math.cos(angulo_cabeza + math.pi - ANGULO_BIFURCACION_LENGUA)
        ),
        int(
            lengua_base[1]
            + LONGITUD_PUNTA_LENGUA
            * math.sin(angulo_cabeza + math.pi - ANGULO_BIFURCACION_LENGUA)
        ),
    )

    # Dibujar las tres líneas que forman la lengua bífida
    pygame.draw.line(
        pantalla, ROJO, (int(ancla_pos[0]), int(ancla_pos[1])), lengua_base, 3
    )
    pygame.draw.line(pantalla, ROJO, lengua_base, lengua_punta1, 2)
    pygame.draw.line(pantalla, ROJO, lengua_base, lengua_punta2, 2)

    # ========================================================================
    # INTERFAZ DE USUARIO
    # ========================================================================

    # Dibujar botón de control de modo
    modo_texto = "MOUSE" if usar_mouse else "FLECHAS"
    boton_rect = dibujar_boton(pantalla, f"Modo: {modo_texto}", 10, 10, 150, 40, True)

    # Mostrar instrucciones
    fuente = pygame.font.Font(None, 20)
    instrucciones = fuente.render("ESPACIO o click para cambiar modo", True, GRIS)
    pantalla.blit(instrucciones, (10, 55))

    # ========================================================================
    # ACTUALIZAR DISPLAY
    # ========================================================================

    pygame.display.flip()
    clock.tick(FPS)

# ============================================================================
# FINALIZACIÓN
# ============================================================================

pygame.quit()
