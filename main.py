"""
Simulación de serpiente con física de cadena cinemática.
Implementa una serpiente que sigue al mouse o se controla con las flechas del teclado,
con restricciones de distancia y ángulo entre segmentos para movimiento realista.
"""

from typing import List, Tuple
import pygame
import math
import random

# ============================================================================
# CONFIGURACIÓN E INICIALIZACIÓN
# ============================================================================

pygame.init()

# ============================================================================
# CLASE MOUSE (RATÓN)
# ============================================================================


class Mouse:
    """
    Representa un ratón que sale del agujero y se mueve hacia un destino aleatorio.
    """

    def __init__(self, x: float, y: float, ancho_pantalla: int, alto_pantalla: int):
        """
        Inicializa un ratón en una posición dada.

        Args:
            x: Posición inicial x (centro de la pantalla).
            y: Posición inicial y (centro de la pantalla).
            ancho_pantalla: Ancho de la pantalla para calcular destino aleatorio.
            alto_pantalla: Alto de la pantalla para calcular destino aleatorio.
        """
        self.x = x
        self.y = y
        self.radio = 12  # Tamaño del ratón
        self.velocidad = 2  # Velocidad de movimiento

        # Calcular destino aleatorio FUERA de la pantalla
        # Elegir un borde aleatorio (0=arriba, 1=derecha, 2=abajo, 3=izquierda)
        borde = random.randint(0, 3)
        margen_fuera = 200  # Qué tan lejos fuera de la pantalla va el destino

        if borde == 0:  # Arriba
            self.destino_x = random.randint(-margen_fuera, ancho_pantalla + margen_fuera)
            self.destino_y = -margen_fuera
        elif borde == 1:  # Derecha
            self.destino_x = ancho_pantalla + margen_fuera
            self.destino_y = random.randint(-margen_fuera, alto_pantalla + margen_fuera)
        elif borde == 2:  # Abajo
            self.destino_x = random.randint(-margen_fuera, ancho_pantalla + margen_fuera)
            self.destino_y = alto_pantalla + margen_fuera
        else:  # Izquierda
            self.destino_x = -margen_fuera
            self.destino_y = random.randint(-margen_fuera, alto_pantalla + margen_fuera)

        # Color del ratón
        self.color = (150, 150, 150)  # Gris
        self.color_orejas = (200, 150, 150)  # Rosa claro

    def actualizar(self) -> None:
        """
        Mueve el ratón hacia su destino (fuera de la pantalla).
        """
        # Calcular dirección hacia el destino
        dx = self.destino_x - self.x
        dy = self.destino_y - self.y
        distancia = math.sqrt(dx * dx + dy * dy)

        # Moverse siempre hacia el destino (no detenerse)
        if distancia > 0:
            self.x += (dx / distancia) * self.velocidad
            self.y += (dy / distancia) * self.velocidad

    def dibujar(self, pantalla: pygame.Surface) -> None:
        """
        Dibuja el ratón en la pantalla.

        Args:
            pantalla: Superficie de Pygame donde dibujar.
        """
        # Calcular ángulo de movimiento para orientar el ratón
        dx = self.destino_x - self.x
        dy = self.destino_y - self.y
        angulo = math.atan2(dy, dx)

        # Dibujar cuerpo (círculo principal)
        pygame.draw.circle(
            pantalla, self.color, (int(self.x), int(self.y)), self.radio
        )

        # Dibujar orejas (dos círculos pequeños)
        offset_oreja = 8
        oreja_izq = (
            int(self.x + offset_oreja * math.cos(angulo + math.pi / 2 + 0.3)),
            int(self.y + offset_oreja * math.sin(angulo + math.pi / 2 + 0.3)),
        )
        oreja_der = (
            int(self.x + offset_oreja * math.cos(angulo - math.pi / 2 - 0.3)),
            int(self.y + offset_oreja * math.sin(angulo - math.pi / 2 - 0.3)),
        )
        pygame.draw.circle(pantalla, self.color_orejas, oreja_izq, 5)
        pygame.draw.circle(pantalla, self.color_orejas, oreja_der, 5)

        # Dibujar nariz (pequeño punto en la dirección de movimiento)
        nariz = (
            int(self.x + (self.radio - 2) * math.cos(angulo)),
            int(self.y + (self.radio - 2) * math.sin(angulo)),
        )
        pygame.draw.circle(pantalla, (50, 50, 50), nariz, 2)

        # Dibujar cola (línea curva detrás del ratón)
        cola_inicio = (
            int(self.x - self.radio * math.cos(angulo)),
            int(self.y - self.radio * math.sin(angulo)),
        )
        cola_fin = (
            int(self.x - (self.radio + 15) * math.cos(angulo + 0.5)),
            int(self.y - (self.radio + 15) * math.sin(angulo + 0.5)),
        )
        pygame.draw.line(pantalla, self.color, cola_inicio, cola_fin, 2)

    def fuera_de_pantalla(self, ancho: int, alto: int) -> bool:
        """
        Verifica si el ratón se ha acercado al borde de la pantalla.

        Args:
            ancho: Ancho de la pantalla.
            alto: Alto de la pantalla.

        Returns:
            True si el ratón está cerca del borde, False en caso contrario.
        """
        margen = 100  # Distancia del borde para considerar que está fuera
        return (
            self.x < margen
            or self.x > ancho - margen
            or self.y < margen
            or self.y > alto - margen
        )


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
SEGMENTOS_POR_PUNTO = 5  # Número de segmentos por cada punto
SEGMENTOS_MINIMOS = 10  # Número mínimo de segmentos
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

# ============================================================================
# INICIALIZACIÓN DE LA CADENA
# ============================================================================


def calcular_num_segmentos(puntos: int) -> int:
    """
    Calcula el número de segmentos basado en los puntos actuales.

    Args:
        puntos: Puntos actuales del jugador.

    Returns:
        Número de segmentos para la serpiente.
    """
    return max(SEGMENTOS_MINIMOS, puntos * SEGMENTOS_POR_PUNTO)


def inicializar_cadena(num_segmentos: int) -> List[List[float]]:
    """
    Crea una cadena de segmentos inicial en línea recta.

    Args:
        num_segmentos: Número de segmentos a crear.

    Returns:
        Lista de posiciones de los segmentos.
    """
    cadena = []
    pos_actual = [float(ANCHO // 2), float(ALTO // 2)]
    for _ in range(num_segmentos):
        pos_actual = [pos_actual[0] + DISTANCIA_SEGMENTO, pos_actual[1]]
        cadena.append(list(pos_actual))
    return cadena


# Crear la cadena de segmentos inicialmente
num_segmentos_inicial = calcular_num_segmentos(1)  # Comienza con 1 punto
puntos_cadena: List[List[float]] = inicializar_cadena(num_segmentos_inicial)

# ============================================================================
# VARIABLES DE CONTROL
# ============================================================================

usar_mouse = True  # True = sigue el mouse, False = control con flechas
ancla_pos: List[float] = [
    float(ANCHO // 2),
    float(ALTO // 2),
]  # Posición de la cabeza de la serpiente

# ============================================================================
# VARIABLES DEL SISTEMA DE JUEGO
# ============================================================================

# Sistema de puntuación
puntos = 1  # El jugador comienza con 1 punto
game_over = False

# Lista de ratones activos
ratones: List[Mouse] = []

# Control de spawn de ratones
tiempo_ultimo_spawn = 0
intervalo_spawn = 2000  # Milisegundos entre cada spawn de ratón (2 segundos)

# Posición del agujero (centro de la pantalla)
AGUJERO_X = ANCHO // 2
AGUJERO_Y = ALTO // 2
RADIO_AGUJERO = 40

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

            # Reiniciar juego si está en Game Over
            if event.key == pygame.K_r and game_over:
                game_over = False
                puntos = 1
                ratones.clear()
                tiempo_ultimo_spawn = 0
                # Reiniciar serpiente a longitud inicial
                num_segmentos_inicial = calcular_num_segmentos(1)
                puntos_cadena = inicializar_cadena(num_segmentos_inicial)

            # Salir del juego si está en Game Over
            if event.key == pygame.K_ESCAPE and game_over:
                running = False

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
    # LÓGICA DEL JUEGO - SPAWN Y ACTUALIZACIÓN DE RATONES
    # ========================================================================

    if not game_over:
        # Spawn de nuevos ratones desde el agujero
        tiempo_actual = pygame.time.get_ticks()
        if tiempo_actual - tiempo_ultimo_spawn > intervalo_spawn:
            nuevo_raton = Mouse(float(AGUJERO_X), float(AGUJERO_Y), ANCHO, ALTO)
            ratones.append(nuevo_raton)
            tiempo_ultimo_spawn = tiempo_actual

        # Actualizar posición de todos los ratones
        for raton in ratones:
            raton.actualizar()

        # Verificar colisiones con la cabeza de la serpiente
        ratones_a_eliminar = []
        puntos_ganados = 0
        puntos_perdidos = 0

        for i, raton in enumerate(ratones):
            # Calcular distancia entre la cabeza de la serpiente y el ratón
            dx = raton.x - ancla_pos[0]
            dy = raton.y - ancla_pos[1]
            distancia = math.sqrt(dx * dx + dy * dy)

            # Si la cabeza toca al ratón
            if distancia < (RADIO_CABEZA + raton.radio):
                ratones_a_eliminar.append(i)
                puntos_ganados += 1

        # Verificar ratones que salieron de la pantalla
        for i, raton in enumerate(ratones):
            if i not in ratones_a_eliminar and raton.fuera_de_pantalla(ANCHO, ALTO):
                ratones_a_eliminar.append(i)
                puntos_perdidos += 1

        # Eliminar ratones capturados o fuera de pantalla
        for i in sorted(ratones_a_eliminar, reverse=True):
            del ratones[i]

        # Actualizar puntos
        puntos_antes = puntos
        puntos += puntos_ganados - puntos_perdidos

        # Verificar condición de Game Over
        if puntos <= 0:
            game_over = True
            puntos = 0

        # Ajustar longitud de la serpiente si cambiaron los puntos
        if puntos != puntos_antes:
            num_segmentos_nuevo = calcular_num_segmentos(puntos)
            num_segmentos_actual = len(puntos_cadena)

            # Si necesitamos más segmentos, agregarlos al final
            if num_segmentos_nuevo > num_segmentos_actual:
                segmentos_a_agregar = num_segmentos_nuevo - num_segmentos_actual
                for _ in range(segmentos_a_agregar):
                    # Agregar nuevo segmento en la dirección del último
                    if len(puntos_cadena) >= 2:
                        # Calcular dirección del último segmento
                        dx = puntos_cadena[-1][0] - puntos_cadena[-2][0]
                        dy = puntos_cadena[-1][1] - puntos_cadena[-2][1]
                        dist = math.sqrt(dx * dx + dy * dy)
                        if dist > 0:
                            dx_norm = dx / dist
                            dy_norm = dy / dist
                        else:
                            dx_norm, dy_norm = 1, 0
                    else:
                        dx_norm, dy_norm = 1, 0

                    # Agregar nuevo segmento
                    ultimo = puntos_cadena[-1]
                    nuevo_segmento = [
                        ultimo[0] + dx_norm * DISTANCIA_SEGMENTO,
                        ultimo[1] + dy_norm * DISTANCIA_SEGMENTO,
                    ]
                    puntos_cadena.append(nuevo_segmento)

            # Si necesitamos menos segmentos, quitarlos del final
            elif num_segmentos_nuevo < num_segmentos_actual:
                segmentos_a_quitar = num_segmentos_actual - num_segmentos_nuevo
                for _ in range(segmentos_a_quitar):
                    if len(puntos_cadena) > SEGMENTOS_MINIMOS:
                        puntos_cadena.pop()

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
        ancla_pos, puntos_cadena[0], DISTANCIA_SEGMENTO
    )

    # Segmentos subsecuentes: aplicar restricciones de distancia y ángulo
    for i in range(1, len(puntos_cadena)):
        puntos_cadena[i], angulo_prev = aplicar_restricciones(
            puntos_cadena[i - 1], puntos_cadena[i], DISTANCIA_SEGMENTO, angulo_prev
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
    # DIBUJAR AGUJERO Y RATONES
    # ========================================================================

    # Dibujar agujero en el centro (círculo oscuro)
    pygame.draw.circle(pantalla, (30, 30, 30), (AGUJERO_X, AGUJERO_Y), RADIO_AGUJERO)
    pygame.draw.circle(pantalla, (60, 60, 60), (AGUJERO_X, AGUJERO_Y), RADIO_AGUJERO, 3)

    # Dibujar todos los ratones
    for raton in ratones:
        raton.dibujar(pantalla)

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

    # Mostrar puntuación
    fuente_puntos = pygame.font.Font(None, 48)
    # Color rojo solo cuando estás a punto de perder (1 punto o menos)
    color_puntos = VERDE_CLARO if puntos > 1 else ROJO
    texto_puntos = fuente_puntos.render(f"Puntos: {puntos}", True, color_puntos)
    pantalla.blit(texto_puntos, (ANCHO - 220, 20))

    # Mostrar cantidad de ratones activos
    fuente_ratones = pygame.font.Font(None, 24)
    texto_ratones = fuente_ratones.render(
        f"Ratones: {len(ratones)}", True, GRIS
    )
    pantalla.blit(texto_ratones, (ANCHO - 220, 70))

    # Mostrar longitud de la serpiente
    texto_longitud = fuente_ratones.render(
        f"Longitud: {len(puntos_cadena)} segmentos", True, GRIS
    )
    pantalla.blit(texto_longitud, (ANCHO - 220, 95))

    # Pantalla de Game Over
    if game_over:
        # Crear overlay semi-transparente
        overlay = pygame.Surface((ANCHO, ALTO))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        pantalla.blit(overlay, (0, 0))

        # Texto de Game Over
        fuente_game_over = pygame.font.Font(None, 120)
        texto_game_over = fuente_game_over.render("GAME OVER", True, ROJO)
        rect_game_over = texto_game_over.get_rect(center=(ANCHO // 2, ALTO // 2 - 50))
        pantalla.blit(texto_game_over, rect_game_over)

        # Texto de reinicio
        fuente_reinicio = pygame.font.Font(None, 36)
        texto_reinicio = fuente_reinicio.render(
            "Presiona R para reiniciar o ESC para salir", True, BLANCO
        )
        rect_reinicio = texto_reinicio.get_rect(center=(ANCHO // 2, ALTO // 2 + 50))
        pantalla.blit(texto_reinicio, rect_reinicio)

    # ========================================================================
    # ACTUALIZAR DISPLAY
    # ========================================================================

    pygame.display.flip()
    clock.tick(FPS)

# ============================================================================
# FINALIZACIÓN
# ============================================================================

pygame.quit()
