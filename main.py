import pygame
import math

# Inicialización de Pygame y configuración de la ventana
pygame.init()
ANCHO, ALTO = 800, 600
pantalla = pygame.display.set_mode((ANCHO, ALTO))
clock = pygame.time.Clock()

# Colores usados en RGB
BLANCO = (255, 255, 255)
ROJO = (255, 0, 0)
NEGRO = (0, 0, 0)

# Lista de distancias entre elementos consecutivos de la cadena.
# Aquí se crean tramos con distintas longitudes: dos de 75, ocho de 50,
# diez de 25 y diez de 10. Puedes ajustar esta lista para cambiar la geometría.
distancias = [75] * 2 + [50] * 8  + [25] * 10 + [10] * 10

# Límite de cambio angular entre segmentos (en radianes).
# Se usa para evitar giros bruscos entre segmentos consecutivos.
ANGULO_MAX = math.radians(30)  # 30 grados

# Construcción inicial de los puntos de la cadena.
# Se empiezan desde el centro de la pantalla y se colocan en línea horizontal.
puntos_cadena = []
pos_actual = [ANCHO // 2, ALTO // 2]
for dist in distancias:
    pos_actual = [pos_actual[0] + dist, pos_actual[1]]
    puntos_cadena.append(list(pos_actual))


def restringir_distancia(ancla, punto, distancia):
    """Ajusta `punto` para que quede exactamente a `distancia` de `ancla`.

    Devuelve la nueva posición y el ángulo (en radianes) desde la ancla
    hasta esa nueva posición.
    """
    dx = punto[0] - ancla[0]
    dy = punto[1] - ancla[1]
    dist_actual = math.sqrt(dx * dx + dy * dy)

    # Si la distancia actual es cero, evitamos división por cero y colocamos
    # el punto a la derecha de la ancla por defecto.
    if dist_actual == 0:
        return [ancla[0] + distancia, ancla[1]], 0

    # Normalizar vector y multiplicar por la distancia deseada
    dx_norm = dx / dist_actual
    dy_norm = dy / dist_actual
    nuevo_x = ancla[0] + dx_norm * distancia
    nuevo_y = ancla[1] + dy_norm * distancia

    # El ángulo se toma sobre el vector normalizado (dy_norm, dx_norm)
    angulo = math.atan2(dy_norm, dx_norm)
    return [nuevo_x, nuevo_y], angulo


def limitar_angulo(angulo_actual, angulo_anterior, limite):
    """Restringe el cambio de ángulo entre `angulo_anterior` y `angulo_actual`.

    Normaliza la diferencia al intervalo [-π, π] y si excede `limite` la
    corrige para que el cambio máximo sea `limite` en sentido positivo o
    negativo.
    """
    diff = angulo_actual - angulo_anterior

    # Normalizar diferencia al rango [-π, π]
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi

    # Aplicar límite
    if diff > limite:
        return angulo_anterior + limite
    elif diff < -limite:
        return angulo_anterior - limite
    return angulo_actual


def aplicar_restricciones(ancla, punto, distancia, angulo_anterior):
    """Calcula la nueva posición de `punto` respetando la distancia fija
    respecto a `ancla` y limitando el cambio de ángulo respecto a
    `angulo_anterior`.

    Devuelve la nueva posición y el ángulo aplicado.
    """
    # Ángulo 'natural' del segmento antes de limitar
    dx = punto[0] - ancla[0]
    dy = punto[1] - ancla[1]
    angulo_actual = math.atan2(dy, dx)

    # Limitar el ángulo usando la función anterior
    angulo_limitado = limitar_angulo(angulo_actual, angulo_anterior, ANGULO_MAX)

    # Calcular la nueva posición a la distancia deseada y con el ángulo limitado
    nuevo_x = ancla[0] + distancia * math.cos(angulo_limitado)
    nuevo_y = ancla[1] + distancia * math.sin(angulo_limitado)

    return [nuevo_x, nuevo_y], angulo_limitado


running = True
while running:
    # Manejo de eventos de Pygame (cerrar ventana, etc.)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # La 'ancla' es la posición actual del ratón
    ancla_pos = list(pygame.mouse.get_pos())
    pantalla.fill(NEGRO)

    # Dibujar la ancla para referencia visual
    pygame.draw.circle(pantalla, BLANCO, (int(ancla_pos[0]), int(ancla_pos[1])), 10)

    # El primer punto de la cadena sigue la ancla pero manteniendo la distancia
    # fijada en `distancias[0]`. Aquí no se aplica límite angular previo.
    puntos_cadena[0], angulo_prev = restringir_distancia(ancla_pos, puntos_cadena[0], distancias[0])

    # Aplicar restricciones de ángulo y distancia a cada segmento posterior
    for i in range(1, len(puntos_cadena)):
        puntos_cadena[i], angulo_prev = aplicar_restricciones(
            puntos_cadena[i-1], 
            puntos_cadena[i], 
            distancias[i], 
            angulo_prev
        )

    # Calculamos y dibujamos los puntos 'izq' y 'der' perpendiculares a cada
    # punto de la cadena para formar una especie de 'banda' a ambos lados.
    puntos_izq = []
    puntos_der = []

    referencia = ancla_pos
    for i, punto in enumerate(puntos_cadena):
        punto_int = (int(punto[0]), int(punto[1]))

        # Dibujar el punto central y un círculo que indica su distancia objetivo
        pygame.draw.circle(pantalla, BLANCO, punto_int, 5)
        pygame.draw.circle(pantalla, BLANCO, punto_int, distancias[i], 2)

        # Vector desde la referencia hasta el punto actual
        dx = punto[0] - referencia[0]
        dy = punto[1] - referencia[1]
        angulo = math.atan2(dy, dx)

        # Punto a la derecha perpendicular al segmento (ángulo + 90°)
        x_der = punto[0] + distancias[i] * math.cos(angulo + math.pi/2)
        y_der = punto[1] + distancias[i] * math.sin(angulo + math.pi/2)
        puntos_der.append((int(x_der), int(y_der)))
        pygame.draw.circle(pantalla, ROJO, (int(x_der), int(y_der)), 5)

        # Punto a la izquierda perpendicular al segmento (ángulo - 90°)
        x_izq = punto[0] + distancias[i] * math.cos(angulo - math.pi/2)
        y_izq = punto[1] + distancias[i] * math.sin(angulo - math.pi/2)
        puntos_izq.append((int(x_izq), int(y_izq)))
        pygame.draw.circle(pantalla, ROJO, (int(x_izq), int(y_izq)), 5)

        referencia = punto

    # Conectar los puntos rojos en cada lado para formar líneas continuas
    for i in range(len(puntos_izq) - 1):
        pygame.draw.line(pantalla, ROJO, puntos_izq[i], puntos_izq[i+1], 2)
        pygame.draw.line(pantalla, ROJO, puntos_der[i], puntos_der[i+1], 2)

    # Mostrar frame y limitar FPS
    pygame.display.flip()
    clock.tick(60)

pygame.quit()