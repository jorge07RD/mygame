"""Funciones de física para la simulación de la cadena."""
from typing import List, Tuple
import math
from config import CONFIG
from geometria import calcular_angulo


def restringir_distancia(
    ancla: List[float],
    punto: List[float],
    distancia: float
) -> Tuple[List[float], float]:
    """
    Mantiene un punto a una distancia fija de su ancla, preservando la dirección.

    Args:
        ancla: Posición [x, y] del punto ancla
        punto: Posición [x, y] del punto a restringir
        distancia: Distancia fija a mantener

    Returns:
        Tupla con (nueva_posición, ángulo_del_segmento)
    """
    if distancia <= 0:
        raise ValueError(f"La distancia debe ser positiva, se recibió: {distancia}")

    # Calcular vector desde el ancla al punto
    dx = punto[0] - ancla[0]
    dy = punto[1] - ancla[1]
    dist_actual = math.sqrt(dx * dx + dy * dy)

    # Si el punto está sobre el ancla, colocarlo horizontalmente
    if dist_actual < CONFIG.EPSILON:
        return [ancla[0] + distancia, ancla[1]], 0.0

    # Normalizar el vector de dirección
    dx_norm = dx / dist_actual
    dy_norm = dy / dist_actual

    # Calcular nueva posición a la distancia correcta
    nuevo_x = ancla[0] + dx_norm * distancia
    nuevo_y = ancla[1] + dy_norm * distancia

    # Calcular el ángulo del segmento
    angulo = math.atan2(dy_norm, dx_norm)
    return [nuevo_x, nuevo_y], angulo


def limitar_angulo(
    angulo_actual: float,
    angulo_anterior: float,
    limite: float
) -> float:
    """
    Limita el cambio de ángulo entre dos segmentos consecutivos.
    Evita que la cadena se doble demasiado bruscamente.

    Args:
        angulo_actual: Ángulo natural hacia donde quiere ir el punto (radianes)
        angulo_anterior: Ángulo del segmento anterior (radianes)
        limite: Límite máximo de cambio angular (radianes)

    Returns:
        Ángulo limitado (radianes)
    """
    # Calcular la diferencia de ángulo entre segmentos
    diff = angulo_actual - angulo_anterior

    # Normalizar la diferencia al rango [-π, π] para manejar el salto en ±π
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi

    # Aplicar el límite de ángulo
    if diff > limite:
        return angulo_anterior + limite  # Giro máximo hacia la izquierda
    elif diff < -limite:
        return angulo_anterior - limite  # Giro máximo hacia la derecha
    return angulo_actual  # El ángulo está dentro del límite permitido


def aplicar_restricciones(
    ancla: List[float],
    punto: List[float],
    distancia: float,
    angulo_anterior: float
) -> Tuple[List[float], float]:
    """
    Aplica restricciones de distancia y ángulo a un punto de la cadena.
    Combina la restricción de distancia fija con el límite de cambio angular.

    Args:
        ancla: Posición [x, y] del punto ancla
        punto: Posición [x, y] del punto a restringir
        distancia: Distancia fija del segmento
        angulo_anterior: Ángulo del segmento anterior (radianes)

    Returns:
        Tupla con (nueva_posición, ángulo_limitado)
    """
    # Calcular el ángulo natural hacia donde quiere ir el punto
    angulo_actual = calcular_angulo(ancla, punto)

    # Aplicar el límite de ángulo respecto al segmento anterior
    angulo_limitado = limitar_angulo(angulo_actual, angulo_anterior, CONFIG.ANGULO_MAX_RAD)

    # Calcular la nueva posición usando el ángulo limitado y la distancia fija
    nuevo_x = ancla[0] + distancia * math.cos(angulo_limitado)
    nuevo_y = ancla[1] + distancia * math.sin(angulo_limitado)

    return [nuevo_x, nuevo_y], angulo_limitado


def actualizar_cadena(
    puntos_cadena: List[List[float]],
    ancla_pos: List[float],
    distancias: List[float]
) -> None:
    """
    Actualiza todas las posiciones de la cadena siguiendo las restricciones de física.

    Args:
        puntos_cadena: Lista de puntos de la cadena (se modifica in-place)
        ancla_pos: Posición del ancla [x, y]
        distancias: Lista de distancias entre puntos
    """
    # El primer punto sigue al ancla sin restricción de ángulo
    puntos_cadena[0], angulo_prev = restringir_distancia(
        ancla_pos,
        puntos_cadena[0],
        distancias[0]
    )

    # Los demás puntos siguen al anterior con restricción de ángulo
    for i in range(1, len(puntos_cadena)):
        puntos_cadena[i], angulo_prev = aplicar_restricciones(
            puntos_cadena[i-1],  # Punto ancla (anterior en la cadena)
            puntos_cadena[i],    # Punto actual a actualizar
            distancias[i],       # Distancia fija del segmento
            angulo_prev          # Ángulo del segmento anterior
        )


def inicializar_cadena(distancias: List[float], pos_inicial: List[float]) -> List[List[float]]:
    """
    Inicializa la cadena de puntos en posiciones horizontales.

    Args:
        distancias: Lista de distancias entre puntos
        pos_inicial: Posición inicial [x, y]

    Returns:
        Lista de puntos de la cadena
    """
    puntos_cadena = []
    pos_actual = list(pos_inicial)

    for dist in distancias:
        # Cada punto se coloca a la derecha del anterior según su distancia
        pos_actual = [pos_actual[0] + dist, pos_actual[1]]
        puntos_cadena.append(list(pos_actual))

    return puntos_cadena
