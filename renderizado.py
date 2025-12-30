"""Funciones de renderizado para visualizar la cadena."""
from typing import List, Tuple
import pygame
import math
from config import CONFIG
from geometria import calcular_angulo


def calcular_puntos_perpendiculares(
    cadena: List[List[float]],
    ancla: List[float],
    distancias: List[float]
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Calcula los puntos perpendiculares izquierdo y derecho de la cadena.

    Args:
        cadena: Lista de puntos de la cadena
        ancla: Posición del ancla [x, y]
        distancias: Lista de distancias (radios) para cada punto

    Returns:
        Tupla con (puntos_izquierda, puntos_derecha)
    """
    puntos_izq = []
    puntos_der = []

    referencia = ancla
    for i, punto in enumerate(cadena):
        angulo = calcular_angulo(referencia, punto)

        # Punto derecho (ángulo + 90 grados)
        x_der = punto[0] + distancias[i] * math.cos(angulo + math.pi/2)
        y_der = punto[1] + distancias[i] * math.sin(angulo + math.pi/2)
        puntos_der.append((int(x_der), int(y_der)))

        # Punto izquierdo (ángulo - 90 grados)
        x_izq = punto[0] + distancias[i] * math.cos(angulo - math.pi/2)
        y_izq = punto[1] + distancias[i] * math.sin(angulo - math.pi/2)
        puntos_izq.append((int(x_izq), int(y_izq)))

        referencia = punto

    return puntos_izq, puntos_der


def dibujar_ancla(pantalla: pygame.Surface, ancla: List[float]) -> None:
    """
    Dibuja el ancla en la posición del mouse.

    Args:
        pantalla: Surface de Pygame donde dibujar
        ancla: Posición del ancla [x, y]
    """
    pygame.draw.circle(
        pantalla,
        CONFIG.BLANCO,
        (int(ancla[0]), int(ancla[1])),
        CONFIG.RADIO_ANCLA
    )


def dibujar_puntos_cadena(
    pantalla: pygame.Surface,
    cadena: List[List[float]],
    distancias: List[float]
) -> None:
    """
    Dibuja los puntos de la cadena y sus círculos de radio.

    Args:
        pantalla: Surface de Pygame donde dibujar
        cadena: Lista de puntos de la cadena
        distancias: Lista de distancias (radios) para cada punto
    """
    for i, punto in enumerate(cadena):
        punto_int = (int(punto[0]), int(punto[1]))

        # Dibujar el punto de la cadena
        pygame.draw.circle(
            pantalla,
            CONFIG.BLANCO,
            punto_int,
            CONFIG.RADIO_PUNTO_CADENA
        )

        # Dibujar círculo de radio igual a la distancia del segmento
        pygame.draw.circle(
            pantalla,
            CONFIG.BLANCO,
            punto_int,
            distancias[i],
            CONFIG.GROSOR_CIRCULO_DISTANCIA
        )


def dibujar_puntos_perpendiculares(
    pantalla: pygame.Surface,
    puntos_izq: List[Tuple[int, int]],
    puntos_der: List[Tuple[int, int]]
) -> None:
    """
    Dibuja los puntos perpendiculares (bordes) de la cadena.

    Args:
        pantalla: Surface de Pygame donde dibujar
        puntos_izq: Lista de puntos del borde izquierdo
        puntos_der: Lista de puntos del borde derecho
    """
    for p_izq, p_der in zip(puntos_izq, puntos_der):
        pygame.draw.circle(
            pantalla,
            CONFIG.ROJO,
            p_izq,
            CONFIG.RADIO_PUNTO_PERPENDICULAR
        )
        pygame.draw.circle(
            pantalla,
            CONFIG.ROJO,
            p_der,
            CONFIG.RADIO_PUNTO_PERPENDICULAR
        )


def dibujar_lineas_borde(
    pantalla: pygame.Surface,
    puntos_izq: List[Tuple[int, int]],
    puntos_der: List[Tuple[int, int]]
) -> None:
    """
    Dibuja las líneas que conectan los puntos perpendiculares.

    Args:
        pantalla: Surface de Pygame donde dibujar
        puntos_izq: Lista de puntos del borde izquierdo
        puntos_der: Lista de puntos del borde derecho
    """
    for i in range(len(puntos_izq) - 1):
        # Borde izquierdo
        pygame.draw.line(
            pantalla,
            CONFIG.ROJO,
            puntos_izq[i],
            puntos_izq[i+1],
            CONFIG.GROSOR_LINEA_BORDE
        )
        # Borde derecho
        pygame.draw.line(
            pantalla,
            CONFIG.ROJO,
            puntos_der[i],
            puntos_der[i+1],
            CONFIG.GROSOR_LINEA_BORDE
        )


def dibujar_cadena(
    pantalla: pygame.Surface,
    ancla: List[float],
    cadena: List[List[float]],
    distancias: List[float]
) -> None:
    """
    Dibuja la cadena completa con todos sus elementos visuales.

    Args:
        pantalla: Surface de Pygame donde dibujar
        ancla: Posición del ancla [x, y]
        cadena: Lista de puntos de la cadena
        distancias: Lista de distancias (radios) para cada punto
    """
    # Dibujar ancla
    dibujar_ancla(pantalla, ancla)

    # Dibujar puntos de la cadena
    dibujar_puntos_cadena(pantalla, cadena, distancias)

    # Calcular puntos perpendiculares
    puntos_izq, puntos_der = calcular_puntos_perpendiculares(cadena, ancla, distancias)

    # Dibujar puntos perpendiculares
    dibujar_puntos_perpendiculares(pantalla, puntos_izq, puntos_der)

    # Dibujar líneas conectoras
    dibujar_lineas_borde(pantalla, puntos_izq, puntos_der)
