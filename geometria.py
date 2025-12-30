"""Clases y funciones geométricas para el manejo de puntos y ángulos."""
from dataclasses import dataclass
from typing import Tuple
import math


@dataclass
class Punto2D:
    """Representa un punto en el plano 2D."""

    x: float
    y: float

    def distancia_a(self, otro: 'Punto2D') -> float:
        """
        Calcula la distancia euclidiana a otro punto.

        Args:
            otro: Otro punto 2D

        Returns:
            Distancia entre los dos puntos
        """
        dx = self.x - otro.x
        dy = self.y - otro.y
        return math.sqrt(dx * dx + dy * dy)

    def angulo_hacia(self, otro: 'Punto2D') -> float:
        """
        Calcula el ángulo desde este punto hacia otro punto.

        Args:
            otro: Punto de destino

        Returns:
            Ángulo en radianes
        """
        return math.atan2(otro.y - self.y, otro.x - self.x)

    def a_tupla(self) -> Tuple[int, int]:
        """
        Convierte a tupla de enteros para uso con Pygame.

        Returns:
            Tupla (x, y) con valores enteros
        """
        return (int(self.x), int(self.y))

    def a_lista(self) -> list:
        """
        Convierte a lista de floats.

        Returns:
            Lista [x, y]
        """
        return [self.x, self.y]

    @staticmethod
    def desde_lista(coords: list) -> 'Punto2D':
        """
        Crea un Punto2D desde una lista [x, y].

        Args:
            coords: Lista con coordenadas [x, y]

        Returns:
            Nuevo punto 2D
        """
        return Punto2D(coords[0], coords[1])

    def __add__(self, otro: 'Punto2D') -> 'Punto2D':
        """Suma de vectores."""
        return Punto2D(self.x + otro.x, self.y + otro.y)

    def __sub__(self, otro: 'Punto2D') -> 'Punto2D':
        """Resta de vectores."""
        return Punto2D(self.x - otro.x, self.y - otro.y)


def calcular_angulo(punto_origen: list, punto_destino: list) -> float:
    """
    Calcula el ángulo entre dos puntos dados como listas.

    Args:
        punto_origen: Punto de referencia [x, y]
        punto_destino: Punto hacia el que apunta el ángulo [x, y]

    Returns:
        Ángulo en radianes
    """
    dx = punto_destino[0] - punto_origen[0]
    dy = punto_destino[1] - punto_origen[1]
    return math.atan2(dy, dx)


def normalizar_angulo(angulo: float) -> float:
    """
    Normaliza un ángulo al rango [-π, π].

    Args:
        angulo: Ángulo en radianes

    Returns:
        Ángulo normalizado
    """
    while angulo > math.pi:
        angulo -= 2 * math.pi
    while angulo < -math.pi:
        angulo += 2 * math.pi
    return angulo
