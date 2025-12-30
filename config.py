"""Configuración global del juego."""
from dataclasses import dataclass
from typing import List, Tuple
import math


@dataclass
class Config:
    """Configuración centralizada del juego."""

    # Pantalla
    ANCHO: int = 800
    ALTO: int = 600
    FPS: int = 60

    # Colores RGB
    BLANCO: Tuple[int, int, int] = (255, 255, 255)
    ROJO: Tuple[int, int, int] = (255, 0, 0)
    NEGRO: Tuple[int, int, int] = (0, 0, 0)

    # Cadena
    DISTANCIAS_SEGMENTOS: List[int] = None
    ANGULO_MAX_GRADOS: float = 10.0

    # Visualización
    RADIO_ANCLA: int = 10
    RADIO_PUNTO_CADENA: int = 5
    RADIO_PUNTO_PERPENDICULAR: int = 5
    GROSOR_LINEA_BORDE: int = 2
    GROSOR_CIRCULO_DISTANCIA: int = 2

    # Precisión matemática
    EPSILON: float = 1e-6

    def __post_init__(self):
        """Inicializa valores calculados después de la creación."""
        if self.DISTANCIAS_SEGMENTOS is None:
            self.DISTANCIAS_SEGMENTOS = [75] * 2 + [50] * 8 + [25] * 10 + [10] * 10
        self.ANGULO_MAX_RAD = math.radians(self.ANGULO_MAX_GRADOS)


# Instancia global de configuración
CONFIG = Config()
