"""
Script para probar visualmente cualquier modelo entrenado.
Uso: python test_modelo.py [ruta_al_modelo]
Ejemplo: python test_modelo.py poblacion/modelo_3.pkl
"""

import sys
import os
import pygame
import math
import random
import pickle
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import numpy as np

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

ANCHO = 1920
ALTO = 1080
FPS = 60

# Colores
NEGRO = (0, 0, 0)
BLANCO = (255, 255, 255)
ROJO = (255, 0, 0)
VERDE = (0, 255, 0)
AZUL = (100, 150, 255)
GRIS_OSCURO = (40, 40, 40)

# Serpiente
RADIO_SEGMENTO = 20
DISTANCIA_OBJETIVO = 30
ANGULO_MAX = 0.2
VELOCIDAD = 5
SEGMENTOS_POR_PUNTO = 5
SEGMENTOS_MINIMOS = 10

# Ratones
VELOCIDAD_RATON = 3
RADIO_RATON = 15

# Agujero
RADIO_AGUJERO = 50
VELOCIDAD_AGUJERO = 2


# ============================================================================
# CLASE MOUSE
# ============================================================================

class Mouse:
    """Ratón que intenta escapar a los bordes de la pantalla."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

        # Elegir un borde aleatorio y establecer destino más allá
        borde = random.choice(['arriba', 'derecha', 'abajo', 'izquierda'])

        if borde == 'arriba':
            self.destino_x = random.uniform(0, ANCHO)
            self.destino_y = -200
        elif borde == 'derecha':
            self.destino_x = ANCHO + 200
            self.destino_y = random.uniform(0, ALTO)
        elif borde == 'abajo':
            self.destino_x = random.uniform(0, ANCHO)
            self.destino_y = ALTO + 200
        else:  # izquierda
            self.destino_x = -200
            self.destino_y = random.uniform(0, ALTO)

    def mover(self):
        """Mueve el ratón hacia su destino."""
        dx = self.destino_x - self.x
        dy = self.destino_y - self.y
        distancia = math.sqrt(dx * dx + dy * dy)

        if distancia > 0:
            self.x += (dx / distancia) * VELOCIDAD_RATON
            self.y += (dy / distancia) * VELOCIDAD_RATON

    def fuera_de_pantalla(self) -> bool:
        """Verifica si el ratón está cerca de salir de la pantalla."""
        margen = 100
        return (
            self.x < -margen or
            self.x > ANCHO + margen or
            self.y < -margen or
            self.y > ALTO + margen
        )

    def dibujar(self, pantalla):
        """Dibuja el ratón."""
        pygame.draw.circle(pantalla, ROJO, (int(self.x), int(self.y)), RADIO_RATON)


# ============================================================================
# CLASE AIAGENT - Versión simplificada solo para jugar
# ============================================================================

class AIAgent:
    """Agente de IA que carga un modelo pre-entrenado para jugar."""

    def __init__(self, archivo_modelo: str = "snake_qlearning_best.pkl"):
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(lambda: np.zeros(4))
        self.archivo_modelo = archivo_modelo

        # Cargar modelo
        if os.path.exists(archivo_modelo):
            self.load(archivo_modelo)
            print(f"✓ Modelo cargado: {archivo_modelo}")
        else:
            print(f"⚠ Advertencia: No se encontró {archivo_modelo}")

    def load(self, filename: str):
        """Carga la Q-table desde un archivo."""
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)

    def discretizar_estado(
        self,
        cabeza_x: float,
        cabeza_y: float,
        ratones: List[Mouse],
        puntos: int
    ) -> Tuple:
        """Convierte el estado continuo en discreto (mismo formato que autogame.py)."""
        # Discretizar posición de la cabeza
        cabeza_cuadrante_x = 0 if cabeza_x < ANCHO // 2 else 1
        cabeza_cuadrante_y = 0 if cabeza_y < ALTO // 2 else 1

        # Variables por defecto cuando no hay ratones
        direccion_cercano_serpiente = 0
        dist_cercano_serpiente = 2
        direccion_cercano_limite = 0
        dist_cercano_limite = 2

        if len(ratones) > 0:
            # 1. RATÓN MÁS CERCANO A LA SERPIENTE
            raton_cercano_serpiente = min(
                ratones,
                key=lambda r: math.sqrt((r.x - cabeza_x) ** 2 + (r.y - cabeza_y) ** 2),
            )

            dx_serpiente = raton_cercano_serpiente.x - cabeza_x
            dy_serpiente = raton_cercano_serpiente.y - cabeza_y
            angulo_serpiente = math.atan2(dy_serpiente, dx_serpiente)
            direccion_cercano_serpiente = int((angulo_serpiente + math.pi) / (math.pi / 4)) % 8

            dist_serpiente = math.sqrt(dx_serpiente * dx_serpiente + dy_serpiente * dy_serpiente)
            if dist_serpiente < 100:
                dist_cercano_serpiente = 0
            elif dist_serpiente < 300:
                dist_cercano_serpiente = 1
            else:
                dist_cercano_serpiente = 2

            # 2. RATÓN MÁS CERCANO AL LÍMITE
            def distancia_al_limite(r: Mouse) -> float:
                """Calcula la distancia más corta de un ratón a cualquier borde."""
                dist_arriba = r.y
                dist_abajo = ALTO - r.y
                dist_izquierda = r.x
                dist_derecha = ANCHO - r.x
                return min(dist_arriba, dist_abajo, dist_izquierda, dist_derecha)

            raton_cercano_limite = min(ratones, key=distancia_al_limite)

            dx_limite = raton_cercano_limite.x - cabeza_x
            dy_limite = raton_cercano_limite.y - cabeza_y
            angulo_limite = math.atan2(dy_limite, dx_limite)
            direccion_cercano_limite = int((angulo_limite + math.pi) / (math.pi / 4)) % 8

            dist_limite = distancia_al_limite(raton_cercano_limite)
            if dist_limite < 100:
                dist_cercano_limite = 0  # Muy cerca del límite (urgente)
            elif dist_limite < 250:
                dist_cercano_limite = 1
            else:
                dist_cercano_limite = 2

        puntos_categoria = min(puntos // 3, 5)
        num_ratones = min(len(ratones), 5)

        return (
            cabeza_cuadrante_x,
            cabeza_cuadrante_y,
            direccion_cercano_serpiente,
            dist_cercano_serpiente,
            direccion_cercano_limite,
            dist_cercano_limite,
            puntos_categoria,
            num_ratones,
        )

    def get_action(
        self,
        cabeza_x: float,
        cabeza_y: float,
        ratones: List[Mouse],
        puntos: int
    ) -> int:
        """Obtiene la mejor acción según la Q-table."""
        estado = self.discretizar_estado(cabeza_x, cabeza_y, ratones, puntos)
        q_values = self.q_table[estado]
        return int(np.argmax(q_values))


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def calcular_num_segmentos(puntos: int) -> int:
    """Calcula el número de segmentos según los puntos."""
    return max(SEGMENTOS_MINIMOS, puntos * SEGMENTOS_POR_PUNTO)


def inicializar_cadena(x_inicial: float, y_inicial: float, num_segmentos: int) -> List[List[float]]:
    """Inicializa la cadena cinemática con posiciones iniciales."""
    puntos_cadena = []
    for i in range(num_segmentos):
        puntos_cadena.append([x_inicial, y_inicial - i * DISTANCIA_OBJETIVO])
    return puntos_cadena


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def jugar_con_modelo(archivo_modelo: str):
    """Ejecuta el juego con un modelo específico."""
    pygame.init()
    pantalla = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption(f"Snake IA - Probando: {os.path.basename(archivo_modelo)}")
    reloj = pygame.time.Clock()
    fuente = pygame.font.Font(None, 36)
    fuente_grande = pygame.font.Font(None, 72)

    # Cargar agente
    agente = AIAgent(archivo_modelo)

    # Variables del juego
    puntos = 1
    game_over = False
    partida_num = 1

    # Posición inicial aleatoria de la serpiente
    x_inicial = float(random.randint(300, ANCHO - 300))
    y_inicial = float(random.randint(300, ALTO - 300))

    num_segmentos = calcular_num_segmentos(puntos)
    puntos_cadena = inicializar_cadena(x_inicial, y_inicial, num_segmentos)

    # Posición del agujero (centro al inicio)
    agujero_x = float(ANCHO // 2)
    agujero_y = float(ALTO // 2)
    agujero_destino_x = float(random.randint(300, ANCHO - 300))
    agujero_destino_y = float(random.randint(300, ALTO - 300))

    # Ratones
    ratones: List[Mouse] = []

    ejecutando = True
    while ejecutando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                ejecutando = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    ejecutando = False

        if not game_over:
            # IA decide acción
            accion = agente.get_action(
                puntos_cadena[0][0],
                puntos_cadena[0][1],
                ratones,
                puntos
            )

            # Aplicar acción (0=arriba, 1=derecha, 2=abajo, 3=izquierda)
            dx, dy = 0, 0
            if accion == 0:  # Arriba
                dy = -VELOCIDAD
            elif accion == 1:  # Derecha
                dx = VELOCIDAD
            elif accion == 2:  # Abajo
                dy = VELOCIDAD
            elif accion == 3:  # Izquierda
                dx = -VELOCIDAD

            puntos_cadena[0][0] += dx
            puntos_cadena[0][1] += dy

            # Restricciones de la cadena
            for i in range(1, len(puntos_cadena)):
                dx_seg = puntos_cadena[i - 1][0] - puntos_cadena[i][0]
                dy_seg = puntos_cadena[i - 1][1] - puntos_cadena[i][1]
                distancia = math.sqrt(dx_seg * dx_seg + dy_seg * dy_seg)

                if distancia > DISTANCIA_OBJETIVO:
                    puntos_cadena[i][0] += dx_seg - (dx_seg / distancia) * DISTANCIA_OBJETIVO
                    puntos_cadena[i][1] += dy_seg - (dy_seg / distancia) * DISTANCIA_OBJETIVO

                if i > 1:
                    dx_prev = puntos_cadena[i - 1][0] - puntos_cadena[i - 2][0]
                    dy_prev = puntos_cadena[i - 1][1] - puntos_cadena[i - 2][1]
                    angulo_prev = math.atan2(dy_prev, dx_prev)
                    angulo_actual = math.atan2(dy_seg, dx_seg)
                    diferencia_angulo = angulo_actual - angulo_prev

                    while diferencia_angulo > math.pi:
                        diferencia_angulo -= 2 * math.pi
                    while diferencia_angulo < -math.pi:
                        diferencia_angulo += 2 * math.pi

                    if abs(diferencia_angulo) > ANGULO_MAX:
                        angulo_nuevo = angulo_prev + math.copysign(ANGULO_MAX, diferencia_angulo)
                        puntos_cadena[i][0] = puntos_cadena[i - 1][0] - math.cos(angulo_nuevo) * DISTANCIA_OBJETIVO
                        puntos_cadena[i][1] = puntos_cadena[i - 1][1] - math.sin(angulo_nuevo) * DISTANCIA_OBJETIVO

            # Mover agujero
            dx_agujero = agujero_destino_x - agujero_x
            dy_agujero = agujero_destino_y - agujero_y
            dist_agujero = math.sqrt(dx_agujero * dx_agujero + dy_agujero * dy_agujero)

            if dist_agujero < 5:
                agujero_destino_x = float(random.randint(300, ANCHO - 300))
                agujero_destino_y = float(random.randint(300, ALTO - 300))
            else:
                agujero_x += (dx_agujero / dist_agujero) * VELOCIDAD_AGUJERO
                agujero_y += (dy_agujero / dist_agujero) * VELOCIDAD_AGUJERO

            # Spawn ratones
            if random.random() < 0.02:
                ratones.append(Mouse(agujero_x, agujero_y))

            # Mover ratones y verificar colisiones
            puntos_antes = puntos
            for raton in ratones[:]:
                raton.mover()

                # Colisión con cabeza
                dx = puntos_cadena[0][0] - raton.x
                dy = puntos_cadena[0][1] - raton.y
                distancia = math.sqrt(dx * dx + dy * dy)

                if distancia < RADIO_SEGMENTO + RADIO_RATON:
                    ratones.remove(raton)
                    puntos += 1
                elif raton.fuera_de_pantalla():
                    ratones.remove(raton)
                    puntos -= 1

            # Ajustar longitud de serpiente
            if puntos != puntos_antes:
                num_segmentos_nuevo = calcular_num_segmentos(puntos)
                num_segmentos_actual = len(puntos_cadena)

                if num_segmentos_nuevo > num_segmentos_actual:
                    for _ in range(num_segmentos_nuevo - num_segmentos_actual):
                        ultimo = puntos_cadena[-1]
                        puntos_cadena.append([ultimo[0], ultimo[1]])
                elif num_segmentos_nuevo < num_segmentos_actual:
                    puntos_cadena = puntos_cadena[:num_segmentos_nuevo]

            # Game over
            if puntos <= 0:
                game_over = True
                print(f"Game Over - Partida {partida_num}: {puntos} puntos")

        # Dibujar
        pantalla.fill(NEGRO)

        # Agujero
        pygame.draw.circle(pantalla, GRIS_OSCURO, (int(agujero_x), int(agujero_y)), RADIO_AGUJERO)

        # Ratones
        for raton in ratones:
            raton.dibujar(pantalla)

        # Serpiente
        for i, punto in enumerate(puntos_cadena):
            color = VERDE if i == 0 else BLANCO
            pygame.draw.circle(pantalla, color, (int(punto[0]), int(punto[1])), RADIO_SEGMENTO)

        # HUD
        texto_puntos = fuente.render(f"Puntos: {puntos}", True, BLANCO)
        texto_partida = fuente.render(f"Partida: {partida_num}", True, BLANCO)
        texto_modelo = fuente.render(f"Modelo: {os.path.basename(archivo_modelo)}", True, AZUL)

        pantalla.blit(texto_puntos, (20, 20))
        pantalla.blit(texto_partida, (20, 60))
        pantalla.blit(texto_modelo, (20, 100))

        if game_over:
            texto_game_over = fuente_grande.render("GAME OVER", True, ROJO)
            pantalla.blit(
                texto_game_over,
                (ANCHO // 2 - texto_game_over.get_width() // 2, ALTO // 2 - 50)
            )

            # Auto-restart
            pygame.time.wait(1000)

            # Reiniciar
            puntos = 1
            game_over = False
            partida_num += 1

            x_inicial = float(random.randint(300, ANCHO - 300))
            y_inicial = float(random.randint(300, ALTO - 300))
            puntos_cadena = inicializar_cadena(x_inicial, y_inicial, calcular_num_segmentos(puntos))

            agujero_x = float(ANCHO // 2)
            agujero_y = float(ALTO // 2)
            agujero_destino_x = float(random.randint(300, ANCHO - 300))
            agujero_destino_y = float(random.randint(300, ALTO - 300))

            ratones.clear()

        pygame.display.flip()
        reloj.tick(FPS)

    pygame.quit()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        archivo = sys.argv[1]
    else:
        archivo = "snake_qlearning_best.pkl"

    if not os.path.exists(archivo):
        print(f"Error: No se encontró el archivo '{archivo}'")
        print("\nUso:")
        print("  python test_modelo.py                    # Prueba el mejor modelo")
        print("  python test_modelo.py poblacion/modelo_3.pkl  # Prueba un modelo específico")
        sys.exit(1)

    print(f"Probando modelo: {archivo}")
    print("Presiona ESC para salir")
    jugar_con_modelo(archivo)
