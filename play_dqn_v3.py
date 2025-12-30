"""
Visualizador del agente DQN V3 jugando en tiempo real.
"""

import pygame
import math
import random
import numpy as np
import torch
from typing import List
from dqn_training_v3 import DQNAgent, Mouse, DEVICE


# ============================================================================
# CONFIGURACIÓN VISUAL
# ============================================================================

ANCHO = 1920
ALTO = 1080
FPS = 60

# Colores
NEGRO = (0, 0, 0)
BLANCO = (255, 255, 255)
VERDE_OSCURO = (34, 120, 50)
VERDE_CLARO = (50, 180, 70)
AMARILLO = (220, 200, 50)
ROJO = (200, 50, 50)
GRIS = (100, 100, 100)
AZUL = (50, 150, 255)
NARANJA = (255, 140, 0)

# Serpiente
RADIO_CABEZA = 24
DISTANCIA_SEGMENTO = 30
SEGMENTOS_POR_PUNTO = 5
SEGMENTOS_MINIMOS = 10


# ============================================================================
# FUNCIONES DE FÍSICA (simplificadas)
# ============================================================================

def calcular_num_segmentos(puntos: int) -> int:
    return max(SEGMENTOS_MINIMOS, puntos * SEGMENTOS_POR_PUNTO)


def inicializar_cadena(num_segs: int, x: float, y: float):
    cadena = []
    for i in range(num_segs):
        cadena.append([x, y - i * DISTANCIA_SEGMENTO])
    return cadena


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def jugar_con_dqn():
    """Ejecuta el juego visual con DQN."""
    pygame.init()

    pantalla = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption("Snake DQN V3 - Deep Q-Learning")
    reloj = pygame.time.Clock()

    # Cargar agente
    print("Cargando modelo DQN...")
    agent = DQNAgent(state_size=20, action_size=4)
    agent.load("snake_dqn_v3_best.pth")
    agent.epsilon = 0.0  # Sin exploración
    print("Modelo cargado!")

    # Variables del juego
    puntos = 1
    game_over = False
    partida_num = 1
    pasos = 0
    recompensa_total = 0

    # Serpiente
    cabeza_x = float(ANCHO // 2)
    cabeza_y = float(ALTO // 2)
    vx, vy = 0.0, 0.0

    puntos_cadena = inicializar_cadena(calcular_num_segmentos(puntos), cabeza_x, cabeza_y)

    # Agujero y ratones
    agujero_x = float(ANCHO // 2)
    agujero_y = float(ALTO // 2)
    radio_agujero = 40

    ratones: List[Mouse] = []
    tiempo_spawn = 0
    intervalo_spawn = 60

    # UI
    fuente = pygame.font.Font(None, 36)
    fuente_pequeña = pygame.font.Font(None, 24)

    ejecutando = True

    while ejecutando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                ejecutando = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    ejecutando = False
                if evento.key == pygame.K_r:
                    # Reiniciar
                    puntos = 1
                    game_over = False
                    pasos = 0
                    recompensa_total = 0
                    cabeza_x = float(ANCHO // 2)
                    cabeza_y = float(ALTO // 2)
                    vx, vy = 0.0, 0.0
                    puntos_cadena = inicializar_cadena(calcular_num_segmentos(puntos), cabeza_x, cabeza_y)
                    ratones.clear()
                    tiempo_spawn = 0
                    partida_num += 1
                    print(f"\n--- Partida {partida_num} ---")

        if not game_over:
            # Construir estado
            state = np.zeros(20, dtype=np.float32)

            # Posición cabeza
            state[0] = cabeza_x / ANCHO
            state[1] = cabeza_y / ALTO

            # Velocidad
            state[2] = vx / 10.0
            state[3] = vy / 10.0

            if len(ratones) > 0:
                # Ratón más cercano
                raton_cercano = min(
                    ratones,
                    key=lambda r: math.sqrt((r.x - cabeza_x)**2 + (r.y - cabeza_y)**2)
                )

                dx = raton_cercano.x - cabeza_x
                dy = raton_cercano.y - cabeza_y
                dist = math.sqrt(dx * dx + dy * dy)

                state[4] = min(dist / 1000.0, 1.0)
                state[5] = math.atan2(dy, dx) / math.pi

                state[6] = (raton_cercano.destino_x - cabeza_x) / ANCHO
                state[7] = (raton_cercano.destino_y - cabeza_y) / ALTO

                # Ratón urgente
                def dist_borde(r):
                    return min(r.y, ALTO - r.y, r.x, ANCHO - r.x)

                raton_urgente = min(ratones, key=dist_borde)
                dist_urgente = dist_borde(raton_urgente)

                state[8] = dist_urgente / (ANCHO / 2)
                dx_urg = raton_urgente.x - cabeza_x
                dy_urg = raton_urgente.y - cabeza_y
                state[9] = math.atan2(dy_urg, dx_urg) / math.pi

                state[19] = 1.0 if dist_urgente < 100 else 0.0

            state[10] = min(len(ratones) / 10.0, 1.0)
            state[11] = min(puntos / 100.0, 1.0)

            state[12] = cabeza_y / ALTO
            state[13] = (ANCHO - cabeza_x) / ANCHO
            state[14] = (ALTO - cabeza_y) / ALTO
            state[15] = cabeza_x / ANCHO

            state[16] = agujero_x / ANCHO
            state[17] = agujero_y / ALTO

            # DQN decide
            accion = agent.select_action(state, training=False)

            # Aplicar acción
            velocidad = 5.0
            if accion == 0:  # Arriba
                vx, vy = 0, -velocidad
            elif accion == 1:  # Derecha
                vx, vy = velocidad, 0
            elif accion == 2:  # Abajo
                vx, vy = 0, velocidad
            elif accion == 3:  # Izquierda
                vx, vy = -velocidad, 0

            cabeza_x += vx
            cabeza_y += vy

            # Límites
            cabeza_x = max(0, min(ANCHO, cabeza_x))
            cabeza_y = max(0, min(ALTO, cabeza_y))

            # Actualizar serpiente (simple follow)
            puntos_cadena[0] = [cabeza_x, cabeza_y]
            for i in range(1, len(puntos_cadena)):
                dx = puntos_cadena[i-1][0] - puntos_cadena[i][0]
                dy = puntos_cadena[i-1][1] - puntos_cadena[i][1]
                dist = math.sqrt(dx*dx + dy*dy)

                if dist > DISTANCIA_SEGMENTO:
                    puntos_cadena[i][0] += dx - (dx/dist) * DISTANCIA_SEGMENTO
                    puntos_cadena[i][1] += dy - (dy/dist) * DISTANCIA_SEGMENTO

            # Spawn ratones
            tiempo_spawn += 1
            if tiempo_spawn >= intervalo_spawn:
                ratones.append(Mouse(agujero_x, agujero_y, ANCHO, ALTO))
                tiempo_spawn = 0

            # Actualizar ratones
            for raton in ratones:
                raton.actualizar()

            # Colisiones
            puntos_antes = puntos
            reward = 0

            for raton in ratones[:]:
                dx = raton.x - cabeza_x
                dy = raton.y - cabeza_y
                distancia = math.sqrt(dx * dx + dy * dy)

                if distancia < RADIO_CABEZA + raton.radio:
                    ratones.remove(raton)
                    puntos += 1
                    reward += 50
                elif raton.fuera_de_pantalla(ANCHO, ALTO):
                    ratones.remove(raton)
                    puntos -= 1
                    reward -= 20

            recompensa_total += reward

            # Ajustar serpiente
            if puntos != puntos_antes:
                num_nuevo = calcular_num_segmentos(puntos)
                num_actual = len(puntos_cadena)

                if num_nuevo > num_actual:
                    for _ in range(num_nuevo - num_actual):
                        ultimo = puntos_cadena[-1]
                        puntos_cadena.append([ultimo[0], ultimo[1]])
                elif num_nuevo < num_actual:
                    puntos_cadena = puntos_cadena[:num_nuevo]

            # Game over
            if puntos <= 0:
                game_over = True
                print(f"Game Over - Partida {partida_num}: {pasos} pasos, {recompensa_total:.1f} reward")

            pasos += 1

        # RENDERIZADO
        pantalla.fill(NEGRO)

        # Agujero
        pygame.draw.circle(pantalla, (30, 30, 30), (int(agujero_x), int(agujero_y)), radio_agujero)
        pygame.draw.circle(pantalla, (60, 60, 60), (int(agujero_x), int(agujero_y)), radio_agujero, 3)

        # Ratones
        for raton in ratones:
            pygame.draw.circle(pantalla, ROJO, (int(raton.x), int(raton.y)), 12)

        # Serpiente (simple)
        for i, punto in enumerate(puntos_cadena):
            color = VERDE_CLARO if i == 0 else VERDE_OSCURO
            radio = RADIO_CABEZA if i == 0 else 18
            pygame.draw.circle(pantalla, color, (int(punto[0]), int(punto[1])), radio)

        # Cabeza destacada
        pygame.draw.circle(pantalla, AMARILLO, (int(cabeza_x), int(cabeza_y)), RADIO_CABEZA, 2)

        # HUD
        texto_titulo = fuente.render("DQN V3 JUGANDO", True, NARANJA)
        pantalla.blit(texto_titulo, (10, 10))

        texto_puntos = fuente.render(f"Puntos: {puntos}", True, VERDE_CLARO if puntos > 1 else ROJO)
        pantalla.blit(texto_puntos, (ANCHO - 220, 20))

        texto_ratones = fuente_pequeña.render(f"Ratones: {len(ratones)}", True, GRIS)
        pantalla.blit(texto_ratones, (ANCHO - 220, 70))

        texto_pasos = fuente_pequeña.render(f"Pasos: {pasos}", True, GRIS)
        pantalla.blit(texto_pasos, (ANCHO - 220, 95))

        texto_reward = fuente_pequeña.render(f"Reward: {recompensa_total:.1f}", True, GRIS)
        pantalla.blit(texto_reward, (ANCHO - 220, 120))

        texto_partida = fuente_pequeña.render(f"Partida: {partida_num}", True, GRIS)
        pantalla.blit(texto_partida, (10, 50))

        if game_over:
            overlay = pygame.Surface((ANCHO, ALTO))
            overlay.set_alpha(180)
            overlay.fill((0, 0, 0))
            pantalla.blit(overlay, (0, 0))

            fuente_go = pygame.font.Font(None, 120)
            texto_go = fuente_go.render("GAME OVER", True, ROJO)
            rect_go = texto_go.get_rect(center=(ANCHO // 2, ALTO // 2 - 50))
            pantalla.blit(texto_go, rect_go)

            texto_r = fuente.render("Presiona R para reiniciar", True, BLANCO)
            rect_r = texto_r.get_rect(center=(ANCHO // 2, ALTO // 2 + 50))
            pantalla.blit(texto_r, rect_r)

        pygame.display.flip()
        reloj.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    jugar_con_dqn()
