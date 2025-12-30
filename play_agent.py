"""
Visualizador del agente entrenado jugando al juego de la serpiente.
Este script carga el modelo entrenado y muestra al agente jugando en tiempo real.
"""

import pickle
import math
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict

# Importar el juego principal
import pygame
import random
from typing import List


# ============================================================================
# COPIAR CLASE MOUSE DEL JUEGO PRINCIPAL
# ============================================================================


class Mouse:
    """Representa un ratón que sale del agujero y se mueve hacia un destino aleatorio."""

    def __init__(self, x: float, y: float, ancho_pantalla: int, alto_pantalla: int):
        self.x = x
        self.y = y
        self.radio = 12
        self.velocidad = 2

        # Calcular destino aleatorio FUERA de la pantalla
        borde = random.randint(0, 3)
        margen_fuera = 200

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

        self.color = (150, 150, 150)
        self.color_orejas = (200, 150, 150)

    def actualizar(self) -> None:
        dx = self.destino_x - self.x
        dy = self.destino_y - self.y
        distancia = math.sqrt(dx * dx + dy * dy)

        if distancia > 0:
            self.x += (dx / distancia) * self.velocidad
            self.y += (dy / distancia) * self.velocidad

    def dibujar(self, pantalla: pygame.Surface) -> None:
        dx = self.destino_x - self.x
        dy = self.destino_y - self.y
        angulo = math.atan2(dy, dx)

        pygame.draw.circle(pantalla, self.color, (int(self.x), int(self.y)), self.radio)

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

        nariz = (
            int(self.x + (self.radio - 2) * math.cos(angulo)),
            int(self.y + (self.radio - 2) * math.sin(angulo)),
        )
        pygame.draw.circle(pantalla, (50, 50, 50), nariz, 2)

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
        margen = 100
        return (
            self.x < margen
            or self.x > ancho - margen
            or self.y < margen
            or self.y > alto - margen
        )


# ============================================================================
# AGENTE DE IA
# ============================================================================


class AIAgent:
    """Agente de IA que usa el modelo entrenado para jugar y puede seguir aprendiendo."""

    def __init__(
        self,
        archivo_modelo: str = "snake_qlearning_best_v2.pkl",
        entrenar: bool = False,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1
    ):
        self.num_acciones = 4
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.num_acciones)
        )
        self.archivo_modelo = archivo_modelo
        self.entrenar = entrenar

        # Parámetros de aprendizaje
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon if entrenar else 0  # Sin exploración si no entrena

        # Contadores
        self.episodios_completados = 0
        self.guardar_cada = 50  # Guardar cada 50 partidas

        self.cargar_modelo(archivo_modelo)

    def cargar_modelo(self, archivo: str):
        """Carga el modelo entrenado."""
        try:
            with open(archivo, "rb") as f:
                self.q_table.update(pickle.load(f))
            print(f"✓ Modelo cargado desde {archivo}")
            print(f"  Estados aprendidos: {len(self.q_table)}")
        except FileNotFoundError:
            print(f"✗ No se encontró {archivo}")
            print("  Ejecuta 'python autogame.py' primero para entrenar el agente")

    def get_state(
        self, cabeza_x: float, cabeza_y: float, ratones: List[Mouse], puntos: int, ancho: int, alto: int
    ) -> Tuple:
        """Obtiene el estado actual del juego (mismo formato que en el entrenamiento)."""
        # Discretizar posición de la cabeza
        cabeza_cuadrante_x = 0 if cabeza_x < ancho // 2 else 1
        cabeza_cuadrante_y = 0 if cabeza_y < alto // 2 else 1

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
                dist_abajo = alto - r.y
                dist_izquierda = r.x
                dist_derecha = ancho - r.x
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

    def get_action(self, estado: Tuple) -> int:
        """Selecciona la mejor acción (con exploración si está entrenando)."""
        if self.entrenar and random.random() < self.epsilon:
            return random.randint(0, self.num_acciones - 1)
        return int(np.argmax(self.q_table[estado]))

    def update(self, estado: Tuple, accion: int, recompensa: float, siguiente_estado: Tuple):
        """Actualiza la tabla Q (solo si está en modo entrenamiento)."""
        if not self.entrenar:
            return

        q_actual = self.q_table[estado][accion]
        q_siguiente_max = np.max(self.q_table[siguiente_estado])

        # Actualización Q-Learning
        nuevo_q = q_actual + self.lr * (recompensa + self.gamma * q_siguiente_max - q_actual)
        self.q_table[estado][accion] = nuevo_q

    def fin_episodio(self):
        """Marca el fin de un episodio y guarda si es necesario."""
        if not self.entrenar:
            return

        self.episodios_completados += 1

        # Guardar periódicamente
        if self.episodios_completados % self.guardar_cada == 0:
            self.guardar_modelo()
            print(f"✓ Modelo guardado (Episodio {self.episodios_completados})")

    def guardar_modelo(self):
        """Guarda el modelo actual."""
        with open(self.archivo_modelo, "wb") as f:
            pickle.dump(dict(self.q_table), f)


# ============================================================================
# JUEGO CON IA
# ============================================================================


def jugar_con_ia(entrenar_modo: bool = False):
    """
    Ejecuta el juego visual con el agente de IA jugando.

    Args:
        entrenar_modo: Si True, el agente entrena mientras juega.
    """
    pygame.init()

    # Configuración de la pantalla
    ANCHO, ALTO = 2560, 1440
    pantalla = pygame.display.set_mode((ANCHO, ALTO))
    titulo = "Snake AI - ENTRENANDO" if entrenar_modo else "Snake AI - Agente Jugando"
    pygame.display.set_caption(titulo)
    clock = pygame.time.Clock()

    # Colores
    BLANCO = (255, 255, 255)
    NEGRO = (0, 0, 0)
    VERDE_OSCURO = (34, 120, 50)
    VERDE_CLARO = (50, 180, 70)
    AMARILLO = (220, 200, 50)
    ROJO = (200, 50, 50)
    GRIS = (100, 100, 100)
    AZUL = (50, 150, 255)
    NARANJA = (255, 140, 0)

    # Configuración de física
    SEGMENTOS_POR_PUNTO = 5
    SEGMENTOS_MINIMOS = 10
    DISTANCIA_SEGMENTO = 30
    ANGULO_MAX = math.radians(25)
    RADIO_CABEZA = 24
    VELOCIDAD = 5
    FPS = 60

    # Agente de IA (con o sin entrenamiento)
    agente = AIAgent(entrenar=entrenar_modo)

    # Variables del juego
    puntos = 1
    game_over = False
    radio_agujero = 40
    velocidad_agujero = 1.5

    # Posición del agujero (inicialmente aleatoria)
    agujero_x = float(random.randint(300, ANCHO - 300))
    agujero_y = float(random.randint(300, ALTO - 300))
    agujero_destino_x = float(random.randint(300, ANCHO - 300))
    agujero_destino_y = float(random.randint(300, ALTO - 300))

    # Inicializar serpiente
    ancla_pos = [float(ANCHO // 2), float(ALTO // 2)]

    def calcular_num_segmentos(pts: int) -> int:
        return max(SEGMENTOS_MINIMOS, pts * SEGMENTOS_POR_PUNTO)

    def inicializar_cadena(num_segs: int, pos_inicial: List[float] = None):
        """Retorna (cadena, posicion_inicial)"""
        cadena = []
        if pos_inicial is None:
            # Posición aleatoria alejada del centro
            pos_inicio = [
                float(random.randint(300, ANCHO - 300)),
                float(random.randint(300, ALTO - 300))
            ]
        else:
            pos_inicio = pos_inicial.copy()

        pos = pos_inicio.copy()
        for _ in range(num_segs):
            pos = [pos[0] + DISTANCIA_SEGMENTO, pos[1]]
            cadena.append(list(pos))
        return cadena, pos_inicio

    num_segmentos = calcular_num_segmentos(1)
    puntos_cadena, posicion_inicial = inicializar_cadena(num_segmentos)
    ancla_pos = list(posicion_inicial)

    # Sistema de ratones
    ratones: List[Mouse] = []
    tiempo_ultimo_spawn = 0
    intervalo_spawn = 2000

    # Funciones de física (copiadas de main.py)
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

    # Loop principal
    running = True
    partida_num = 1

    print("\n" + "=" * 60)
    print("AGENTE DE IA JUGANDO")
    print("=" * 60)
    print("Presiona ESC para salir")
    print("Presiona R para reiniciar")
    print("-" * 60)

    while running:
        # Eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    # Reiniciar
                    puntos = 1
                    game_over = False
                    ratones.clear()
                    tiempo_ultimo_spawn = 0
                    # Nueva posición aleatoria para serpiente
                    puntos_cadena, posicion_inicial = inicializar_cadena(calcular_num_segmentos(1))
                    ancla_pos = list(posicion_inicial)
                    # Nueva posición aleatoria para agujero
                    agujero_x = float(random.randint(300, ANCHO - 300))
                    agujero_y = float(random.randint(300, ALTO - 300))
                    agujero_destino_x = float(random.randint(300, ANCHO - 300))
                    agujero_destino_y = float(random.randint(300, ALTO - 300))
                    partida_num += 1
                    print(f"\n--- Partida {partida_num} iniciada ---")

        if not game_over:
            # Mover el agujero hacia su destino
            dx_agujero = agujero_destino_x - agujero_x
            dy_agujero = agujero_destino_y - agujero_y
            dist_agujero = math.sqrt(dx_agujero * dx_agujero + dy_agujero * dy_agujero)

            if dist_agujero < 5:  # Si llegó al destino, elegir uno nuevo
                agujero_destino_x = float(random.randint(300, ANCHO - 300))
                agujero_destino_y = float(random.randint(300, ALTO - 300))
            else:
                # Mover hacia el destino
                agujero_x += (dx_agujero / dist_agujero) * velocidad_agujero
                agujero_y += (dy_agujero / dist_agujero) * velocidad_agujero

            # IA toma decisión
            estado = agente.get_state(ancla_pos[0], ancla_pos[1], ratones, puntos, ANCHO, ALTO)
            accion = agente.get_action(estado)

            # Ejecutar acción (0=arriba, 1=derecha, 2=abajo, 3=izquierda)
            if accion == 0:
                ancla_pos[1] -= VELOCIDAD
            elif accion == 1:
                ancla_pos[0] += VELOCIDAD
            elif accion == 2:
                ancla_pos[1] += VELOCIDAD
            elif accion == 3:
                ancla_pos[0] -= VELOCIDAD

            # Límites de pantalla
            ancla_pos[0] = max(0.0, min(float(ANCHO), ancla_pos[0]))
            ancla_pos[1] = max(0.0, min(float(ALTO), ancla_pos[1]))

            # Spawn de ratones
            tiempo_actual = pygame.time.get_ticks()
            if tiempo_actual - tiempo_ultimo_spawn > intervalo_spawn:
                ratones.append(Mouse(float(agujero_x), float(agujero_y), ANCHO, ALTO))
                tiempo_ultimo_spawn = tiempo_actual

            # Actualizar ratones
            for raton in ratones:
                raton.actualizar()

            # Colisiones
            ratones_a_eliminar = []
            puntos_antes = puntos

            for i, raton in enumerate(ratones):
                dx = raton.x - ancla_pos[0]
                dy = raton.y - ancla_pos[1]
                distancia = math.sqrt(dx * dx + dy * dy)

                if distancia < (RADIO_CABEZA + raton.radio):
                    ratones_a_eliminar.append(i)
                    puntos += 1

            for i, raton in enumerate(ratones):
                if i not in ratones_a_eliminar and raton.fuera_de_pantalla(ANCHO, ALTO):
                    ratones_a_eliminar.append(i)
                    puntos -= 1

            for i in sorted(set(ratones_a_eliminar), reverse=True):
                del ratones[i]

            # Ajustar longitud serpiente
            if puntos != puntos_antes:
                num_nuevo = calcular_num_segmentos(puntos)
                num_actual = len(puntos_cadena)

                if num_nuevo > num_actual:
                    for _ in range(num_nuevo - num_actual):
                        if len(puntos_cadena) >= 2:
                            dx = puntos_cadena[-1][0] - puntos_cadena[-2][0]
                            dy = puntos_cadena[-1][1] - puntos_cadena[-2][1]
                            dist = math.sqrt(dx * dx + dy * dy)
                            if dist > 0:
                                dx_norm, dy_norm = dx / dist, dy / dist
                            else:
                                dx_norm, dy_norm = 1, 0
                        else:
                            dx_norm, dy_norm = 1, 0
                        ultimo = puntos_cadena[-1]
                        puntos_cadena.append(
                            [
                                ultimo[0] + dx_norm * DISTANCIA_SEGMENTO,
                                ultimo[1] + dy_norm * DISTANCIA_SEGMENTO,
                            ]
                        )
                elif num_nuevo < num_actual:
                    for _ in range(num_actual - num_nuevo):
                        if len(puntos_cadena) > SEGMENTOS_MINIMOS:
                            puntos_cadena.pop()

            # Game Over - Auto reinicio
            if puntos <= 0:
                print(f"Game Over en partida {partida_num}")

                # Notificar fin de episodio al agente
                agente.fin_episodio()

                # Auto reinicio después de breve pausa
                pygame.time.wait(500)  # Esperar 0.5 segundos

                # Reiniciar automáticamente
                puntos = 1
                game_over = False
                ratones.clear()
                tiempo_ultimo_spawn = 0
                # Nueva posición aleatoria para serpiente
                puntos_cadena, posicion_inicial = inicializar_cadena(calcular_num_segmentos(1))
                ancla_pos = list(posicion_inicial)
                # Nueva posición aleatoria para agujero
                agujero_x = float(random.randint(300, ANCHO - 300))
                agujero_y = float(random.randint(300, ALTO - 300))
                agujero_destino_x = float(random.randint(300, ANCHO - 300))
                agujero_destino_y = float(random.randint(300, ALTO - 300))
                partida_num += 1
                print(f"--- Partida {partida_num} iniciada ---")

        # Renderizado
        pantalla.fill(NEGRO)

        # Actualizar física serpiente
        puntos_cadena[0], angulo_prev = restringir_distancia(
            ancla_pos, puntos_cadena[0], DISTANCIA_SEGMENTO
        )
        for i in range(1, len(puntos_cadena)):
            puntos_cadena[i], angulo_prev = aplicar_restricciones(
                puntos_cadena[i - 1], puntos_cadena[i], DISTANCIA_SEGMENTO, angulo_prev
            )

        # Dibujar cuerpo serpiente
        puntos_izq = []
        puntos_der = []
        puntos_centro = [ancla_pos]

        dx = puntos_cadena[0][0] - ancla_pos[0]
        dy = puntos_cadena[0][1] - ancla_pos[1]
        angulo_cabeza = math.atan2(dy, dx)

        puntos_izq.append(
            (
                int(ancla_pos[0] + 22 * math.cos(angulo_cabeza + math.pi / 2)),
                int(ancla_pos[1] + 22 * math.sin(angulo_cabeza + math.pi / 2)),
            )
        )
        puntos_der.append(
            (
                int(ancla_pos[0] + 22 * math.cos(angulo_cabeza - math.pi / 2)),
                int(ancla_pos[1] + 22 * math.sin(angulo_cabeza - math.pi / 2)),
            )
        )

        referencia = ancla_pos
        angulo = angulo_cabeza

        for i, punto in enumerate(puntos_cadena):
            puntos_centro.append(punto)
            dx = punto[0] - referencia[0]
            dy = punto[1] - referencia[1]
            angulo = math.atan2(dy, dx)
            grosor = obtener_grosor(i, len(puntos_cadena))

            x_der = punto[0] + grosor * math.cos(angulo + math.pi / 2)
            y_der = punto[1] + grosor * math.sin(angulo + math.pi / 2)
            puntos_der.append((int(x_der), int(y_der)))

            x_izq = punto[0] + grosor * math.cos(angulo - math.pi / 2)
            y_izq = punto[1] + grosor * math.sin(angulo - math.pi / 2)
            puntos_izq.append((int(x_izq), int(y_izq)))

            referencia = punto

        ultimo = puntos_cadena[-1]
        puntos_cola = (
            int(ultimo[0] + 15 * math.cos(angulo)),
            int(ultimo[1] + 15 * math.sin(angulo)),
        )

        manto = puntos_izq + [puntos_cola] + puntos_der[::-1]

        if len(manto) > 2:
            pygame.draw.polygon(pantalla, VERDE_OSCURO, manto)
            pygame.draw.polygon(pantalla, VERDE_CLARO, manto, 3)

        for i in range(len(puntos_izq) - 1):
            if i % 2 == 0:
                pygame.draw.line(pantalla, VERDE_CLARO, puntos_izq[i], puntos_der[i], 1)

        for i in range(len(puntos_centro) - 1):
            pygame.draw.line(
                pantalla,
                AMARILLO,
                (int(puntos_centro[i][0]), int(puntos_centro[i][1])),
                (int(puntos_centro[i + 1][0]), int(puntos_centro[i + 1][1])),
                4,
            )

        # Cabeza
        pygame.draw.circle(
            pantalla, VERDE_OSCURO, (int(ancla_pos[0]), int(ancla_pos[1])), RADIO_CABEZA
        )
        pygame.draw.circle(
            pantalla, VERDE_CLARO, (int(ancla_pos[0]), int(ancla_pos[1])), RADIO_CABEZA, 2
        )

        # Ojos
        ojo_izq = (
            int(ancla_pos[0] + 12 * math.cos(angulo_cabeza + math.pi + 0.6)),
            int(ancla_pos[1] + 12 * math.sin(angulo_cabeza + math.pi + 0.6)),
        )
        ojo_der = (
            int(ancla_pos[0] + 12 * math.cos(angulo_cabeza + math.pi - 0.6)),
            int(ancla_pos[1] + 12 * math.sin(angulo_cabeza + math.pi - 0.6)),
        )
        pygame.draw.circle(pantalla, AMARILLO, ojo_izq, 6)
        pygame.draw.circle(pantalla, AMARILLO, ojo_der, 6)
        pygame.draw.circle(pantalla, NEGRO, ojo_izq, 3)
        pygame.draw.circle(pantalla, NEGRO, ojo_der, 3)

        # Lengua
        lengua_base = (
            int(ancla_pos[0] + 20 * math.cos(angulo_cabeza + math.pi)),
            int(ancla_pos[1] + 20 * math.sin(angulo_cabeza + math.pi)),
        )
        lengua_punta1 = (
            int(lengua_base[0] + 15 * math.cos(angulo_cabeza + math.pi + 0.3)),
            int(lengua_base[1] + 15 * math.sin(angulo_cabeza + math.pi + 0.3)),
        )
        lengua_punta2 = (
            int(lengua_base[0] + 15 * math.cos(angulo_cabeza + math.pi - 0.3)),
            int(lengua_base[1] + 15 * math.sin(angulo_cabeza + math.pi - 0.3)),
        )
        pygame.draw.line(pantalla, ROJO, (int(ancla_pos[0]), int(ancla_pos[1])), lengua_base, 3)
        pygame.draw.line(pantalla, ROJO, lengua_base, lengua_punta1, 2)
        pygame.draw.line(pantalla, ROJO, lengua_base, lengua_punta2, 2)

        # Agujero y ratones
        pygame.draw.circle(pantalla, (30, 30, 30), (agujero_x, agujero_y), radio_agujero)
        pygame.draw.circle(pantalla, (60, 60, 60), (agujero_x, agujero_y), radio_agujero, 3)

        for raton in ratones:
            raton.dibujar(pantalla)

        # HUD
        fuente_titulo = pygame.font.Font(None, 36)
        if entrenar_modo:
            titulo = fuente_titulo.render("IA ENTRENANDO", True, NARANJA)
        else:
            titulo = fuente_titulo.render("IA JUGANDO", True, AZUL)
        pantalla.blit(titulo, (10, 10))

        fuente_puntos = pygame.font.Font(None, 48)
        color_puntos = VERDE_CLARO if puntos > 1 else ROJO
        texto_puntos = fuente_puntos.render(f"Puntos: {puntos}", True, color_puntos)
        pantalla.blit(texto_puntos, (ANCHO - 220, 20))

        fuente_info = pygame.font.Font(None, 24)
        texto_ratones = fuente_info.render(f"Ratones: {len(ratones)}", True, GRIS)
        pantalla.blit(texto_ratones, (ANCHO - 220, 70))

        texto_longitud = fuente_info.render(
            f"Longitud: {len(puntos_cadena)} segs", True, GRIS
        )
        pantalla.blit(texto_longitud, (ANCHO - 220, 95))

        texto_partida = fuente_info.render(f"Partida: {partida_num}", True, GRIS)
        pantalla.blit(texto_partida, (10, 50))

        # Si está entrenando, mostrar episodios completados
        if entrenar_modo:
            texto_episodios = fuente_info.render(
                f"Episodios: {agente.episodios_completados}", True, NARANJA
            )
            pantalla.blit(texto_episodios, (10, 75))

        # Game Over
        if game_over:
            overlay = pygame.Surface((ANCHO, ALTO))
            overlay.set_alpha(180)
            overlay.fill((0, 0, 0))
            pantalla.blit(overlay, (0, 0))

            fuente_go = pygame.font.Font(None, 120)
            texto_go = fuente_go.render("GAME OVER", True, ROJO)
            rect_go = texto_go.get_rect(center=(ANCHO // 2, ALTO // 2 - 50))
            pantalla.blit(texto_go, rect_go)

            fuente_r = pygame.font.Font(None, 36)
            texto_r = fuente_r.render("Presiona R para que la IA juegue de nuevo", True, BLANCO)
            rect_r = texto_r.get_rect(center=(ANCHO // 2, ALTO // 2 + 50))
            pantalla.blit(texto_r, rect_r)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    print("\n" + "=" * 60)
    print("Sesión finalizada")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    # Verificar argumentos de línea de comandos
    entrenar = False
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        entrenar = True
        print("\n" + "=" * 60)
        print("MODO ENTRENAMIENTO VISUAL ACTIVADO")
        print("=" * 60)
        print("El agente jugará y aprenderá en tiempo real.")
        print("El modelo se guardará cada 50 partidas automáticamente.")
        print("Presiona ESC para salir.")
        print("-" * 60 + "\n")
    else:
        print("\n" + "=" * 60)
        print("MODO VISUALIZACIÓN")
        print("=" * 60)
        print("El agente jugará usando el modelo entrenado.")
        print("Para entrenar mientras juega, ejecuta: python play_agent.py train")
        print("Presiona ESC para salir.")
        print("-" * 60 + "\n")

    jugar_con_ia(entrenar_modo=entrenar)
