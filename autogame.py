"""
Sistema de aprendizaje automático para entrenar un agente que aprenda a jugar
el juego de la serpiente cazando ratones usando Q-Learning.
"""

import math
import random
import pickle
from typing import List, Tuple, Dict
from collections import defaultdict
import numpy as np


# ============================================================================
# CLASE MOUSE - Versión simplificada para el entrenamiento
# ============================================================================


class Mouse:
    """Ratón que intenta escapar del centro hacia los bordes."""

    def __init__(self, ancho: int, alto: int):
        """Inicializa un ratón en el centro de la pantalla."""
        self.ancho = ancho
        self.alto = alto
        self.x = float(ancho // 2)
        self.y = float(alto // 2)
        self.radio = 12
        self.velocidad = 2

        # Elegir un borde aleatorio como destino
        borde = random.randint(0, 3)
        margen = 200

        if borde == 0:  # Arriba
            self.destino_x = random.randint(-margen, ancho + margen)
            self.destino_y = -margen
        elif borde == 1:  # Derecha
            self.destino_x = ancho + margen
            self.destino_y = random.randint(-margen, alto + margen)
        elif borde == 2:  # Abajo
            self.destino_x = random.randint(-margen, ancho + margen)
            self.destino_y = alto + margen
        else:  # Izquierda
            self.destino_x = -margen
            self.destino_y = random.randint(-margen, alto + margen)

    def actualizar(self):
        """Mueve el ratón hacia su destino."""
        dx = self.destino_x - self.x
        dy = self.destino_y - self.y
        distancia = math.sqrt(dx * dx + dy * dy)

        if distancia > 0:
            self.x += (dx / distancia) * self.velocidad
            self.y += (dy / distancia) * self.velocidad

    def fuera_de_pantalla(self) -> bool:
        """Verifica si el ratón está cerca del borde."""
        margen = 100
        return (
            self.x < margen
            or self.x > self.ancho - margen
            or self.y < margen
            or self.y > self.alto - margen
        )


# ============================================================================
# CLASE GAME ENVIRONMENT - Simulador del juego
# ============================================================================


class SnakeGameEnv:
    """Entorno del juego de serpiente para entrenamiento de RL."""

    def __init__(self, ancho: int = 800, alto: int = 600):
        """
        Inicializa el entorno del juego.

        Args:
            ancho: Ancho de la pantalla de juego.
            alto: Alto de la pantalla de juego.
        """
        self.ancho = ancho
        self.alto = alto
        self.reset()

    def reset(self):
        """Reinicia el juego a su estado inicial."""
        self.cabeza_x = float(self.ancho // 2)
        self.cabeza_y = float(self.alto // 2)
        self.puntos = 1
        self.game_over = False
        self.ratones: List[Mouse] = []
        self.pasos = 0
        self.max_pasos = 5000  # Limitar pasos por episodio
        self.tiempo_spawn = 0
        self.intervalo_spawn = 60  # Frames entre spawns
        self.radio_cabeza = 24

        return self.get_state()

    def spawn_raton(self):
        """Genera un nuevo ratón si es tiempo."""
        self.tiempo_spawn += 1
        if self.tiempo_spawn >= self.intervalo_spawn:
            self.ratones.append(Mouse(self.ancho, self.alto))
            self.tiempo_spawn = 0

    def get_state(self) -> Tuple:
        """
        Obtiene el estado actual del juego.

        Estado discretizado para Q-Learning:
        - Posición de la cabeza (cuadrante)
        - Número de ratones
        - Ratón más cercano a la serpiente (dirección y distancia)
        - Ratón más cercano al límite (dirección y distancia al límite)
        - Puntos (discretizado)
        """
        # Discretizar posición de la cabeza (4 cuadrantes)
        cabeza_cuadrante_x = 0 if self.cabeza_x < self.ancho // 2 else 1
        cabeza_cuadrante_y = 0 if self.cabeza_y < self.alto // 2 else 1

        # Variables por defecto cuando no hay ratones
        direccion_cercano_serpiente = 0
        dist_cercano_serpiente = 2
        direccion_cercano_limite = 0
        dist_cercano_limite = 2

        if len(self.ratones) > 0:
            # 1. RATÓN MÁS CERCANO A LA SERPIENTE
            raton_cercano_serpiente = min(
                self.ratones,
                key=lambda r: math.sqrt(
                    (r.x - self.cabeza_x) ** 2 + (r.y - self.cabeza_y) ** 2
                ),
            )

            # Dirección relativa al ratón más cercano a la serpiente (8 direcciones)
            dx_serpiente = raton_cercano_serpiente.x - self.cabeza_x
            dy_serpiente = raton_cercano_serpiente.y - self.cabeza_y
            angulo_serpiente = math.atan2(dy_serpiente, dx_serpiente)
            direccion_cercano_serpiente = int((angulo_serpiente + math.pi) / (math.pi / 4)) % 8

            # Distancia al ratón más cercano a la serpiente (cerca/medio/lejos)
            dist_serpiente = math.sqrt(dx_serpiente * dx_serpiente + dy_serpiente * dy_serpiente)
            if dist_serpiente < 100:
                dist_cercano_serpiente = 0  # Cerca
            elif dist_serpiente < 300:
                dist_cercano_serpiente = 1  # Medio
            else:
                dist_cercano_serpiente = 2  # Lejos

            # 2. RATÓN MÁS CERCANO AL LÍMITE
            def distancia_al_limite(r: Mouse) -> float:
                """Calcula la distancia más corta de un ratón a cualquier borde."""
                dist_arriba = r.y
                dist_abajo = self.alto - r.y
                dist_izquierda = r.x
                dist_derecha = self.ancho - r.x
                return min(dist_arriba, dist_abajo, dist_izquierda, dist_derecha)

            raton_cercano_limite = min(self.ratones, key=distancia_al_limite)

            # Dirección relativa al ratón más cercano al límite (8 direcciones)
            dx_limite = raton_cercano_limite.x - self.cabeza_x
            dy_limite = raton_cercano_limite.y - self.cabeza_y
            angulo_limite = math.atan2(dy_limite, dx_limite)
            direccion_cercano_limite = int((angulo_limite + math.pi) / (math.pi / 4)) % 8

            # Distancia de ese ratón al límite (cerca/medio/lejos)
            dist_limite = distancia_al_limite(raton_cercano_limite)
            if dist_limite < 100:
                dist_cercano_limite = 0  # Muy cerca del límite (urgente)
            elif dist_limite < 250:
                dist_cercano_limite = 1  # Medio
            else:
                dist_cercano_limite = 2  # Lejos del límite

        # Discretizar puntos
        puntos_categoria = min(self.puntos // 3, 5)  # 0-5

        # Número de ratones (limitado)
        num_ratones = min(len(self.ratones), 5)

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

    def step(self, accion: int) -> Tuple[Tuple, float, bool]:
        """
        Ejecuta una acción en el entorno.

        Args:
            accion: 0=arriba, 1=derecha, 2=abajo, 3=izquierda

        Returns:
            (nuevo_estado, recompensa, terminado)
        """
        if self.game_over:
            return self.get_state(), 0, True

        # Mover la cabeza según la acción
        velocidad = 5
        if accion == 0:  # Arriba
            self.cabeza_y -= velocidad
        elif accion == 1:  # Derecha
            self.cabeza_x += velocidad
        elif accion == 2:  # Abajo
            self.cabeza_y += velocidad
        elif accion == 3:  # Izquierda
            self.cabeza_x -= velocidad

        # Mantener la cabeza dentro de límites
        self.cabeza_x = max(0, min(self.ancho, self.cabeza_x))
        self.cabeza_y = max(0, min(self.alto, self.cabeza_y))

        # Spawn de ratones
        self.spawn_raton()

        # Actualizar ratones
        for raton in self.ratones:
            raton.actualizar()

        # Verificar colisiones y ratones fuera
        recompensa = 0
        ratones_a_eliminar = []

        for i, raton in enumerate(self.ratones):
            # Colisión con cabeza
            dx = raton.x - self.cabeza_x
            dy = raton.y - self.cabeza_y
            distancia = math.sqrt(dx * dx + dy * dy)

            if distancia < (self.radio_cabeza + raton.radio):
                ratones_a_eliminar.append(i)
                self.puntos += 1
                recompensa += 10  # Recompensa por capturar

            # Ratón fuera de pantalla
            elif raton.fuera_de_pantalla():
                ratones_a_eliminar.append(i)
                self.puntos -= 1
                recompensa -= 5  # Penalización por perder ratón

        # Eliminar ratones
        for i in sorted(set(ratones_a_eliminar), reverse=True):
            del self.ratones[i]

        # Pequeña recompensa por sobrevivir
        recompensa += 0.01

        # Verificar Game Over
        if self.puntos <= 0:
            self.game_over = True
            recompensa -= 100  # Gran penalización por perder

        # Incrementar pasos
        self.pasos += 1
        if self.pasos >= self.max_pasos:
            self.game_over = True

        return self.get_state(), recompensa, self.game_over


# ============================================================================
# AGENTE Q-LEARNING
# ============================================================================


class QLearningAgent:
    """Agente que aprende a jugar usando Q-Learning."""

    def __init__(
        self,
        num_acciones: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        """
        Inicializa el agente de Q-Learning.

        Args:
            num_acciones: Número de acciones posibles.
            learning_rate: Tasa de aprendizaje (alpha).
            discount_factor: Factor de descuento (gamma).
            epsilon: Probabilidad inicial de exploración.
            epsilon_decay: Factor de decaimiento de epsilon.
            epsilon_min: Epsilon mínimo.
        """
        self.num_acciones = num_acciones
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Tabla Q: diccionario de (estado, accion) -> valor
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(num_acciones)
        )

    def get_action(self, estado: Tuple) -> int:
        """
        Selecciona una acción usando epsilon-greedy.

        Args:
            estado: Estado actual del juego.

        Returns:
            Acción a tomar (0-3).
        """
        # Exploración vs Explotación
        if random.random() < self.epsilon:
            return random.randint(0, self.num_acciones - 1)
        else:
            return int(np.argmax(self.q_table[estado]))

    def update(
        self, estado: Tuple, accion: int, recompensa: float, siguiente_estado: Tuple
    ):
        """
        Actualiza la tabla Q usando la ecuación de Q-Learning.

        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        """
        q_actual = self.q_table[estado][accion]
        q_siguiente_max = np.max(self.q_table[siguiente_estado])

        # Actualización Q-Learning
        nuevo_q = q_actual + self.lr * (
            recompensa + self.gamma * q_siguiente_max - q_actual
        )
        self.q_table[estado][accion] = nuevo_q

    def decay_epsilon(self):
        """Reduce epsilon para disminuir exploración con el tiempo."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename: str):
        """Guarda la tabla Q en un archivo."""
        with open(filename, "wb") as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Modelo guardado en {filename}")

    def load(self, filename: str):
        """Carga la tabla Q desde un archivo."""
        try:
            with open(filename, "rb") as f:
                self.q_table = defaultdict(lambda: np.zeros(self.num_acciones))
                self.q_table.update(pickle.load(f))
            print(f"Modelo cargado desde {filename}")
        except FileNotFoundError:
            print(f"No se encontró el archivo {filename}, empezando desde cero")


# ============================================================================
# ENTRENAMIENTO
# ============================================================================


def entrenar(
    num_episodios: int = 1000,
    guardar_cada: int = 100,
    mostrar_cada: int = 10,
    archivo_modelo: str = "snake_qlearning.pkl",
):
    """
    Entrena el agente de Q-Learning.

    Args:
        num_episodios: Número de partidas para entrenar.
        guardar_cada: Guardar modelo cada N episodios.
        mostrar_cada: Mostrar progreso cada N episodios.
        archivo_modelo: Nombre del archivo para guardar/cargar el modelo.
    """
    env = SnakeGameEnv()
    agente = QLearningAgent()

    # Intentar cargar modelo previo
    agente.load(archivo_modelo)

    estadisticas = {
        "recompensas": [],
        "puntos_finales": [],
        "pasos": [],
    }

    print("Iniciando entrenamiento...")
    print(f"Episodios: {num_episodios}")
    print("-" * 60)

    for episodio in range(num_episodios):
        estado = env.reset()
        recompensa_total = 0
        terminado = False

        while not terminado:
            # Seleccionar y ejecutar acción
            accion = agente.get_action(estado)
            siguiente_estado, recompensa, terminado = env.step(accion)

            # Actualizar Q-table
            agente.update(estado, accion, recompensa, siguiente_estado)

            estado = siguiente_estado
            recompensa_total += recompensa

        # Decay epsilon
        agente.decay_epsilon()

        # Guardar estadísticas
        estadisticas["recompensas"].append(recompensa_total)
        estadisticas["puntos_finales"].append(env.puntos)
        estadisticas["pasos"].append(env.pasos)

        # Mostrar progreso
        if (episodio + 1) % mostrar_cada == 0:
            promedio_recompensa = np.mean(estadisticas["recompensas"][-mostrar_cada:])
            promedio_puntos = np.mean(estadisticas["puntos_finales"][-mostrar_cada:])
            promedio_pasos = np.mean(estadisticas["pasos"][-mostrar_cada:])

            print(
                f"Episodio {episodio + 1}/{num_episodios} | "
                f"Epsilon: {agente.epsilon:.3f} | "
                f"Recompensa Promedio: {promedio_recompensa:.2f} | "
                f"Puntos Promedio: {promedio_puntos:.1f} | "
                f"Pasos Promedio: {promedio_pasos:.0f}"
            )

        # Guardar modelo
        if (episodio + 1) % guardar_cada == 0:
            agente.save(archivo_modelo)

    # Guardar modelo final
    agente.save(archivo_modelo)

    print("\n" + "=" * 60)
    print("Entrenamiento completado!")
    print(
        f"Recompensa promedio final (últimos 100): {np.mean(estadisticas['recompensas'][-100:]):.2f}"
    )
    print(
        f"Puntos promedio final (últimos 100): {np.mean(estadisticas['puntos_finales'][-100:]):.1f}"
    )
    print(
        f"Estados en Q-table: {len(agente.q_table)}"
    )
    print("=" * 60)

    return agente, estadisticas


def probar_agente(archivo_modelo: str = "snake_qlearning.pkl", num_partidas: int = 10):
    """
    Prueba el agente entrenado.

    Args:
        archivo_modelo: Archivo del modelo a probar.
        num_partidas: Número de partidas de prueba.
    """
    env = SnakeGameEnv()
    agente = QLearningAgent()
    agente.load(archivo_modelo)
    agente.epsilon = 0  # Sin exploración, solo explotación

    print(f"\nProbando agente entrenado ({num_partidas} partidas)...")
    print("-" * 60)

    puntos_totales = []

    for partida in range(num_partidas):
        estado = env.reset()
        terminado = False
        pasos = 0

        while not terminado:
            accion = agente.get_action(estado)
            estado, _, terminado = env.step(accion)
            pasos += 1

        puntos_totales.append(env.puntos)
        print(f"Partida {partida + 1}: {env.puntos} puntos, {pasos} pasos")

    print("-" * 60)
    print(f"Puntos promedio: {np.mean(puntos_totales):.2f}")
    print(f"Puntos máximos: {max(puntos_totales)}")
    print(f"Puntos mínimos: {min(puntos_totales)}")


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Modo prueba
        probar_agente()
    else:
        # Modo entrenamiento
        agente, stats = entrenar(num_episodios=2000, mostrar_cada=50)

        # Probar agente entrenado
        print("\n")
        probar_agente()
