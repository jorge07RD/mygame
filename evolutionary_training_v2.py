"""
Sistema de entrenamiento evolutivo MEJORADO con técnicas para romper plateaus.

MEJORAS CLAVE:
1. Estado más rico (12 dimensiones vs 8)
2. Mutación adaptativa (aumenta cuando hay estancamiento)
3. Diversidad forzada en la población
4. Recompensas moldeadas (reward shaping)
5. Mejor evaluación con más episodios
"""

import os
import sys
import time
import random
import pickle
import numpy as np
from typing import List, Tuple, Dict
from multiprocessing import Pool, cpu_count
from collections import deque
import math


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

POBLACION_SIZE = 25  # Población grande para máxima diversidad
EPISODIOS_ENTRENAMIENTO = 300  # Más entrenamiento por generación
EPISODIOS_EVALUACION = 100  # Más evaluaciones para reducir varianza
DIRECTORIO_POBLACION = "poblacion_v2"


# ============================================================================
# ENTORNO MEJORADO CON ESTADO MÁS RICO
# ============================================================================

class Mouse:
    """Ratón que intenta escapar."""

    def __init__(self, x: float, y: float, ancho: int, alto: int):
        self.x = x
        self.y = y
        self.radio = 12
        self.velocidad = 2

        # Destino hacia un borde aleatorio
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
        dx = self.destino_x - self.x
        dy = self.destino_y - self.y
        distancia = math.sqrt(dx * dx + dy * dy)

        if distancia > 0:
            self.x += (dx / distancia) * self.velocidad
            self.y += (dy / distancia) * self.velocidad

    def fuera_de_pantalla(self, ancho: int, alto: int) -> bool:
        margen = 100
        return (
            self.x < margen or self.x > ancho - margen or
            self.y < margen or self.y > alto - margen
        )


class SnakeGameEnvMejorado:
    """Entorno mejorado con estado más rico y recompensas moldeadas."""

    def __init__(self, ancho: int = 800, alto: int = 600):
        self.ancho = ancho
        self.alto = alto
        self.reset()

    def reset(self):
        self.cabeza_x = float(self.ancho // 2)
        self.cabeza_y = float(self.alto // 2)
        self.puntos = 1
        self.game_over = False
        self.ratones: List[Mouse] = []
        self.pasos = 0
        self.max_pasos = 8000
        self.tiempo_spawn = 0
        self.intervalo_spawn = 60
        self.radio_cabeza = 24

        # Historial para recompensas moldeadas
        self.ultimo_puntos = 1
        self.capturas_recientes = 0
        self.perdidas_recientes = 0

        return self.get_state()

    def get_state(self) -> Tuple:
        """
        ESTADO MEJORADO con 12 dimensiones (vs 8 original):

        1-2. Posición de cabeza (4 cuadrantes)
        3-4. Ratón más cercano: dirección (8 dirs) + distancia (4 categorías)
        5-6. Ratón más urgente: dirección (8 dirs) + urgencia (4 niveles)
        7. Puntos (6 categorías)
        8. Número de ratones (6 categorías)
        9. Zona de la pantalla donde está la cabeza (9 zonas)
        10. Tendencia de puntos (subiendo/bajando/estable)
        11. Densidad de ratones (baja/media/alta)
        12. Amenaza inmediata (hay ratón a punto de escapar)
        """
        # 1-2. Posición de cabeza (cuadrantes)
        cabeza_cuad_x = 0 if self.cabeza_x < self.ancho // 2 else 1
        cabeza_cuad_y = 0 if self.cabeza_y < self.alto // 2 else 1

        # Valores por defecto
        dir_cercano = 0
        dist_cercano = 3
        dir_urgente = 0
        urgencia = 3
        zona = 4  # Centro
        amenaza = 0
        densidad = 0

        if len(self.ratones) > 0:
            # 3-4. Ratón más cercano a serpiente
            raton_cercano = min(
                self.ratones,
                key=lambda r: math.sqrt((r.x - self.cabeza_x)**2 + (r.y - self.cabeza_y)**2)
            )

            dx = raton_cercano.x - self.cabeza_x
            dy = raton_cercano.y - self.cabeza_y
            angulo = math.atan2(dy, dx)
            dir_cercano = int((angulo + math.pi) / (math.pi / 4)) % 8

            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 80:
                dist_cercano = 0  # Muy cerca
            elif dist < 200:
                dist_cercano = 1  # Cerca
            elif dist < 400:
                dist_cercano = 2  # Medio
            else:
                dist_cercano = 3  # Lejos

            # 5-6. Ratón más urgente (más cerca del borde)
            def distancia_al_borde(r: Mouse) -> float:
                return min(r.y, self.alto - r.y, r.x, self.ancho - r.x)

            raton_urgente = min(self.ratones, key=distancia_al_borde)

            dx_urg = raton_urgente.x - self.cabeza_x
            dy_urg = raton_urgente.y - self.cabeza_y
            angulo_urg = math.atan2(dy_urg, dx_urg)
            dir_urgente = int((angulo_urg + math.pi) / (math.pi / 4)) % 8

            dist_borde = distancia_al_borde(raton_urgente)
            if dist_borde < 80:
                urgencia = 0  # Crítico
                amenaza = 1
            elif dist_borde < 150:
                urgencia = 1  # Urgente
            elif dist_borde < 250:
                urgencia = 2  # Moderado
            else:
                urgencia = 3  # Bajo

            # 11. Densidad de ratones
            if len(self.ratones) <= 1:
                densidad = 0  # Baja
            elif len(self.ratones) <= 3:
                densidad = 1  # Media
            else:
                densidad = 2  # Alta

        # 7. Puntos (categorías)
        puntos_cat = min(self.puntos // 5, 5)

        # 8. Número de ratones
        num_ratones = min(len(self.ratones), 5)

        # 9. Zona de pantalla (3x3 grid)
        zona_x = 0 if self.cabeza_x < self.ancho / 3 else (1 if self.cabeza_x < 2 * self.ancho / 3 else 2)
        zona_y = 0 if self.cabeza_y < self.alto / 3 else (1 if self.cabeza_y < 2 * self.alto / 3 else 2)
        zona = zona_y * 3 + zona_x

        # 10. Tendencia de puntos
        if self.puntos > self.ultimo_puntos:
            tendencia = 1  # Subiendo
        elif self.puntos < self.ultimo_puntos:
            tendencia = 0  # Bajando
        else:
            tendencia = 2  # Estable

        return (
            cabeza_cuad_x,
            cabeza_cuad_y,
            dir_cercano,
            dist_cercano,
            dir_urgente,
            urgencia,
            puntos_cat,
            num_ratones,
            zona,
            tendencia,
            densidad,
            amenaza
        )

    def step(self, accion: int) -> Tuple[Tuple, float, bool]:
        """Ejecuta acción con REWARD SHAPING mejorado."""
        if self.game_over:
            return self.get_state(), 0, True

        self.ultimo_puntos = self.puntos

        # Mover cabeza
        velocidad = 5
        if accion == 0:  # Arriba
            self.cabeza_y -= velocidad
        elif accion == 1:  # Derecha
            self.cabeza_x += velocidad
        elif accion == 2:  # Abajo
            self.cabeza_y += velocidad
        elif accion == 3:  # Izquierda
            self.cabeza_x -= velocidad

        # Límites
        self.cabeza_x = max(0, min(self.ancho, self.cabeza_x))
        self.cabeza_y = max(0, min(self.alto, self.cabeza_y))

        # Spawn ratones
        self.tiempo_spawn += 1
        if self.tiempo_spawn >= self.intervalo_spawn:
            self.ratones.append(Mouse(
                float(self.ancho // 2),
                float(self.alto // 2),
                self.ancho,
                self.alto
            ))
            self.tiempo_spawn = 0

        # Actualizar ratones
        for raton in self.ratones:
            raton.actualizar()

        # REWARD SHAPING MEJORADO
        recompensa = 0

        # 1. Recompensa por acercarse al ratón más cercano
        if len(self.ratones) > 0:
            raton_cercano = min(
                self.ratones,
                key=lambda r: math.sqrt((r.x - self.cabeza_x)**2 + (r.y - self.cabeza_y)**2)
            )
            dist = math.sqrt((raton_cercano.x - self.cabeza_x)**2 + (raton_cercano.y - self.cabeza_y)**2)

            # Recompensa por proximidad (incentiva acercarse)
            if dist < 50:
                recompensa += 0.5
            elif dist < 100:
                recompensa += 0.2

            # Penalización si ratón está muy cerca del borde
            def dist_borde(r):
                return min(r.y, self.alto - r.y, r.x, self.ancho - r.x)

            if dist_borde(raton_cercano) < 100:
                recompensa -= 0.3  # Urgencia

        # Colisiones y pérdidas
        ratones_a_eliminar = []

        for i, raton in enumerate(self.ratones):
            dx = raton.x - self.cabeza_x
            dy = raton.y - self.cabeza_y
            distancia = math.sqrt(dx * dx + dy * dy)

            if distancia < (self.radio_cabeza + raton.radio):
                ratones_a_eliminar.append(i)
                self.puntos += 1
                self.capturas_recientes += 1
                recompensa += 20  # Gran recompensa por capturar

            elif raton.fuera_de_pantalla(self.ancho, self.alto):
                ratones_a_eliminar.append(i)
                self.puntos -= 1
                self.perdidas_recientes += 1
                recompensa -= 10  # Penalización por perder

        # Eliminar ratones
        for i in sorted(set(ratones_a_eliminar), reverse=True):
            del self.ratones[i]

        # Bonificación por racha de capturas
        if self.capturas_recientes > self.perdidas_recientes + 2:
            recompensa += 1.0

        # Pequeña recompensa por sobrevivir
        recompensa += 0.02

        # Game Over
        if self.puntos <= 0:
            self.game_over = True
            recompensa -= 200  # Gran penalización

        self.pasos += 1
        if self.pasos >= self.max_pasos:
            self.game_over = True

        return self.get_state(), recompensa, self.game_over


# ============================================================================
# Q-LEARNING AGENT
# ============================================================================

class QLearningAgentMejorado:
    """Agente Q-Learning compatible con el entorno mejorado."""

    def __init__(
        self,
        num_acciones: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        self.num_acciones = num_acciones
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        from collections import defaultdict
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(num_acciones)
        )

    def get_action(self, estado: Tuple) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.num_acciones - 1)
        else:
            return int(np.argmax(self.q_table[estado]))

    def update(self, estado: Tuple, accion: int, recompensa: float, siguiente_estado: Tuple):
        q_actual = self.q_table[estado][accion]
        q_siguiente_max = np.max(self.q_table[siguiente_estado])

        nuevo_q = q_actual + self.lr * (recompensa + self.gamma * q_siguiente_max - q_actual)
        self.q_table[estado][accion] = nuevo_q

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, filename: str):
        try:
            with open(filename, "rb") as f:
                from collections import defaultdict
                self.q_table = defaultdict(lambda: np.zeros(self.num_acciones))
                self.q_table.update(pickle.load(f))
        except FileNotFoundError:
            pass


# ============================================================================
# INDIVIDUO CON MUTACIÓN ADAPTATIVA
# ============================================================================

class IndividuoMejorado:
    """Individuo con mutación adaptativa y diversidad forzada."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        id_individuo: int = 0
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.id = id_individuo

        self.puntuacion_promedio = 0.0
        self.puntuacion_maxima = 0
        self.generaciones_sin_mejora = 0

    def mutar(self, tasa_base: float = 0.2, forzar_diversidad: bool = False) -> 'IndividuoMejorado':
        """
        Mutación adaptativa PROPORCIONAL al estancamiento.

        La tasa crece gradualmente:
        - 0 gens sin mejora: tasa base
        - 5 gens sin mejora: tasa base × 1.5
        - 10 gens sin mejora: tasa base × 2.0
        - 20 gens sin mejora: tasa base × 3.0
        - 30 gens sin mejora: tasa base × 4.0
        - 50+ gens sin mejora: tasa base × 6.0 (máximo)
        """
        # Cálculo proporcional de la tasa de mutación
        multiplicador = 1.0 + (self.generaciones_sin_mejora / 10.0)

        # Limitar el multiplicador máximo a 6.0
        multiplicador = min(multiplicador, 6.0)

        tasa_mutacion = tasa_base * multiplicador

        if forzar_diversidad:
            tasa_mutacion *= 1.5  # Boost adicional

        def mutar_valor(valor, min_val, max_val):
            ruido = random.gauss(0, tasa_mutacion * (max_val - min_val))
            nuevo = valor + ruido
            return max(min_val, min(max_val, nuevo))

        return IndividuoMejorado(
            learning_rate=mutar_valor(self.learning_rate, 0.001, 0.3),
            discount_factor=mutar_valor(self.discount_factor, 0.80, 0.999),
            epsilon_decay=mutar_valor(self.epsilon_decay, 0.985, 0.999),
            epsilon_min=mutar_valor(self.epsilon_min, 0.001, 0.08),
            id_individuo=self.id
        )

    def __str__(self):
        return (
            f"Ind {self.id}: LR={self.learning_rate:.3f}, γ={self.discount_factor:.3f}, "
            f"ε_d={self.epsilon_decay:.3f}, ε_m={self.epsilon_min:.3f} | "
            f"Score: {self.puntuacion_promedio:.1f} (max: {self.puntuacion_maxima})"
        )


# ============================================================================
# ENTRENAMIENTO
# ============================================================================

def entrenar_individuo_mejorado(args: Tuple) -> IndividuoMejorado:
    """Entrena un individuo con el entorno mejorado."""
    individuo, episodios_train, episodios_eval = args

    agente = QLearningAgentMejorado(
        learning_rate=individuo.learning_rate,
        discount_factor=individuo.discount_factor,
        epsilon_decay=individuo.epsilon_decay,
        epsilon_min=individuo.epsilon_min
    )

    archivo_modelo = os.path.join(DIRECTORIO_POBLACION, f"modelo_{individuo.id}.pkl")
    if os.path.exists(archivo_modelo):
        agente.load(archivo_modelo)

    # Entrenar
    env = SnakeGameEnvMejorado()
    for _ in range(episodios_train):
        estado = env.reset()
        terminado = False

        while not terminado:
            accion = agente.get_action(estado)
            siguiente_estado, recompensa, terminado = env.step(accion)
            agente.update(estado, accion, recompensa, siguiente_estado)
            estado = siguiente_estado

        agente.decay_epsilon()

    agente.save(archivo_modelo)

    # Evaluar
    agente.epsilon = 0
    puntuaciones = []

    for _ in range(episodios_eval):
        estado = env.reset()
        terminado = False

        while not terminado:
            accion = agente.get_action(estado)
            estado, _, terminado = env.step(accion)

        puntuaciones.append(env.puntos)

    individuo.puntuacion_promedio = np.mean(puntuaciones)
    individuo.puntuacion_maxima = max(puntuaciones)

    return individuo


# ============================================================================
# ENTRENADOR EVOLUTIVO MEJORADO
# ============================================================================

class EntrenadorEvolutivoMejorado:
    """Entrenador con técnicas anti-plateau."""

    def __init__(
        self,
        tamano_poblacion: int = 8,
        episodios_entrenamiento: int = 150,
        episodios_evaluacion: int = 100
    ):
        self.tamano_poblacion = tamano_poblacion
        self.episodios_entrenamiento = episodios_entrenamiento
        self.episodios_evaluacion = episodios_evaluacion
        self.generacion = 0
        self.mejor_historico: IndividuoMejorado = None
        self.mejor_score_historico = 0
        self.generaciones_sin_mejora = 0

        # Historial para detectar estancamiento
        self.historial_scores = deque(maxlen=20)

        os.makedirs(DIRECTORIO_POBLACION, exist_ok=True)
        self.poblacion: List[IndividuoMejorado] = self._crear_poblacion_inicial()

    def _crear_poblacion_inicial(self) -> List[IndividuoMejorado]:
        """Población inicial con ALTA DIVERSIDAD."""
        poblacion = []

        for i in range(self.tamano_poblacion):
            individuo = IndividuoMejorado(
                learning_rate=random.uniform(0.001, 0.3),
                discount_factor=random.uniform(0.80, 0.999),
                epsilon_decay=random.uniform(0.985, 0.999),
                epsilon_min=random.uniform(0.001, 0.08),
                id_individuo=i
            )
            poblacion.append(individuo)

        return poblacion

    def _detectar_estancamiento(self) -> bool:
        """Detecta si estamos en un plateau."""
        if len(self.historial_scores) < 10:
            return False

        # Calcular varianza de últimos scores
        varianza = np.var(list(self.historial_scores))

        # Si varianza es muy baja, estamos estancados
        return varianza < 5.0

    def entrenar_generacion(self) -> Tuple:
        print(f"\n{'='*80}")
        print(f"GENERACIÓN {self.generacion}")
        print(f"{'='*80}\n")

        args = [
            (ind, self.episodios_entrenamiento, self.episodios_evaluacion)
            for ind in self.poblacion
        ]

        num_procesos = min(cpu_count(), self.tamano_poblacion)
        with Pool(processes=num_procesos) as pool:
            self.poblacion = pool.map(entrenar_individuo_mejorado, args)

        self.poblacion.sort(key=lambda x: x.puntuacion_promedio, reverse=True)

        print("\nResultados:")
        print("-" * 80)
        for ind in self.poblacion:
            print(ind)
        print("-" * 80)

        mejor = self.poblacion[0]
        self.historial_scores.append(mejor.puntuacion_promedio)

        # Actualizar contador de estancamiento
        if mejor.puntuacion_promedio > self.mejor_score_historico:
            self.mejor_score_historico = mejor.puntuacion_promedio
            self.mejor_historico = mejor
            self.generaciones_sin_mejora = 0

            print(f"\n🏆 NUEVO RÉCORD: {self.mejor_score_historico:.1f} puntos!")

            # Guardar mejor modelo
            archivo_origen = os.path.join(DIRECTORIO_POBLACION, f"modelo_{mejor.id}.pkl")
            if os.path.exists(archivo_origen):
                import shutil
                shutil.copy(archivo_origen, "snake_qlearning_best_v2.pkl")
        else:
            self.generaciones_sin_mejora += 1
            mejor.generaciones_sin_mejora = self.generaciones_sin_mejora

        # Mostrar información de mutación proporcional
        if self.generaciones_sin_mejora > 0:
            # Calcular multiplicador actual
            multiplicador = min(1.0 + (self.generaciones_sin_mejora / 10.0), 6.0)
            tasa_efectiva = 0.2 * multiplicador

            if self.generaciones_sin_mejora >= 15:
                print(f"\n⚠️  ESTANCAMIENTO: {self.generaciones_sin_mejora} generaciones sin mejora")
                print(f"   Tasa de mutación: {tasa_efectiva:.2f} (×{multiplicador:.1f})")
                print("   🔬 Aumentando variabilidad automáticamente...")
            elif self.generaciones_sin_mejora >= 10:
                print(f"\n⏳ Sin mejora: {self.generaciones_sin_mejora} gens | Mutación: {tasa_efectiva:.2f} (×{multiplicador:.1f})")

        return mejor, self.poblacion

    def evolucionar(self):
        """Evolución con diversidad forzada si hay estancamiento."""
        mejor = self.poblacion[0]
        estancado = self._detectar_estancamiento()

        print(f"\n{'='*80}")
        print(f"EVOLUCIÓN: Generando {self.tamano_poblacion} variantes")
        if estancado:
            print("⚡ MODO ANTI-ESTANCAMIENTO: Mutación aumentada")
        print(f"{'='*80}")

        # Estrategia adaptativa con mutación proporcional
        if estancado or self.generaciones_sin_mejora > 10:
            # ALTA DIVERSIDAD - Las mutaciones ya son proporcionales al estancamiento
            nueva_poblacion = [mejor]  # 1 Elite

            # Top 5 élites (los 5 mejores sin cambios)
            num_elites = min(5, len(self.poblacion))
            for i in range(1, num_elites):
                elite = self.poblacion[i]
                elite.id = i
                nueva_poblacion.append(elite)

            # Mutantes moderados (usarán mutación proporcional automática)
            num_moderados = int(self.tamano_poblacion * 0.3)
            for i in range(len(nueva_poblacion), len(nueva_poblacion) + num_moderados):
                mutante = mejor.mutar(tasa_base=0.20)  # Base conservadora
                mutante.id = i
                nueva_poblacion.append(mutante)

            # Mutantes agresivos con forzar_diversidad
            num_agresivos = int(self.tamano_poblacion * 0.3)
            for i in range(len(nueva_poblacion), len(nueva_poblacion) + num_agresivos):
                mutante = mejor.mutar(tasa_base=0.35, forzar_diversidad=True)
                mutante.id = i
                nueva_poblacion.append(mutante)

            # Resto completamente aleatorios (exploración pura)
            for i in range(len(nueva_poblacion), self.tamano_poblacion):
                aleatorio = IndividuoMejorado(
                    learning_rate=random.uniform(0.001, 0.3),
                    discount_factor=random.uniform(0.85, 0.999),
                    epsilon_decay=random.uniform(0.985, 0.999),
                    epsilon_min=random.uniform(0.001, 0.06),
                    id_individuo=i
                )
                nueva_poblacion.append(aleatorio)
        else:
            # Evolución normal - mutación proporcional aplicada automáticamente
            nueva_poblacion = [mejor]  # Elite

            # Todos los demás son mutaciones del mejor (con tasa proporcional)
            for i in range(1, self.tamano_poblacion):
                # Tasa base más conservadora cuando no hay estancamiento
                mutante = mejor.mutar(tasa_base=0.18)
                mutante.id = i
                nueva_poblacion.append(mutante)

        self.poblacion = nueva_poblacion
        self.generacion += 1

    def entrenar_continuo(self):
        print("\n" + "="*80)
        print("ENTRENAMIENTO EVOLUTIVO V2 - CON ANTI-PLATEAU")
        print("="*80)
        print(f"Población: {self.tamano_poblacion} modelos (MÁXIMA DIVERSIDAD)")
        print(f"Estado: 12 dimensiones (mejorado)")
        print(f"Recompensas: Moldeadas (reward shaping)")
        print(f"Entrenamiento: {self.episodios_entrenamiento} episodios/generación")
        print(f"Evaluación: {self.episodios_evaluacion} episodios")
        print(f"Procesadores: {min(cpu_count(), self.tamano_poblacion)} en paralelo")
        print("\nMEJORAS:")
        print("  ✓ Estado más rico (12 dims)")
        print("  ✓ Reward shaping (8 tipos)")
        print("  ✓ Mutación adaptativa (0.15-0.6)")
        print("  ✓ Diversidad forzada (50 agentes)")
        print("  ✓ Anti-estancamiento automático")
        print("\nPresiona Ctrl+C para detener")
        print("="*80)

        try:
            while True:
                inicio = time.time()
                mejor, poblacion = self.entrenar_generacion()
                self.evolucionar()
                tiempo = time.time() - inicio

                # Calcular tasa de mutación actual
                multiplicador = min(1.0 + (self.generaciones_sin_mejora / 10.0), 6.0)
                tasa_actual = 0.18 * multiplicador  # Tasa base normal

                print(f"\nTiempo: {tiempo:.1f}s | Mejor histórico: {self.mejor_score_historico:.1f}")
                print(f"Sin mejora: {self.generaciones_sin_mejora} gens | Mutación: {tasa_actual:.3f} (×{multiplicador:.1f})")

        except KeyboardInterrupt:
            print("\n\n" + "="*80)
            print("ENTRENAMIENTO INTERRUMPIDO")
            print("="*80)
            print(f"Generaciones: {self.generacion}")
            print(f"Mejor score: {self.mejor_score_historico:.1f}")
            if self.mejor_historico:
                print(f"\nMejor configuración:")
                print(self.mejor_historico)


# ============================================================================
# MAIN
# ============================================================================

def main():
    entrenador = EntrenadorEvolutivoMejorado(
        tamano_poblacion=POBLACION_SIZE,
        episodios_entrenamiento=EPISODIOS_ENTRENAMIENTO,
        episodios_evaluacion=EPISODIOS_EVALUACION
    )

    entrenador.entrenar_continuo()


if __name__ == "__main__":
    main()
