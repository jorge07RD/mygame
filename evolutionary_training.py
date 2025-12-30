"""
Sistema de entrenamiento evolutivo genético para el agente de Snake.
Ejecuta 6 modelos en paralelo, selecciona el mejor y genera 6 variantes.
"""

import os
import sys
import time
import random
import pickle
import numpy as np
from typing import List, Tuple, Dict
from multiprocessing import Pool, cpu_count
from autogame import SnakeGameEnv, QLearningAgent, entrenar, probar_agente


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

POBLACION_SIZE = 6  # Número de modelos por generación
EPISODIOS_EVALUACION = 50  # Episodios para evaluar cada modelo
DIRECTORIO_POBLACION = "poblacion"  # Carpeta para guardar modelos


# ============================================================================
# CLASE INDIVIDUO - Representa un modelo con sus hiperparámetros
# ============================================================================

class Individuo:
    """Representa un modelo de RL con sus hiperparámetros."""

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

        # Métricas de rendimiento
        self.puntuacion_promedio = 0.0
        self.puntuacion_maxima = 0
        self.pasos_promedio = 0.0
        self.recompensa_promedio = 0.0

    def mutar(self, tasa_mutacion: float = 0.2) -> 'Individuo':
        """
        Crea una variante mutada de este individuo.

        Args:
            tasa_mutacion: Magnitud de la mutación (0-1)

        Returns:
            Nuevo individuo con parámetros mutados
        """
        def mutar_valor(valor, min_val, max_val):
            """Muta un valor añadiendo ruido gaussiano."""
            ruido = random.gauss(0, tasa_mutacion * (max_val - min_val))
            nuevo = valor + ruido
            return max(min_val, min(max_val, nuevo))

        return Individuo(
            learning_rate=mutar_valor(self.learning_rate, 0.01, 0.5),
            discount_factor=mutar_valor(self.discount_factor, 0.8, 0.99),
            epsilon_decay=mutar_valor(self.epsilon_decay, 0.99, 0.999),
            epsilon_min=mutar_valor(self.epsilon_min, 0.001, 0.05),
            id_individuo=self.id
        )

    def __str__(self):
        return (
            f"Individuo {self.id}: "
            f"LR={self.learning_rate:.3f}, "
            f"γ={self.discount_factor:.3f}, "
            f"ε_decay={self.epsilon_decay:.3f}, "
            f"ε_min={self.epsilon_min:.3f} | "
            f"Score: {self.puntuacion_promedio:.1f} (max: {self.puntuacion_maxima})"
        )


# ============================================================================
# ENTRENAMIENTO Y EVALUACIÓN
# ============================================================================

def entrenar_individuo(args: Tuple[Individuo, int, int]) -> Individuo:
    """
    Entrena y evalúa un individuo.

    Args:
        args: Tupla (individuo, episodios_entrenamiento, episodios_evaluacion)

    Returns:
        Individuo con métricas actualizadas
    """
    individuo, episodios_train, episodios_eval = args

    # Crear agente con los hiperparámetros del individuo
    agente = QLearningAgent(
        learning_rate=individuo.learning_rate,
        discount_factor=individuo.discount_factor,
        epsilon_decay=individuo.epsilon_decay,
        epsilon_min=individuo.epsilon_min
    )

    # Intentar cargar modelo previo si existe
    archivo_modelo = os.path.join(DIRECTORIO_POBLACION, f"modelo_{individuo.id}.pkl")
    if os.path.exists(archivo_modelo):
        agente.load(archivo_modelo)

    # Entrenar
    env = SnakeGameEnv()
    for _ in range(episodios_train):
        estado = env.reset()
        terminado = False

        while not terminado:
            accion = agente.get_action(estado)
            siguiente_estado, recompensa, terminado = env.step(accion)
            agente.update(estado, accion, recompensa, siguiente_estado)
            estado = siguiente_estado

        agente.decay_epsilon()

    # Guardar modelo entrenado
    agente.save(archivo_modelo)

    # Evaluar (sin exploración)
    agente.epsilon = 0
    puntuaciones = []
    pasos_totales = []
    recompensas = []

    for _ in range(episodios_eval):
        estado = env.reset()
        terminado = False
        recompensa_episodio = 0

        while not terminado:
            accion = agente.get_action(estado)
            estado, recompensa, terminado = env.step(accion)
            recompensa_episodio += recompensa

        puntuaciones.append(env.puntos)
        pasos_totales.append(env.pasos)
        recompensas.append(recompensa_episodio)

    # Actualizar métricas
    individuo.puntuacion_promedio = np.mean(puntuaciones)
    individuo.puntuacion_maxima = max(puntuaciones)
    individuo.pasos_promedio = np.mean(pasos_totales)
    individuo.recompensa_promedio = np.mean(recompensas)

    return individuo


# ============================================================================
# ALGORITMO EVOLUTIVO
# ============================================================================

class EntrenadorEvolutivo:
    """Gestiona el entrenamiento evolutivo de múltiples agentes."""

    def __init__(
        self,
        tamano_poblacion: int = 6,
        episodios_entrenamiento: int = 100,
        episodios_evaluacion: int = 50
    ):
        self.tamano_poblacion = tamano_poblacion
        self.episodios_entrenamiento = episodios_entrenamiento
        self.episodios_evaluacion = episodios_evaluacion
        self.generacion = 0
        self.mejor_historico: Individuo = None
        self.mejor_score_historico = 0

        # Crear directorio para población
        os.makedirs(DIRECTORIO_POBLACION, exist_ok=True)

        # Inicializar población
        self.poblacion: List[Individuo] = self._crear_poblacion_inicial()

    def _crear_poblacion_inicial(self) -> List[Individuo]:
        """Crea la población inicial con hiperparámetros aleatorios."""
        poblacion = []

        for i in range(self.tamano_poblacion):
            individuo = Individuo(
                learning_rate=random.uniform(0.05, 0.3),
                discount_factor=random.uniform(0.85, 0.98),
                epsilon_decay=random.uniform(0.99, 0.998),
                epsilon_min=random.uniform(0.005, 0.03),
                id_individuo=i
            )
            poblacion.append(individuo)

        return poblacion

    def entrenar_generacion(self) -> Tuple[Individuo, List[Individuo]]:
        """
        Entrena una generación completa.

        Returns:
            (mejor_individuo, poblacion_completa)
        """
        print(f"\n{'='*80}")
        print(f"GENERACIÓN {self.generacion}")
        print(f"{'='*80}\n")

        # Preparar argumentos para entrenamiento paralelo
        args = [
            (ind, self.episodios_entrenamiento, self.episodios_evaluacion)
            for ind in self.poblacion
        ]

        # Entrenar en paralelo
        num_procesos = min(cpu_count(), self.tamano_poblacion)
        with Pool(processes=num_procesos) as pool:
            self.poblacion = pool.map(entrenar_individuo, args)

        # Ordenar por rendimiento
        self.poblacion.sort(key=lambda x: x.puntuacion_promedio, reverse=True)

        # Mostrar resultados
        print("\nResultados de la generación:")
        print("-" * 80)
        for ind in self.poblacion:
            print(ind)
        print("-" * 80)

        mejor = self.poblacion[0]

        # Actualizar mejor histórico
        if mejor.puntuacion_promedio > self.mejor_score_historico:
            self.mejor_score_historico = mejor.puntuacion_promedio
            self.mejor_historico = mejor
            print(f"\n🏆 ¡NUEVO RÉCORD! Score: {self.mejor_score_historico:.1f}")

            # Guardar mejor modelo
            archivo_mejor = "snake_qlearning_best.pkl"
            archivo_origen = os.path.join(DIRECTORIO_POBLACION, f"modelo_{mejor.id}.pkl")
            if os.path.exists(archivo_origen):
                import shutil
                shutil.copy(archivo_origen, archivo_mejor)
                print(f"Mejor modelo guardado en {archivo_mejor}")

        return mejor, self.poblacion

    def evolucionar(self):
        """Crea la siguiente generación a partir del mejor individuo."""
        mejor = self.poblacion[0]

        print(f"\n{'='*80}")
        print(f"EVOLUCIÓN: Generando {self.tamano_poblacion} variantes del mejor modelo")
        print(f"{'='*80}")

        # Crear nueva población: 1 élite + 5 mutaciones
        nueva_poblacion = [mejor]  # Mantener el mejor sin cambios

        for i in range(1, self.tamano_poblacion):
            mutante = mejor.mutar(tasa_mutacion=0.15)
            mutante.id = i
            nueva_poblacion.append(mutante)

        self.poblacion = nueva_poblacion
        self.generacion += 1

    def entrenar_continuo(self):
        """Entrena indefinidamente hasta interrupción."""
        print("\n" + "="*80)
        print("ENTRENAMIENTO EVOLUTIVO CONTINUO")
        print("="*80)
        print(f"Población: {self.tamano_poblacion} modelos")
        print(f"Episodios de entrenamiento: {self.episodios_entrenamiento}")
        print(f"Episodios de evaluación: {self.episodios_evaluacion}")
        print("\nPresiona Ctrl+C para detener")
        print("="*80)

        try:
            while True:
                inicio = time.time()

                # Entrenar generación
                mejor, poblacion = self.entrenar_generacion()

                # Evolucionar
                self.evolucionar()

                # Tiempo transcurrido
                tiempo = time.time() - inicio
                print(f"\nTiempo de generación: {tiempo:.1f}s")
                print(f"Mejor histórico: {self.mejor_score_historico:.1f} puntos")

        except KeyboardInterrupt:
            print("\n\n" + "="*80)
            print("ENTRENAMIENTO INTERRUMPIDO")
            print("="*80)
            print(f"Generaciones completadas: {self.generacion}")
            print(f"Mejor score alcanzado: {self.mejor_score_historico:.1f}")

            if self.mejor_historico:
                print("\nMejor configuración encontrada:")
                print(self.mejor_historico)

            print("\nModelos guardados en:")
            print(f"  - Mejor modelo: snake_qlearning_best.pkl")
            print(f"  - Población: {DIRECTORIO_POBLACION}/modelo_*.pkl")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Ejecuta el entrenamiento evolutivo."""
    entrenador = EntrenadorEvolutivo(
        tamano_poblacion=POBLACION_SIZE,
        episodios_entrenamiento=100,  # Episodios de entrenamiento por generación
        episodios_evaluacion=EPISODIOS_EVALUACION
    )

    entrenador.entrenar_continuo()


if __name__ == "__main__":
    main()
