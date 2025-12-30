"""
Script para comparar el rendimiento de V1 vs V2.
Ejecuta ambos modelos y muestra estadísticas lado a lado.
"""

import os
import pickle
import numpy as np
from autogame import SnakeGameEnv, QLearningAgent
from evolutionary_training_v2 import SnakeGameEnvMejorado, QLearningAgentMejorado


def evaluar_v1(num_episodios=100):
    """Evalúa el modelo V1 original."""
    print("\n" + "="*60)
    print("EVALUANDO MODELO V1 (Original)")
    print("="*60)

    if not os.path.exists("snake_qlearning_best.pkl"):
        print("⚠️  No se encontró snake_qlearning_best.pkl")
        return None

    agente = QLearningAgent()
    agente.load("snake_qlearning_best.pkl")
    agente.epsilon = 0  # Sin exploración

    env = SnakeGameEnv()
    puntuaciones = []
    pasos_totales = []

    for i in range(num_episodios):
        estado = env.reset()
        terminado = False

        while not terminado:
            accion = agente.get_action(estado)
            estado, _, terminado = env.step(accion)

        puntuaciones.append(env.puntos)
        pasos_totales.append(env.pasos)

        if (i + 1) % 20 == 0:
            print(f"  Progreso: {i+1}/{num_episodios} episodios")

    stats = {
        'promedio': np.mean(puntuaciones),
        'max': np.max(puntuaciones),
        'min': np.min(puntuaciones),
        'std': np.std(puntuaciones),
        'pasos_promedio': np.mean(pasos_totales),
        'estados_aprendidos': len(agente.q_table)
    }

    return stats


def evaluar_v2(num_episodios=100):
    """Evalúa el modelo V2 mejorado."""
    print("\n" + "="*60)
    print("EVALUANDO MODELO V2 (Mejorado)")
    print("="*60)

    if not os.path.exists("snake_qlearning_best_v2.pkl"):
        print("⚠️  No se encontró snake_qlearning_best_v2.pkl")
        print("   Ejecuta primero: python evolutionary_training_v2.py")
        return None

    agente = QLearningAgentMejorado()
    agente.load("snake_qlearning_best_v2.pkl")
    agente.epsilon = 0  # Sin exploración

    env = SnakeGameEnvMejorado()
    puntuaciones = []
    pasos_totales = []

    for i in range(num_episodios):
        estado = env.reset()
        terminado = False

        while not terminado:
            accion = agente.get_action(estado)
            estado, _, terminado = env.step(accion)

        puntuaciones.append(env.puntos)
        pasos_totales.append(env.pasos)

        if (i + 1) % 20 == 0:
            print(f"  Progreso: {i+1}/{num_episodios} episodios")

    stats = {
        'promedio': np.mean(puntuaciones),
        'max': np.max(puntuaciones),
        'min': np.min(puntuaciones),
        'std': np.std(puntuaciones),
        'pasos_promedio': np.mean(pasos_totales),
        'estados_aprendidos': len(agente.q_table)
    }

    return stats


def mostrar_comparacion(stats_v1, stats_v2):
    """Muestra comparación lado a lado."""
    print("\n" + "="*80)
    print("COMPARACIÓN DE RESULTADOS")
    print("="*80)

    if stats_v1 is None and stats_v2 is None:
        print("No hay modelos para comparar.")
        return

    print(f"\n{'Métrica':<25} {'V1 Original':<20} {'V2 Mejorado':<20} {'Mejora':<15}")
    print("-" * 80)

    if stats_v1 and stats_v2:
        # Comparar todas las métricas
        metricas = [
            ('Puntos Promedio', 'promedio'),
            ('Puntos Máximos', 'max'),
            ('Puntos Mínimos', 'min'),
            ('Desviación Estándar', 'std'),
            ('Pasos Promedio', 'pasos_promedio'),
            ('Estados Aprendidos', 'estados_aprendidos')
        ]

        for nombre, clave in metricas:
            v1_val = stats_v1[clave]
            v2_val = stats_v2[clave]

            if v1_val > 0:
                mejora = ((v2_val - v1_val) / v1_val) * 100
                simbolo = "📈" if mejora > 0 else ("📉" if mejora < 0 else "➡️")
                mejora_str = f"{simbolo} {mejora:+.1f}%"
            else:
                mejora_str = "N/A"

            print(f"{nombre:<25} {v1_val:<20.2f} {v2_val:<20.2f} {mejora_str:<15}")

    elif stats_v1:
        print("\nSolo V1 disponible:")
        for clave, valor in stats_v1.items():
            print(f"  {clave}: {valor:.2f}")

    elif stats_v2:
        print("\nSolo V2 disponible:")
        for clave, valor in stats_v2.items():
            print(f"  {clave}: {valor:.2f}")

    print("-" * 80)

    # Veredicto
    if stats_v1 and stats_v2:
        print("\n📊 VEREDICTO:")
        if stats_v2['promedio'] > stats_v1['promedio']:
            mejora_pct = ((stats_v2['promedio'] - stats_v1['promedio']) / stats_v1['promedio']) * 100
            print(f"✅ V2 es MEJOR: +{mejora_pct:.1f}% de puntos en promedio")
            print(f"   V1: {stats_v1['promedio']:.1f} pts → V2: {stats_v2['promedio']:.1f} pts")
        elif stats_v2['promedio'] < stats_v1['promedio']:
            print(f"❌ V2 es peor (necesita más entrenamiento)")
        else:
            print(f"➡️  Empate (considera entrenar más)")

        # Análisis adicional
        if stats_v2['max'] > stats_v1['max']:
            print(f"💪 V2 alcanzó nuevo máximo: {stats_v2['max']:.0f} pts (V1: {stats_v1['max']:.0f})")

        if stats_v2['std'] < stats_v1['std']:
            print(f"🎯 V2 es más consistente (menor varianza)")

    print("="*80)


def main():
    """Función principal."""
    import sys

    num_episodios = 100
    if len(sys.argv) > 1:
        try:
            num_episodios = int(sys.argv[1])
        except ValueError:
            pass

    print("\n" + "="*80)
    print("COMPARADOR DE MODELOS V1 vs V2")
    print("="*80)
    print(f"Evaluando con {num_episodios} episodios por modelo...")
    print("Esto puede tomar varios minutos.")
    print("="*80)

    # Evaluar ambos
    stats_v1 = evaluar_v1(num_episodios)
    stats_v2 = evaluar_v2(num_episodios)

    # Mostrar comparación
    mostrar_comparacion(stats_v1, stats_v2)

    # Guardar resultados
    if stats_v1 or stats_v2:
        import json
        resultados = {
            'v1': stats_v1,
            'v2': stats_v2,
            'num_episodios': num_episodios
        }

        with open('comparacion_resultados.json', 'w') as f:
            json.dump(resultados, f, indent=2)

        print("\n💾 Resultados guardados en: comparacion_resultados.json")


if __name__ == "__main__":
    main()
