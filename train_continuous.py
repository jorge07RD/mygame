"""
Entrenamiento continuo del agente - Mejora el modelo automáticamente
ejecutando partidas y guardando el progreso periódicamente.
"""

import sys
from autogame import entrenar, probar_agente

def main():
    """
    Entrenamiento continuo mejorado.
    El modelo se entrena en bloques y se guarda cada cierto tiempo.
    """
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO CONTINUO - SNAKE AI")
    print("=" * 70)
    print("\nEste programa mejorará tu modelo existente entrenándolo continuamente.")
    print("Puedes interrumpirlo en cualquier momento con Ctrl+C")
    print("El progreso se guarda automáticamente cada 100 episodios.\n")

    # Configuración
    episodios_por_bloque = 500  # Cuántos episodios entrenar por bloque
    bloques_totales = 10  # Cuántos bloques ejecutar
    archivo_modelo = "snake_qlearning.pkl"

    try:
        bloque_actual = 1
        while bloque_actual <= bloques_totales:
            print(f"\n{'='*70}")
            print(f"BLOQUE {bloque_actual}/{bloques_totales}")
            print(f"{'='*70}\n")

            # Entrenar un bloque
            agente, stats = entrenar(
                num_episodios=episodios_por_bloque,
                guardar_cada=100,
                mostrar_cada=50,
                archivo_modelo=archivo_modelo
            )

            # Probar el agente después de cada bloque
            print(f"\n--- Prueba después del bloque {bloque_actual} ---")
            probar_agente(archivo_modelo, num_partidas=5)

            bloque_actual += 1

        print("\n" + "=" * 70)
        print("ENTRENAMIENTO COMPLETO!")
        print(f"Se entrenaron {bloques_totales * episodios_por_bloque} episodios en total")
        print("=" * 70)

        # Prueba final extendida
        print("\nPRUEBA FINAL (20 partidas):")
        probar_agente(archivo_modelo, num_partidas=20)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("ENTRENAMIENTO INTERRUMPIDO")
        print("=" * 70)
        print("El progreso se ha guardado automáticamente.")
        print(f"Puedes continuar ejecutando este script de nuevo.")

        # Probar el estado actual
        print("\nProbando el modelo actual:")
        probar_agente(archivo_modelo, num_partidas=5)


if __name__ == "__main__":
    main()
