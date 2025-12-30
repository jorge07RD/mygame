"""
Visualización de cómo aumenta la tasa de mutación con el estancamiento.
"""

import matplotlib.pyplot as plt
import numpy as np


def calcular_tasa_mutacion(generaciones_sin_mejora, tasa_base=0.18):
    """Calcula la tasa de mutación según generaciones sin mejora."""
    multiplicador = min(1.0 + (generaciones_sin_mejora / 10.0), 6.0)
    return tasa_base * multiplicador, multiplicador


def main():
    # Rango de generaciones sin mejora
    generaciones = np.arange(0, 61, 1)

    # Calcular tasas
    tasas_normal = []
    tasas_agresiva = []
    multiplicadores = []

    for gen in generaciones:
        tasa_normal, mult = calcular_tasa_mutacion(gen, tasa_base=0.18)
        tasa_agresiva, _ = calcular_tasa_mutacion(gen, tasa_base=0.35)

        tasas_normal.append(tasa_normal)
        tasas_agresiva.append(tasa_agresiva)
        multiplicadores.append(mult)

    # Crear gráfica
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Gráfica 1: Tasa de mutación
    ax1.plot(generaciones, tasas_normal, 'b-', linewidth=2, label='Normal (base=0.18)')
    ax1.plot(generaciones, tasas_agresiva, 'r--', linewidth=2, label='Agresiva (base=0.35)')
    ax1.axhline(y=0.18, color='gray', linestyle=':', alpha=0.5, label='Tasa base')
    ax1.axhline(y=1.08, color='gray', linestyle=':', alpha=0.5, label='Tasa máxima (×6)')

    # Zonas de interés
    ax1.axvspan(0, 10, alpha=0.1, color='green', label='Zona normal')
    ax1.axvspan(10, 30, alpha=0.1, color='yellow', label='Zona alerta')
    ax1.axvspan(30, 60, alpha=0.1, color='red', label='Zona crítica')

    ax1.set_xlabel('Generaciones sin mejora', fontsize=12)
    ax1.set_ylabel('Tasa de mutación', fontsize=12)
    ax1.set_title('Evolución de la Tasa de Mutación con Estancamiento', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Gráfica 2: Multiplicador
    ax2.plot(generaciones, multiplicadores, 'g-', linewidth=2)
    ax2.fill_between(generaciones, 1, multiplicadores, alpha=0.3, color='green')
    ax2.axhline(y=6.0, color='red', linestyle='--', label='Máximo (×6)')

    ax2.set_xlabel('Generaciones sin mejora', fontsize=12)
    ax2.set_ylabel('Multiplicador', fontsize=12)
    ax2.set_title('Multiplicador de Mutación', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Anotaciones importantes
    puntos_clave = [0, 10, 20, 30, 50]
    for gen in puntos_clave:
        tasa, mult = calcular_tasa_mutacion(gen, 0.18)
        ax1.annotate(f'{gen}g: ×{mult:.1f}\n{tasa:.3f}',
                    xy=(gen, tasa), xytext=(gen, tasa + 0.1),
                    fontsize=8, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()
    plt.savefig('mutacion_proporcional.png', dpi=150, bbox_inches='tight')
    print("✓ Gráfica guardada en: mutacion_proporcional.png")

    # Imprimir tabla
    print("\n" + "="*80)
    print("TABLA DE MUTACIÓN PROPORCIONAL")
    print("="*80)
    print(f"{'Gens sin mejora':<20} {'Multiplicador':<15} {'Tasa Normal':<15} {'Tasa Agresiva':<15}")
    print("-"*80)

    for gen in [0, 5, 10, 15, 20, 25, 30, 40, 50, 60]:
        tasa_n, mult = calcular_tasa_mutacion(gen, 0.18)
        tasa_a, _ = calcular_tasa_mutacion(gen, 0.35)
        print(f"{gen:<20} {mult:<15.2f} {tasa_n:<15.3f} {tasa_a:<15.3f}")

    print("="*80)
    print("\nINTERPRETACIÓN:")
    print("  🟢 0-10 gens: Mutación conservadora (exploración local)")
    print("  🟡 10-30 gens: Mutación aumentada (exploración moderada)")
    print("  🔴 30+ gens: Mutación agresiva (exploración global)")
    print("\nCuanto más tiempo estancado, mayor diversidad se introduce.")
    print("="*80)

    plt.show()


if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("⚠️  matplotlib no está instalado.")
        print("Ejecuta: pip install matplotlib")
        print("\nMostrando solo la tabla:")

        # Tabla sin gráfica
        print("\n" + "="*80)
        print("TABLA DE MUTACIÓN PROPORCIONAL")
        print("="*80)
        print(f"{'Gens sin mejora':<20} {'Multiplicador':<15} {'Tasa Normal':<15} {'Tasa Agresiva':<15}")
        print("-"*80)

        for gen in [0, 5, 10, 15, 20, 25, 30, 40, 50, 60]:
            mult = min(1.0 + (gen / 10.0), 6.0)
            tasa_n = 0.18 * mult
            tasa_a = 0.35 * mult
            print(f"{gen:<20} {mult:<15.2f} {tasa_n:<15.3f} {tasa_a:<15.3f}")

        print("="*80)
