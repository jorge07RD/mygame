"""Juego de simulación de cadena con física de restricciones."""
import pygame
from config import CONFIG
from fisica import inicializar_cadena, actualizar_cadena
from renderizado import dibujar_cadena


def main() -> None:
    """Función principal del juego."""
    # Inicializar Pygame
    pygame.init()
    pantalla = pygame.display.set_mode((CONFIG.ANCHO, CONFIG.ALTO))
    pygame.display.set_caption("Simulación de Cadena")
    clock = pygame.time.Clock()

    # Inicializar la cadena de puntos
    pos_inicial = [CONFIG.ANCHO // 2, CONFIG.ALTO // 2]
    puntos_cadena = inicializar_cadena(CONFIG.DISTANCIAS_SEGMENTOS, pos_inicial)

    # Bucle principal del juego
    running = True
    while running:
        # Procesar eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # El ancla sigue la posición del mouse
        ancla_pos = list(pygame.mouse.get_pos())

        # Limpiar pantalla
        pantalla.fill(CONFIG.NEGRO)

        # Actualizar física de la cadena
        actualizar_cadena(puntos_cadena, ancla_pos, CONFIG.DISTANCIAS_SEGMENTOS)

        # Dibujar todo
        dibujar_cadena(pantalla, ancla_pos, puntos_cadena, CONFIG.DISTANCIAS_SEGMENTOS)

        # Actualizar pantalla
        pygame.display.flip()

        # Limitar a FPS configurados
        clock.tick(CONFIG.FPS)

    # Cerrar Pygame
    pygame.quit()


if __name__ == "__main__":
    main()
