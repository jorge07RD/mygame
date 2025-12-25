import time
import sys

import pygame
import numpy as np
import moderngl

# Simple viewer using moderngl + pygame
# Instalación recomendada:
#   pip install moderngl pygame numpy

VERTEX_SHADER = '''#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
'''

FRAGMENT_SHADER = '''#version 330
in vec2 v_uv;
out vec4 fragColor;
uniform vec2 iResolution;
uniform float iTime;

vec3 palette( float t ) {
    vec3 a = vec3(0.498,-0.402,1.118);
    vec3 b = vec3(0.158,0.608,1.148);
    vec3 c = vec3(-1.262,1.078,0.988);
    vec3 d = vec3(-1.232,-1.433,-4.083);
    return a + b * cos( 6.28318*(c*t+d) );
}

void main() {
    // convertir v_uv (-1..1) a coordenadas de fragmento en píxeles
    vec2 fragCoord = (v_uv * 0.5 + 0.5) * iResolution;
    vec2 uv = (fragCoord * 2.0 - iResolution.xy) / iResolution.y;
    vec2 uv0 = uv;
    vec3 finalColor = vec3(0.0);

    for (int ii = 0; ii < 4; ii++) {
        float i = float(ii);
        uv = fract(uv * 1.5) - 0.5;

        float d = length(uv) * exp(-length(uv0));

        vec3 col = palette(length(uv0) + i*0.4 + iTime*0.4);

        d = sin(d*8. + iTime)/8.0;
        d = abs(d);

        d = pow(0.01 / d, 1.2);

        finalColor += col * d;
    }

    fragColor = vec4(finalColor, 1.0);
}
'''


def main():
    pygame.init()
    width, height = 800, 600

    # Pedir un contexto OpenGL 3.3 core
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)

    screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption('Shader viewer')

    ctx = moderngl.create_context()

    # Crear programa
    prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

    # Cuadrado que cubre la pantalla usando TRIANGLE_STRIP
    vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    ], dtype='f4')

    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, 'in_pos')

    start = time.time()

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        current_time = time.time() - start

        # actualizar uniforms
        prog['iTime'].value = float(current_time)
        prog['iResolution'].value = (float(width), float(height))

        ctx.clear(0.0, 0.0, 0.0, 0.0)
        vao.render(moderngl.TRIANGLE_STRIP)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Error:', e)
        print('Asegúrate de tener instaladas las dependencias: pip install moderngl pygame numpy')
        sys.exit(1)
