"""
Deep Q-Network (DQN) V3 con PyTorch para el juego de la serpiente.

MEJORAS sobre V2:
1. Red neuronal en vez de tabla Q → generalización
2. Estados continuos en vez de discretos → más información
3. Experience Replay → aprende de pasado
4. Target Network → entrenamiento estable
5. Epsilon decay suave → exploración optimizada
"""

import os
import random
import math
import numpy as np
from collections import deque, namedtuple
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Hiperparámetros DQN
LEARNING_RATE = 0.001
GAMMA = 0.99  # Factor de descuento
BATCH_SIZE = 128  # Tamaño de batch para entrenamiento
MEMORY_SIZE = 100000  # Tamaño del replay buffer
TARGET_UPDATE = 10  # Actualizar target network cada N episodios

# Exploración
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995

# Entrenamiento
NUM_EPISODIOS = 5000
MAX_PASOS = 10000

# Arquitectura de red
HIDDEN_SIZE_1 = 256
HIDDEN_SIZE_2 = 128
HIDDEN_SIZE_3 = 64

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Usando dispositivo: {DEVICE}")


# ============================================================================
# EXPERIENCIA (para Replay Buffer)
# ============================================================================

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """Buffer circular para almacenar experiencias."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Añade una experiencia al buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Experience]:
        """Muestrea un batch aleatorio de experiencias."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# RED NEURONAL DQN
# ============================================================================

class DQN(nn.Module):
    """
    Red neuronal profunda para Q-Learning.

    Arquitectura:
    - Capa entrada: estado del juego (continuo, ~20 features)
    - 3 capas ocultas con ReLU
    - Capa salida: Q-values para cada acción (4 acciones)
    """

    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_size, HIDDEN_SIZE_1)
        self.fc2 = nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
        self.fc3 = nn.Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_3)
        self.fc4 = nn.Linear(HIDDEN_SIZE_3, action_size)

        # Inicialización de pesos (Xavier)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        """Forward pass."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# ============================================================================
# CLASE MOUSE
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


# ============================================================================
# ENTORNO CON ESTADOS CONTINUOS
# ============================================================================

class SnakeEnvContinuo:
    """
    Entorno del juego con estados CONTINUOS (no discretizados).

    Estado (20 features):
    1-2. Posición normalizada de la cabeza (x, y) [0-1]
    3-4. Velocidad/dirección de la cabeza (dx, dy)
    5-6. Ratón más cercano: distancia normalizada + ángulo
    7-8. Ratón más cercano: velocidad relativa (vx, vy)
    9-10. Ratón más urgente: distancia al borde + ángulo
    11. Número de ratones normalizado [0-1]
    12. Puntos normalizados [0-1]
    13-16. Distancia a cada borde (arriba, derecha, abajo, izquierda) [0-1]
    17-18. Posición del agujero normalizada
    19. Densidad de ratones en cuadrante actual
    20. Amenaza inmediata (hay ratón crítico) [0-1]
    """

    def __init__(self, ancho: int = 800, alto: int = 600):
        self.ancho = ancho
        self.alto = alto
        self.radio_cabeza = 24

        self.reset()

    def reset(self):
        """Reinicia el entorno."""
        self.cabeza_x = float(self.ancho // 2)
        self.cabeza_y = float(self.alto // 2)
        self.vx = 0.0  # Velocidad
        self.vy = 0.0
        self.puntos = 1
        self.game_over = False
        self.ratones: List[Mouse] = []
        self.pasos = 0
        self.max_pasos = MAX_PASOS
        self.tiempo_spawn = 0
        self.intervalo_spawn = 60

        # Posición del agujero (centro)
        self.agujero_x = float(self.ancho // 2)
        self.agujero_y = float(self.alto // 2)

        return self.get_state()

    def get_state(self) -> np.ndarray:
        """
        Retorna el estado CONTINUO como array numpy.

        Returns:
            Estado de 20 features normalizadas.
        """
        state = np.zeros(20, dtype=np.float32)

        # 1-2. Posición cabeza normalizada
        state[0] = self.cabeza_x / self.ancho
        state[1] = self.cabeza_y / self.alto

        # 3-4. Velocidad/dirección normalizada
        state[2] = self.vx / 10.0  # Normalizar velocidad
        state[3] = self.vy / 10.0

        if len(self.ratones) > 0:
            # 5-6. Ratón más cercano
            raton_cercano = min(
                self.ratones,
                key=lambda r: math.sqrt((r.x - self.cabeza_x)**2 + (r.y - self.cabeza_y)**2)
            )

            dx = raton_cercano.x - self.cabeza_x
            dy = raton_cercano.y - self.cabeza_y
            dist = math.sqrt(dx * dx + dy * dy)

            # Distancia normalizada (max ~1000px)
            state[4] = min(dist / 1000.0, 1.0)
            # Ángulo (-π a π) normalizado a [-1, 1]
            state[5] = math.atan2(dy, dx) / math.pi

            # 7-8. Velocidad relativa al ratón
            state[6] = (raton_cercano.destino_x - self.cabeza_x) / self.ancho
            state[7] = (raton_cercano.destino_y - self.cabeza_y) / self.alto

            # 9-10. Ratón más urgente (cerca del borde)
            def dist_borde(r):
                return min(r.y, self.alto - r.y, r.x, self.ancho - r.x)

            raton_urgente = min(self.ratones, key=dist_borde)
            dist_urgente = dist_borde(raton_urgente)

            state[8] = dist_urgente / (self.ancho / 2)  # Normalizado
            dx_urg = raton_urgente.x - self.cabeza_x
            dy_urg = raton_urgente.y - self.cabeza_y
            state[9] = math.atan2(dy_urg, dx_urg) / math.pi

            # 20. Amenaza inmediata
            state[19] = 1.0 if dist_urgente < 100 else 0.0
        else:
            # Sin ratones: valores por defecto
            state[4:10] = 0.0
            state[19] = 0.0

        # 11. Número de ratones normalizado (max ~10)
        state[10] = min(len(self.ratones) / 10.0, 1.0)

        # 12. Puntos normalizados (max ~100)
        state[11] = min(self.puntos / 100.0, 1.0)

        # 13-16. Distancia a bordes normalizada
        state[12] = self.cabeza_y / self.alto  # Arriba
        state[13] = (self.ancho - self.cabeza_x) / self.ancho  # Derecha
        state[14] = (self.alto - self.cabeza_y) / self.alto  # Abajo
        state[15] = self.cabeza_x / self.ancho  # Izquierda

        # 17-18. Posición agujero normalizada
        state[16] = self.agujero_x / self.ancho
        state[17] = self.agujero_y / self.alto

        # 19. Densidad en cuadrante
        if len(self.ratones) > 0:
            # Contar ratones en mismo cuadrante que cabeza
            cuad_x = 0 if self.cabeza_x < self.ancho / 2 else 1
            cuad_y = 0 if self.cabeza_y < self.alto / 2 else 1

            ratones_cuadrante = sum(
                1 for r in self.ratones
                if (0 if r.x < self.ancho / 2 else 1) == cuad_x and
                   (0 if r.y < self.alto / 2 else 1) == cuad_y
            )
            state[18] = min(ratones_cuadrante / 5.0, 1.0)
        else:
            state[18] = 0.0

        return state

    def step(self, accion: int) -> Tuple[np.ndarray, float, bool]:
        """
        Ejecuta una acción.

        Args:
            accion: 0=arriba, 1=derecha, 2=abajo, 3=izquierda

        Returns:
            (nuevo_estado, recompensa, terminado)
        """
        if self.game_over:
            return self.get_state(), 0.0, True

        # Aplicar acción (actualizar velocidad)
        velocidad = 5.0
        if accion == 0:  # Arriba
            self.vx = 0
            self.vy = -velocidad
        elif accion == 1:  # Derecha
            self.vx = velocidad
            self.vy = 0
        elif accion == 2:  # Abajo
            self.vx = 0
            self.vy = velocidad
        elif accion == 3:  # Izquierda
            self.vx = -velocidad
            self.vy = 0

        # Mover cabeza
        self.cabeza_x += self.vx
        self.cabeza_y += self.vy

        # Límites
        self.cabeza_x = max(0, min(self.ancho, self.cabeza_x))
        self.cabeza_y = max(0, min(self.alto, self.cabeza_y))

        # Spawn ratones
        self.tiempo_spawn += 1
        if self.tiempo_spawn >= self.intervalo_spawn:
            self.ratones.append(Mouse(self.agujero_x, self.agujero_y, self.ancho, self.alto))
            self.tiempo_spawn = 0

        # Actualizar ratones
        for raton in self.ratones:
            raton.actualizar()

        # REWARD SHAPING MEJORADO
        recompensa = 0.0

        # Recompensa por acercarse al ratón más cercano
        if len(self.ratones) > 0:
            raton_cercano = min(
                self.ratones,
                key=lambda r: math.sqrt((r.x - self.cabeza_x)**2 + (r.y - self.cabeza_y)**2)
            )
            dist = math.sqrt((raton_cercano.x - self.cabeza_x)**2 + (raton_cercano.y - self.cabeza_y)**2)

            # Recompensa inversa a distancia (incentiva acercarse)
            if dist < 50:
                recompensa += 1.0
            elif dist < 100:
                recompensa += 0.5
            elif dist < 200:
                recompensa += 0.2

            # Penalización si ratón urgente
            def dist_borde(r):
                return min(r.y, self.alto - r.y, r.x, self.ancho - r.x)

            if dist_borde(raton_cercano) < 100:
                recompensa -= 0.5  # Urgencia

        # Colisiones y pérdidas
        ratones_a_eliminar = []

        for i, raton in enumerate(self.ratones):
            dx = raton.x - self.cabeza_x
            dy = raton.y - self.cabeza_y
            distancia = math.sqrt(dx * dx + dy * dy)

            if distancia < (self.radio_cabeza + raton.radio):
                ratones_a_eliminar.append(i)
                self.puntos += 1
                recompensa += 50.0  # Gran recompensa por capturar

            elif raton.fuera_de_pantalla(self.ancho, self.alto):
                ratones_a_eliminar.append(i)
                self.puntos -= 1
                recompensa -= 20.0  # Penalización por perder

        # Eliminar ratones
        for i in sorted(set(ratones_a_eliminar), reverse=True):
            del self.ratones[i]

        # Pequeña recompensa por sobrevivir
        recompensa += 0.05

        # Game Over
        if self.puntos <= 0:
            self.game_over = True
            recompensa -= 300.0  # Gran penalización

        self.pasos += 1
        if self.pasos >= self.max_pasos:
            self.game_over = True

        return self.get_state(), recompensa, self.game_over


# ============================================================================
# AGENTE DQN
# ============================================================================

class DQNAgent:
    """Agente que usa Deep Q-Learning."""

    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size

        # Redes
        self.policy_net = DQN(state_size, action_size).to(DEVICE)
        self.target_net = DQN(state_size, action_size).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizador
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)

        # Replay buffer
        self.memory = ReplayBuffer(MEMORY_SIZE)

        # Exploración
        self.epsilon = EPSILON_START

        # Métricas
        self.losses = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Selecciona acción usando epsilon-greedy.

        Args:
            state: Estado actual
            training: Si True, usa epsilon-greedy; si False, siempre greedy

        Returns:
            Acción seleccionada
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def train_step(self):
        """Realiza un paso de entrenamiento."""
        if len(self.memory) < BATCH_SIZE:
            return

        # Muestrear batch
        experiences = self.memory.sample(BATCH_SIZE)
        batch = Experience(*zip(*experiences))

        # Convertir a tensores
        state_batch = torch.FloatTensor(np.array(batch.state)).to(DEVICE)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(DEVICE)
        reward_batch = torch.FloatTensor(batch.reward).to(DEVICE)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(DEVICE)
        done_batch = torch.FloatTensor(batch.done).to(DEVICE)

        # Q-values actuales
        current_q = self.policy_net(state_batch).gather(1, action_batch)

        # Q-values siguiente estado (target network)
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(1)[0]
            target_q = reward_batch + (GAMMA * next_q * (1 - done_batch))

        # Loss
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)

        # Optimizar
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.losses.append(loss.item())

    def update_target_network(self):
        """Actualiza la target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decae epsilon."""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def save(self, filepath: str):
        """Guarda el modelo."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Modelo guardado en {filepath}")

    def load(self, filepath: str):
        """Carga el modelo."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=DEVICE)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            print(f"Modelo cargado desde {filepath}")
        else:
            print(f"No se encontró {filepath}")


# ============================================================================
# ENTRENAMIENTO
# ============================================================================

def entrenar_dqn(num_episodios: int = NUM_EPISODIOS):
    """Entrena el agente DQN."""
    env = SnakeEnvContinuo()
    state_size = 20  # Tamaño del estado
    action_size = 4  # Número de acciones

    agent = DQNAgent(state_size, action_size)

    # Intentar cargar modelo previo
    agent.load("snake_dqn_v3.pth")

    # Métricas
    scores = []
    avg_scores = []

    print("\n" + "="*80)
    print("ENTRENAMIENTO DQN V3")
    print("="*80)
    print(f"Episodios: {num_episodios}")
    print(f"Device: {DEVICE}")
    print(f"Estado: {state_size} features continuas")
    print(f"Acciones: {action_size}")
    print(f"Red: {HIDDEN_SIZE_1}-{HIDDEN_SIZE_2}-{HIDDEN_SIZE_3}")
    print("="*80)

    try:
        for episodio in range(num_episodios):
            state = env.reset()
            score = 0
            done = False

            while not done:
                # Seleccionar acción
                action = agent.select_action(state, training=True)

                # Ejecutar acción
                next_state, reward, done = env.step(action)

                # Guardar experiencia
                agent.memory.push(state, action, reward, next_state, done)

                # Entrenar
                agent.train_step()

                state = next_state
                score += reward

            # Decay epsilon
            agent.decay_epsilon()

            # Actualizar target network
            if (episodio + 1) % TARGET_UPDATE == 0:
                agent.update_target_network()

            # Guardar scores
            scores.append(env.puntos)
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            avg_scores.append(avg_score)

            # Mostrar progreso
            if (episodio + 1) % 10 == 0:
                avg_loss = np.mean(agent.losses[-100:]) if agent.losses else 0
                print(f"Ep {episodio + 1:4d} | "
                      f"Score: {env.puntos:3d} | "
                      f"Avg(100): {avg_score:.1f} | "
                      f"ε: {agent.epsilon:.3f} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Buffer: {len(agent.memory)}")

            # Guardar modelo
            if (episodio + 1) % 100 == 0:
                agent.save("snake_dqn_v3.pth")

                # Mejor modelo
                if len(avg_scores) > 0 and avg_score == max(avg_scores):
                    agent.save("snake_dqn_v3_best.pth")
                    print(f"  🏆 Nuevo récord! Avg: {avg_score:.1f}")

    except KeyboardInterrupt:
        print("\n\nEntrenamiento interrumpido")

    # Guardar modelo final
    agent.save("snake_dqn_v3.pth")

    print("\n" + "="*80)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*80)
    print(f"Episodios: {len(scores)}")
    print(f"Score promedio final: {np.mean(scores[-100:]):.1f}")
    print(f"Score máximo: {max(scores)}")
    print("="*80)

    return agent, scores


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    num_eps = NUM_EPISODIOS
    if len(sys.argv) > 1:
        try:
            num_eps = int(sys.argv[1])
        except ValueError:
            pass

    agent, scores = entrenar_dqn(num_eps)
