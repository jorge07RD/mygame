# 🚀 Sistema de Entrenamiento V2 - Rompiendo el Plateau de 84 Puntos

## 📊 Diagnóstico del Problema Original

Tu sistema estaba estancado en 84 puntos debido a:

1. **Estado simplificado**: Solo 8 dimensiones → poca información
2. **Convergencia prematura**: Todos con LR=0.010, γ=0.990
3. **Mutación conservadora**: tasa=0.15 demasiado pequeña
4. **Pocas evaluaciones**: 50 episodios → alta varianza
5. **Recompensas básicas**: No incentiva comportamientos intermedios

## 🎯 Mejoras Implementadas en V2

### 1. Estado Mejorado (8 → 12 dimensiones)

**Original:**
- Cuadrante cabeza (2 dims)
- Ratón cercano dirección/distancia (2 dims)
- Ratón urgente dirección/distancia (2 dims)
- Puntos + num ratones (2 dims)
**Total: 8 dimensiones**

**V2:**
- ✅ Todo lo anterior +
- ✅ Zona de pantalla (9 zonas en grid 3x3)
- ✅ Tendencia de puntos (subiendo/bajando/estable)
- ✅ Densidad de ratones (baja/media/alta)
- ✅ Amenaza inmediata (ratón a punto de escapar)
**Total: 12 dimensiones**

→ **33% más información** para tomar decisiones

### 2. Reward Shaping (Recompensas Moldeadas)

**Original:**
```python
+10  captura ratón
-5   pierde ratón
-100 game over
```

**V2:**
```python
+20     captura ratón (aumentado!)
-10     pierde ratón (aumentado!)
+0.5    si ratón muy cerca (< 50 px)
+0.2    si ratón cerca (< 100 px)
-0.3    si ratón urgente (cerca del borde)
+1.0    bonificación por racha de capturas
+0.02   pequeña recompensa por sobrevivir
-200    game over (aumentado!)
```

→ **Incentiva comportamientos intermedios**, no solo el resultado final

### 3. Mutación Adaptativa

**Original:**
```python
tasa_mutacion = 0.15  # Siempre fija
```

**V2:**
```python
tasa_base = 0.2

# Si estancado 10 generaciones
tasa = tasa_base * 2.0

# Si estancado 20 generaciones
tasa = tasa_base * 3.0

# Si detecta plateau
tasa = tasa_base * 1.5
```

→ **Aumenta exploración automáticamente** cuando se detecta estancamiento

### 4. Diversidad Forzada

**Original:**
```python
# 1 elite + 5 mutaciones del mejor
nueva_poblacion = [mejor]
for i in range(5):
    mutante = mejor.mutar(0.15)
```

**V2 (cuando hay estancamiento):**
```python
# 1 elite
# 2 mutantes moderados
# 3 mutantes agresivos
# 2 COMPLETAMENTE ALEATORIOS
nueva_poblacion = [elite] + moderados + agresivos + aleatorios
```

→ **Evita convergencia prematura** con individuos completamente nuevos

### 5. Más Evaluaciones

**Original:** 50 episodios
**V2:** 100 episodios

→ **Reduce varianza** en la evaluación, scores más confiables

### 6. Población Más Grande

**Original:** 6 individuos
**V2:** 8 individuos

→ **Mayor diversidad** genética

## 🎮 Cómo Usar

### Opción 1: Empezar desde cero (recomendado)

```bash
python evolutionary_training_v2.py
```

Esto creará una carpeta `poblacion_v2/` con los modelos mejorados.

### Opción 2: Migrar tu mejor modelo actual

Si quieres aprovechar lo que ya aprendiste:

```bash
# 1. Copiar tu mejor modelo a la nueva carpeta
mkdir -p poblacion_v2
cp snake_qlearning_best.pkl poblacion_v2/modelo_0.pkl

# 2. Ejecutar V2 (usará modelo_0 como base)
python evolutionary_training_v2.py
```

### Probar el modelo V2

```bash
# Modificar test_modelo.py para usar estado de 12 dims
python test_modelo.py snake_qlearning_best_v2.pkl
```

## 📈 Qué Esperar

Con estas mejoras, deberías ver:

1. **Primeras 20 generaciones**: Exploración caótica, puede bajar
2. **Generaciones 20-50**: Comienza a superar 84 puntos
3. **Generaciones 50-100**: Debería alcanzar 100-120 puntos
4. **Generaciones 100+**: Potencial de 150+ puntos

## 🔍 Monitoreo

El sistema te avisará:

```
⚠️  ESTANCAMIENTO: 15 generaciones sin mejora
   Aumentando mutación...
```

```
⚡ MODO ANTI-ESTANCAMIENTO: Mutación aumentada
```

Cuando veas estos mensajes, el sistema está **activamente** intentando romper el plateau.

## 🎛️ Ajustes Opcionales

Si después de 100 generaciones todavía estás estancado:

### Aumentar aún más el estado

Editar `get_state()` para añadir:
- Velocidad de ratones
- Historia de últimas 3 acciones
- Distancia a los bordes de la pantalla

### Cambiar a Deep Q-Learning

Para estados muy complejos, considera usar una red neuronal en lugar de tabla Q:

```bash
# Requerirá PyTorch o TensorFlow
pip install torch
```

(Puedo ayudarte a implementar esto si lo necesitas)

### Ajustar hiperparámetros

En el archivo `evolutionary_training_v2.py`:

```python
# Línea 20-23
POBLACION_SIZE = 12  # Más diversidad (más lento)
EPISODIOS_ENTRENAMIENTO = 200  # Más aprendizaje por generación
EPISODIOS_EVALUACION = 150  # Evaluación más precisa
```

## 🆚 Comparación de Rendimiento

| Métrica | V1 Original | V2 Mejorado | Mejora |
|---------|-------------|-------------|---------|
| **Estado** | 8 dims | 12 dims | +50% |
| **Población** | 6 | 8 | +33% |
| **Evaluación** | 50 eps | 100 eps | +100% |
| **Mutación** | Fija 0.15 | Adaptativa 0.2-0.6 | 2-4x |
| **Recompensas** | 3 tipos | 8 tipos | +167% |
| **Diversidad** | Baja | Forzada | ∞ |

## 💡 Próximos Pasos si Sigue Estancado

1. **Curriculum Learning**: Entrenar primero en versión fácil del juego
2. **Deep Q-Network**: Usar red neuronal en vez de tabla
3. **PPO/A3C**: Algoritmos más modernos que Q-Learning
4. **Imitation Learning**: Jugar tú mismo y que aprenda de ti

## 📞 Necesitas Ayuda?

Si después de ejecutar V2 necesitas más optimizaciones, puedo:
- Implementar Deep Q-Learning
- Ajustar las recompensas específicamente
- Añadir más dimensiones al estado
- Visualizar la tabla Q para debugging
