# Proyecto 3: Space Travel 🚀

**Estudiante:** Kevin Villagrán  
**Carnet:** 23584  
**Curso:** Gráficas por Computadora

Este proyecto consiste en un motor de renderizado 3D por software (sin GPU) escrito en Rust, que simula un sistema solar completo navegable mediante una nave espacial texturizada. Implementa shaders procedurales avanzados, texturizado UV, mecánicas de vuelo y efectos de post-proceso.

---

## 🎥 Video Demostrativo

[![Demo - Proyecto 3: Space Travel](https://img.youtube.com/vi/kWOwwDbjCws/0.jpg)](https://www.youtube.com/watch?v=kWOwwDbjCws)

---

## ✅ Cumplimiento de Requerimientos

Este proyecto cumple con la totalidad de los puntos solicitados en la rúbrica:

### 1. Sistema Solar y Cuerpos Celestes (Máx 50 pts)
- **Sol y Planetas:** Se renderiza un Sol central y 5 planetas (Tierra, Marte, Júpiter, Saturno, Urano) alineados al plano eclíptico.
- **Lunas:** Se incluyen lunas orbitando la Tierra y Júpiter.
- **Movimiento:** Todos los cuerpos tienen traslación (orbitan al sol) y rotación sobre su propio eje con velocidades variables.
- **Visualización de Órbitas:** Se renderizan las líneas de las órbitas para visualizar la trayectoria (20 pts).

### 2. Nave y Cámara (70 pts combinados)
- **Nave Modelada (30 pts):** Se incluye una nave modelo "Pelican" completamente texturizada (Diffuse Map) que el jugador controla.
- **Cámara 3D (40 pts):** Implementación de una cámara en tercera persona que sigue a la nave con movimiento fluido en 3 dimensiones (Pitch, Yaw y movimiento libre), no limitado solo al plano eclíptico.
- **Cinemática:** Suavizado de cámara (lerp) para alinear la vista con la nave.

### 3. Mecánicas de Juego (30 pts combinados)
- **Instant Warping Animado (20 pts):**
  - Sistema de viaje rápido a cualquier planeta.
  - **Efecto Animado:** Incluye una animación de "burbuja warp" que distorsiona el espacio y un efecto de *White Flash* (pantalla blanca) para transicionar suavemente entre ubicaciones.
- **Colisiones (10 pts):** Sistema de detección de colisiones que impide que la nave o la cámara atraviesen los planetas, empujando al jugador fuera del radio del cuerpo celeste.

### 4. Entorno y Estética (40 pts combinados)
- **Skybox (10 pts):** Implementación de un *Cube Map* (Skybox) texturizado para simular el fondo estelar.
- **Shaders Procedurales:** Uso de ruido Perlin, Voronoi, Flowmaps y capas atmosféricas para dar estética única a cada planeta.
- **Anillos:** Shaders especiales para los anillos de Saturno y Urano.

---

## 🎮 Controles

### Navegación de la Nave
- **W**: Acelerar (aumentar velocidad).
- **S**: Frenar / Reversa.
- **J / L**: Girar nave (Yaw) - Izquierda / Derecha.
- **I / K**: Inclinar nave (Pitch) - Arriba / Abajo.
- **Espacio**: Alinear cámara suavemente detrás de la nave (Modo cinemático).

### Sistema Warp (Viaje Rápido)
Presiona el número correspondiente para iniciar el salto warp animado hacia el planeta:
- **0**: Sol ☀️
- **1**: Tierra 🌍
- **2**: Marte 🔴
- **3**: Júpiter 🪐
- **4**: Saturno 🟡
- **5**: Urano 💙
- **Backspace**: Iniciar Warp de retorno a la vista general del sistema.

---

## 🛠 Detalles Técnicos

El motor fue construido desde cero utilizando `minifb` para el manejo de la ventana y buffer, y `nalgebra-glm` para las matemáticas vectoriales.

### Características del Engine:
- **Vertex Shader:** Transformación de vértices, proyección de perspectiva y paso de coordenadas UV.
- **Fragment Shader:** - Soporte para **Texturas** (cargado de imágenes para la nave y skybox).
  - Soporte para **Shaders Procedurales** (generación de terrenos y nubes matemáticamente).
- **Rasterización:** Algoritmo de llenado de triángulos con coordenadas baricéntricas y corrección de perspectiva.
- **Z-Buffer:** Manejo de profundidad para asegurar que los objetos se dibujen en el orden correcto.
- **Blending:** Soporte para transparencias (Alpha Blending) para nubes, anillos y efectos visuales.
- **Iluminación:** Modelo de iluminación Blinn-Phong básico y luz ambiental.

---

### 🌍 Descripción de los Planetas
Cada planeta utiliza una combinación de shaders para lograr su apariencia:
1. **Sol:** Shader de "lava" con *Flow Noise* animado.
2. **Tierra:** Shader de terreno con océanos especulares y capa de nubes con alpha.
3. **Marte:** Shader rocoso con coloración rojiza y atmósfera tenue.
4. **Júpiter:** Shader de gigante gaseoso con bandas turbulentas y *Great Red Spot* simulada.
5. **Saturno:** Bandas de gas amarillentas y anillos con gradiente radial.
6. **Urano:** Coloración cian uniforme y anillos verticales finos.