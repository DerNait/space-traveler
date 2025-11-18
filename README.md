# Lab 5: Sistema Solar con Shaders Procedurales 🌌

Mini renderizador 3D en software que muestra un **sistema solar completo** con el sol en el centro y 5 planetas orbitando a diferentes velocidades, cada uno usando **shaders procedurales** únicos (Perlin, Voronoi, BandedGas, Flow, etc.) sobre modelos OBJ.

## ✨ Características principales

- **Sol central** con shader de "lava" animada usando flowmaps
- **5 planetas orbitando** alrededor del sol con diferentes velocidades según su distancia:
  - 🌍 **Tierra** - océanos, continentes y nubes animadas (órbita más cercana, rápida)
  - 🪐 **Júpiter** - gigante gaseoso con bandas turbulentas
  - 🔴 **Marte** - planeta rocoso rojo con tormentas de polvo
  - 💙 **Urano** - tonos azulados pastel con anillos finos
  - 🟡 **Saturno** - bandas amarillas con anillos prominentes (órbita más lejana, lenta)
- **Translación realista**: planetas más lejanos se mueven más lento
- **Rotación individual** de cada planeta sobre su propio eje
- **Lunas orbitando** la Tierra con shaders procedurales propios
- Sombreado procedural para cada cuerpo celeste
- Z-buffer, iluminación difusa y capas con alpha para nubes/anillos
- Anillos con diferentes estilos para Saturno y Urano

---

## 🎥 Video de demostración

[![Demo - Lab 5 Shaders en Planetas](https://img.youtube.com/vi/8V3RQKlX4dk/0.jpg)](https://www.youtube.com/watch?v=8V3RQKlX4dk)

---

## 📸 Capturas

![Render](captura%201.png)
![Render](captura%202.png)
![Render](captura%203.png)
![Render](captura%204.png)
![Render](captura%205.png)
![Render](captura%206.png)

---

## 🎮 Controles

### Cámara y navegación
- **Flechas**: mover la cámara en X/Y
- **A / S**: alejar / acercar zoom (movimiento en profundidad)
- **Q / W**: rotar cámara en eje **X** (pitch)
- **E / R**: rotar cámara en eje **Y** (yaw)
- **T / Y**: rotar cámara en eje **Z** (roll)

### Visualización
- **Z**: activar/desactivar anillos de Saturno y Urano
- **X**: activar/desactivar lunas de la Tierra

---

## 🌍 Planetas del sistema

1. **Sol** ☀️ - Centro del sistema, autoiluminado con efecto de lava
2. **Tierra** 🌍 - Órbita: 150px, velocidad: 0.15 rad/s (con 2 lunas)
3. **Marte** 🔴 - Órbita: 180px, velocidad: 0.12 rad/s
4. **Júpiter** 🪐 - Órbita: 220px, velocidad: 0.08 rad/s
5. **Urano** 💙 - Órbita: 260px, velocidad: 0.06 rad/s (con anillos verticales)
6. **Saturno** 🟡 - Órbita: 300px, velocidad: 0.05 rad/s (con anillos icónicos)

---

## 🛠 Detalles técnicos

- Rasterización por triángulos en CPU con **z-buffer**
- **Sistema solar dinámico** con órbitas circulares y velocidades variables
- Shaders procedurales basados en:
  - Ruido Perlin / Value / Voronoi
  - Shaders tipo **BandedGas** para planetas gaseosos
  - Flow maps para animar bandas y "lava"
  - Gradientes radiales para anillos
- Iluminación difusa simple con vector de luz configurable
- Soporte de múltiples capas con alpha (nubes, atmósferas, anillos)
- Rotación independiente de cada planeta
- Translación orbital con velocidad proporcional a la distancia
- 2 lunas orbitando la Tierra con texturas procedurales distintas
