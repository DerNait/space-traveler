// triangle.rs
use nalgebra_glm::{Vec3, Vec2, dot};
use crate::vertex::Vertex;
use crate::shaders::{FragmentShader, Uniforms, FragAttrs};
use crate::framebuffer::Framebuffer;

// Función edge básica para calcular el área total (usada una vez)
fn edge_function(a: &Vec3, b: &Vec3, c: &Vec3) -> f32 {
    (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)
}

pub fn triangle_with_shader(
    v1: &Vertex,
    v2: &Vertex,
    v3: &Vertex,
    shader: &dyn FragmentShader,
    uniforms: &Uniforms,
    framebuffer: &mut Framebuffer
) {
    let (p1, p2, p3) = (v1.transformed_position, v2.transformed_position, v3.transformed_position);

    // 1. Backface Culling y Área
    let area = edge_function(&p1, &p2, &p3);
    if area.abs() < 1e-6 { return; }
    let inv_area = 1.0 / area;

    // 2. Bounding Box
    let min_x = p1.x.min(p2.x).min(p3.x).floor().max(0.0) as usize;
    let min_y = p1.y.min(p2.y).min(p3.y).floor().max(0.0) as usize;
    let max_x = p1.x.max(p2.x).max(p3.x).ceil().min(framebuffer.width as f32) as usize;
    let max_y = p1.y.max(p2.y).max(p3.y).ceil().min(framebuffer.height as f32) as usize;

    if min_x >= max_x || min_y >= max_y { return; }

    // 3. Pre-cálculo de constantes geométricas
    // Edge 1 (v2 -> v3): Para calcular w1
    let v2v3_x = p3.x - p2.x;
    let v2v3_y = p3.y - p2.y;

    // Edge 2 (v3 -> v1): Para calcular w2
    let v3v1_x = p1.x - p3.x;
    let v3v1_y = p1.y - p3.y;

    // Edge 3 (v1 -> v2): Para calcular w3
    let v1v2_x = p2.x - p1.x;
    let v1v2_y = p2.y - p1.y;

    // Extraemos datos para interpolación (evita acceso a memoria en el inner loop)
    let (pos1, pos2, pos3) = (v1.position, v2.position, v3.position);
    let (norm1, norm2, norm3) = (v1.transformed_normal, v2.transformed_normal, v3.transformed_normal);
    let (uv1, uv2, uv3) = (v1.tex_coords, v2.tex_coords, v3.tex_coords);
    let (z1, z2, z3) = (p1.z, p2.z, p3.z);

    // 4. Rasterización Factorizada
    for y in min_y..max_y {
        let py = y as f32 + 0.5;

        // Pre-calculamos la parte de la fórmula que depende solo de Y
        // Fórmula original: (p.x - a.x)*(b.y - a.y) - (p.y - a.y)*(b.x - a.x)
        // Factorizada:      (p.x - a.x)*DY          - (py  - a.y)*DX
        
        // Parte Y para w1 (usando p2 como ancla)
        let w1_base_y = -(py - p2.y) * v2v3_x;
        // Parte Y para w2 (usando p3 como ancla)
        let w2_base_y = -(py - p3.y) * v3v1_x;
        // Parte Y para w3 (usando p1 como ancla)
        let w3_base_y = -(py - p1.y) * v1v2_x;

        for x in min_x..max_x {
            let px = x as f32 + 0.5;

            // Calculamos W sumando la parte X (que varía) a la base Y (constante en la fila)
            // Mantenemos la resta (px - ancla.x) para preservar precisión local
            let w1 = (px - p2.x) * v2v3_y + w1_base_y;
            let w2 = (px - p3.x) * v3v1_y + w2_base_y;
            let w3 = (px - p1.x) * v1v2_y + w3_base_y;

            // Chequeo con w sin normalizar (mucho más robusto numéricamente)
            // Equivalente a w1*inv_area >= 0, pero nos ahorramos el inv_area por ahora
            // Multiplicamos por el signo del área para manejar backface culling implícito si fuera necesario
            // Pero como ya hicimos culling arriba, basta con verificar si tienen el mismo signo que el área
            // O simplemente >= 0 si asumimos orden correcto y área positiva.
            
            // Usamos la lógica de tu código "lento" pero con las variables optimizadas:
            // w1, w2, w3 aquí son el DOLE del área (no normalizados).
            // Para verificar si está dentro, deben ser todos positivos (o todos negativos si el winding es al revés).
            // Como normalizaremos multiplicando por inv_area, basta comprobar:
            
            let bc1 = w1 * inv_area;
            let bc2 = w2 * inv_area;
            let bc3 = w3 * inv_area;

            if bc1 >= 0.0 && bc2 >= 0.0 && bc3 >= 0.0 {
                
                // --- EARLY Z-CHECK (La optimización más importante para FPS) ---
                // Calculamos profundidad ANTES de interpolar normales o llamar al shader
                let depth = z1 * bc1 + z2 * bc2 + z3 * bc3;
                let index = y * framebuffer.width + x;

                // Usamos acceso directo al vector (más rápido que métodos get/set con bounds check)
                // Safety: min_x/max_x garantizan que estamos en rango
                if depth < framebuffer.zbuffer[index] {
                    
                    // Ahora sí, hacemos el trabajo pesado
                    let normal = (norm1 * bc1 + norm2 * bc2 + norm3 * bc3).normalize();
                    let obj_pos = pos1 * bc1 + pos2 * bc2 + pos3 * bc3;
                    let uv = uv1 * bc1 + uv2 * bc2 + uv3 * bc3;

                    let attrs = FragAttrs { obj_pos, normal, uv, depth };
                    
                    let (color, alpha) = shader.shade(&attrs, uniforms);

                    // Dibujado
                    if alpha >= 1.0 {
                        framebuffer.buffer[index] = color.to_hex();
                        framebuffer.zbuffer[index] = depth;
                    } else {
                        // Blending simple
                        let bg = framebuffer.buffer[index];
                        let bg_r = ((bg >> 16) & 0xFF) as f32;
                        let bg_g = ((bg >> 8) & 0xFF) as f32;
                        let bg_b = (bg & 0xFF) as f32;

                        let r = (color.r as f32 * alpha + bg_r * (1.0 - alpha)) as u32;
                        let g = (color.g as f32 * alpha + bg_g * (1.0 - alpha)) as u32;
                        let b = (color.b as f32 * alpha + bg_b * (1.0 - alpha)) as u32;

                        framebuffer.buffer[index] = (r << 16) | (g << 8) | b;
                        framebuffer.zbuffer[index] = depth; // Escribir Z en transparencia es debatible, pero lo mantenemos
                    }
                }
            }
        }
    }
}