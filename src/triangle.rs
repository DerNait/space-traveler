use nalgebra_glm::{Vec3, Vec2};
use crate::vertex::Vertex;
use crate::shaders::{FragmentShader, Uniforms, FragAttrs};
use crate::framebuffer::Framebuffer;

// Ahora recibe framebuffer mutable y no devuelve nada
pub fn triangle_with_shader(
    v1: &Vertex,
    v2: &Vertex,
    v3: &Vertex,
    shader: &dyn FragmentShader,
    uniforms: &Uniforms,
    framebuffer: &mut Framebuffer
) {
    let (a, b, c) = (v1.transformed_position, v2.transformed_position, v3.transformed_position);

    // Bounding box + Clipping a pantalla
    let min_x = a.x.min(b.x).min(c.x).floor().max(0.0) as usize;
    let min_y = a.y.min(b.y).min(c.y).floor().max(0.0) as usize;
    let max_x = a.x.max(b.x).max(c.x).ceil().min(uniforms.screen_width as f32) as usize;
    let max_y = a.y.max(b.y).max(c.y).ceil().min(uniforms.screen_height as f32) as usize;

    let area = edge_function(&a, &b, &c);
    if area.abs() < 1e-6 { return; }

    // Precalcular UVs y Normales y Posiciones para interpolación baricéntrica
    // (Evitamos accesos repetidos a los Vertex)
    let (pos1, pos2, pos3) = (v1.position, v2.position, v3.position);
    let (norm1, norm2, norm3) = (v1.transformed_normal, v2.transformed_normal, v3.transformed_normal);
    let (uv1, uv2, uv3) = (v1.tex_coords, v2.tex_coords, v3.tex_coords);
    let (depth1, depth2, depth3) = (a.z, b.z, c.z);

    for y in min_y..max_y {
        for x in min_x..max_x {
            let p = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, 0.0);
            let (w1, w2, w3) = barycentric_coordinates(&p, &a, &b, &c, area);

            if w1 >= 0.0 && w2 >= 0.0 && w3 >= 0.0 {
                // OPTIMIZACIÓN CRÍTICA: Early Z-Rejection
                // Calculamos Z antes de ejecutar el shader pesado
                let depth = depth1 * w1 + depth2 * w2 + depth3 * w3;
                let index = y * framebuffer.width + x;
                
                // Solo si el pixel está más cerca que lo que ya hay en el buffer, procesamos el shader
                // Nota: Si hay transparencia (alpha < 1.0), necesitamos dibujar igual para mezclar.
                // Para simplificar y ganar velocidad en planetas opacos, podemos ser estrictos.
                // Pero para nubes, necesitamos que pase.
                // Como el zbuffer guarda el objeto más cercano, si depth < zbuffer[index], es visible.
                if depth < framebuffer.zbuffer[index] {
                    
                    // Interpolamos el resto SOLO si pasa el Z-check
                    let mut normal = norm1 * w1 + norm2 * w2 + norm3 * w3;
                    normal = normal.normalize(); // Normalizar es un poco costoso, pero necesario para iluminación

                    let obj_pos = pos1 * w1 + pos2 * w2 + pos3 * w3;
                    let uv = uv1 * w1 + uv2 * w2 + uv3 * w3;

                    let attrs = FragAttrs { obj_pos, normal, uv, depth };
                    
                    // Ejecutamos el shader (aquí está el ruido Perlin costoso)
                    let (color, alpha) = shader.shade(&attrs, uniforms);

                    // Dibujamos directamente
                    framebuffer.draw_rgba(x, y, depth, color.to_hex(), alpha);
                }
            }
        }
    }
}

fn barycentric_coordinates(p: &Vec3, a: &Vec3, b: &Vec3, c: &Vec3, area: f32) -> (f32, f32, f32) {
    let w1 = edge_function(b, c, p) / area;
    let w2 = edge_function(c, a, p) / area;
    let w3 = edge_function(a, b, p) / area;
    (w1, w2, w3)
}

fn edge_function(a: &Vec3, b: &Vec3, c: &Vec3) -> f32 {
    (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)
}