// triangle.rs
use nalgebra_glm::{Vec3};
use crate::fragment::Fragment;
use crate::vertex::Vertex;
use crate::shaders::{FragmentShader, Uniforms, FragAttrs};

pub fn triangle_with_shader(
    v1: &Vertex,
    v2: &Vertex,
    v3: &Vertex,
    shader: &dyn FragmentShader,
    uniforms: &Uniforms
) -> Vec<Fragment> {
    let mut fragments = Vec::new();
    let (a, b, c) = (v1.transformed_position, v2.transformed_position, v3.transformed_position);

    // Bounding box + Clipping básico
    let min_x = a.x.min(b.x).min(c.x).floor().max(0.0) as i32;
    let min_y = a.y.min(b.y).min(c.y).floor().max(0.0) as i32;
    let max_x = a.x.max(b.x).max(c.x).ceil().min(uniforms.screen_width) as i32;
    let max_y = a.y.max(b.y).max(c.y).ceil().min(uniforms.screen_height) as i32;

    let area = edge_function(&a, &b, &c);
    if area.abs() < 1e-6 { return fragments; }

    for y in min_y..max_y {
        for x in min_x..max_x {
            let p = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, 0.0);
            let (w1, w2, w3) = barycentric_coordinates(&p, &a, &b, &c, area);

            if w1 >= 0.0 && w2 >= 0.0 && w3 >= 0.0 {
                // Interpolación
                let depth = a.z * w1 + b.z * w2 + c.z * w3;
                
                // Normal
                let mut normal = v1.transformed_normal * w1 + v2.transformed_normal * w2 + v3.transformed_normal * w3;
                normal = normal.normalize();

                // Posición Objeto (Clave para el ruido)
                let obj_pos = v1.position * w1 + v2.position * w2 + v3.position * w3;
                
                // UVs
                let uv = v1.tex_coords * w1 + v2.tex_coords * w2 + v3.tex_coords * w3;

                // Llamar al Shader
                let attrs = FragAttrs { obj_pos, normal, uv, depth };
                let (color, alpha) = shader.shade(&attrs, uniforms);

                if alpha > 0.0 {
                     fragments.push(Fragment::new(x as f32, y as f32, color, depth, alpha));
                }
            }
        }
    }
    fragments
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