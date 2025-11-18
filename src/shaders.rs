use nalgebra_glm::{Vec3, Vec4, Mat3};
use crate::vertex::Vertex;
use crate::Uniforms;

pub fn vertex_shader(vertex: &Vertex, uniforms: &Uniforms) -> Vertex {
    // Posición original del vértice
    let position = Vec4::new(
        vertex.position.x,
        vertex.position.y,
        vertex.position.z,
        1.0,
    );

    // ===== Espacio de mundo =====
    let world_pos = uniforms.model_matrix * position;

    // ===== Espacio de vista (cámara) =====
    let view_pos = uniforms.view_matrix * world_pos;

    // ===== Profundidad para el z-buffer =====
    // Suponiendo que la cámara mira hacia -Z en espacio de vista,
    // usamos la distancia positiva a lo largo de ese eje.
    let depth = -view_pos.z;

    // ===== Proyección en clip space =====
    let clip_pos = uniforms.projection_matrix * view_pos;

    let w = clip_pos.w;
    let ndc_x = clip_pos.x / w;
    let ndc_y = clip_pos.y / w;
    // ndc_z se podría usar si quieres algo más “canónico”, pero aquí usamos depth.

    // ===== Viewport transform: NDC [-1,1] -> píxeles =====
    let half_w = uniforms.screen_width * 0.5;
    let half_h = uniforms.screen_height * 0.5;

    // x: -1 -> 0, 1 -> width
    let screen_x = ndc_x * half_w + half_w;
    // y: -1 -> height, 1 -> 0 (invertimos eje Y de pantalla)
    let screen_y = -ndc_y * half_h + half_h;

    // ===== Normal transform =====
    let model_mat3 = Mat3::new(
        uniforms.model_matrix[(0, 0)], uniforms.model_matrix[(0, 1)], uniforms.model_matrix[(0, 2)],
        uniforms.model_matrix[(1, 0)], uniforms.model_matrix[(1, 1)], uniforms.model_matrix[(1, 2)],
        uniforms.model_matrix[(2, 0)], uniforms.model_matrix[(2, 1)], uniforms.model_matrix[(2, 2)],
    );

    let normal_matrix = model_mat3
        .transpose()
        .try_inverse()
        .unwrap_or(Mat3::identity());

    let transformed_normal = normal_matrix * vertex.normal;

    Vertex {
        position: vertex.position,
        normal: vertex.normal,
        tex_coords: vertex.tex_coords,
        color: vertex.color,
        transformed_position: Vec3::new(screen_x, screen_y, depth),
        transformed_normal,
    }
}
