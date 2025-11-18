// shaders.rs
use nalgebra_glm::{Vec3, Vec4, Mat3};
use crate::vertex::Vertex;
use crate::Uniforms;

/// Vertex shader:
/// - Usa depth = -view_pos.z (distancia en espacio de vista)
/// - Hace culling solo para vértices detrás de la cámara (z >= 0)
/// - Sin culling agresivo por NDC (dejamos que el triangle+clipping hagan su trabajo)
pub fn vertex_shader(vertex: &Vertex, uniforms: &Uniforms) -> Option<Vertex> {
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

    // Con look_at RH, lo que está frente a la cámara tiene z < 0.
    // Todo lo que queda en z >= 0 está detrás o en el ojo -> lo descartamos.
    if view_pos.z >= 0.0 {
        return None;
    }

    // ===== Proyección en clip space =====
    let clip_pos = uniforms.projection_matrix * view_pos;
    let w = clip_pos.w;

    // Evitar divisiones locas cuando w es casi 0
    if w.abs() < 1e-5 {
        return None;
    }

    // NDC (no hacemos culling, solo los usamos para pasar a pantalla)
    let ndc_x = clip_pos.x / w;
    let ndc_y = clip_pos.y / w;
    // ndc_z no lo usamos para culling ni para depth

    // ===== Viewport transform: NDC [-1,1] -> píxeles =====
    let half_w = uniforms.screen_width * 0.5;
    let half_h = uniforms.screen_height * 0.5;

    let screen_x = ndc_x * half_w + half_w;
    let screen_y = -ndc_y * half_h + half_h;

    // ===== Depth en espacio de vista =====
    // -view_pos.z es positivo y crece con la distancia, ideal para tu z-buffer
    let depth = -view_pos.z;

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

    Some(Vertex {
        position: vertex.position,
        normal: vertex.normal,
        tex_coords: vertex.tex_coords,
        color: vertex.color,
        transformed_position: Vec3::new(screen_x, screen_y, depth),
        transformed_normal,
    })
}
