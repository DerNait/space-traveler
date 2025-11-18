use nalgebra_glm::{Vec3, Mat4};
use minifb::{Key, Window, WindowOptions};
use std::time::Duration;
use std::f32::consts::PI;

mod framebuffer;
mod triangle;
mod line;
mod vertex;
mod obj;
mod color;
mod fragment;
mod shaders;
mod camera;

use framebuffer::Framebuffer;
use vertex::Vertex;
use obj::Obj;
use triangle::triangle;
use shaders::vertex_shader;
use camera::Camera;

pub struct Uniforms {
    pub model_matrix: Mat4,
    pub view_matrix: Mat4,
    pub projection_matrix: Mat4,
    pub screen_width: f32,
    pub screen_height: f32,
}

fn create_model_matrix(translation: Vec3, scale: f32, rotation: Vec3) -> Mat4 {
    let (sin_x, cos_x) = rotation.x.sin_cos();
    let (sin_y, cos_y) = rotation.y.sin_cos();
    let (sin_z, cos_z) = rotation.z.sin_cos();

    let rotation_matrix_x = Mat4::new(
        1.0,  0.0,    0.0,   0.0,
        0.0,  cos_x, -sin_x, 0.0,
        0.0,  sin_x,  cos_x, 0.0,
        0.0,  0.0,    0.0,   1.0,
    );

    let rotation_matrix_y = Mat4::new(
        cos_y,  0.0,  sin_y, 0.0,
        0.0,    1.0,  0.0,   0.0,
        -sin_y, 0.0,  cos_y, 0.0,
        0.0,    0.0,  0.0,   1.0,
    );

    let rotation_matrix_z = Mat4::new(
        cos_z, -sin_z, 0.0, 0.0,
        sin_z,  cos_z, 0.0, 0.0,
        0.0,    0.0,   1.0, 0.0,
        0.0,    0.0,   0.0, 1.0,
    );

    let rotation_matrix = rotation_matrix_z * rotation_matrix_y * rotation_matrix_x;

    // Transformación en espacio de mundo
    let transform_matrix = Mat4::new(
        scale, 0.0,   0.0,   translation.x,
        0.0,   scale, 0.0,   translation.y,
        0.0,   0.0,   scale, translation.z,
        0.0,   0.0,   0.0,   1.0,
    );

    transform_matrix * rotation_matrix
}

fn render(framebuffer: &mut Framebuffer, uniforms: &Uniforms, obj: &Obj) {
    let (positions, normals, uvs) = obj.mesh_buffers();

    let mut all_fragments = Vec::new();

    obj.for_each_face(|i0, i1, i2| {
        let p0 = positions[i0];
        let p1 = positions[i1];
        let p2 = positions[i2];

        let n0 = normals.get(i0).cloned().unwrap_or(nalgebra_glm::Vec3::new(0.0, 1.0, 0.0));
        let n1 = normals.get(i1).cloned().unwrap_or(nalgebra_glm::Vec3::new(0.0, 1.0, 0.0));
        let n2 = normals.get(i2).cloned().unwrap_or(nalgebra_glm::Vec3::new(0.0, 1.0, 0.0));

        let t0 = uvs.get(i0).cloned().unwrap_or(nalgebra_glm::Vec2::new(0.0, 0.0));
        let t1 = uvs.get(i1).cloned().unwrap_or(nalgebra_glm::Vec2::new(0.0, 0.0));
        let t2 = uvs.get(i2).cloned().unwrap_or(nalgebra_glm::Vec2::new(0.0, 0.0));

        let v0 = vertex_shader(&Vertex::new(p0, n0, t0), uniforms);
        let v1 = vertex_shader(&Vertex::new(p1, n1, t1), uniforms);
        let v2 = vertex_shader(&Vertex::new(p2, n2, t2), uniforms);

        all_fragments.extend(triangle(&v0, &v1, &v2));
    });

    for fragment in all_fragments {
        let x = fragment.position.x as usize;
        let y = fragment.position.y as usize;
        if x < framebuffer.width && y < framebuffer.height {
            framebuffer.set_current_color(fragment.color.to_hex());
            framebuffer.point(x, y, fragment.depth);
        }
    }
}

fn main() {
    let window_width = 800;
    let window_height = 600;
    let framebuffer_width = 800;
    let framebuffer_height = 600;
    let frame_delay = Duration::from_millis(16);

    let mut framebuffer = Framebuffer::new(framebuffer_width, framebuffer_height);
    let mut window = Window::new(
        "Nave 3D - Mundo",
        window_width,
        window_height,
        WindowOptions::default(),
    )
    .unwrap();

    window.set_position(200, 200);
    window.update();

    framebuffer.set_background_color(0x101020);

    let obj = Obj::load("assets/models/Pelican.obj").expect("Failed to load obj");

    // ===== Normalización de tamaño para la nave =====
    let (min_v, max_v) = obj.bounds();
    let size = max_v - min_v;
    let center = (min_v + max_v) * 0.5;

    let max_extent = size.x.abs().max(size.y.abs()).max(size.z.abs());
    let mut ship_scale = if max_extent > 0.0 { 5.0 / max_extent } else { 1.0 };

    // Offset para que el modelo quede centrado alrededor del origen
    let model_offset = -center * ship_scale;

    // Nave en el mundo (posición y rotación en espacio de mundo)
    let mut ship_position = Vec3::new(0.0, 0.0, 0.0);
    let mut ship_rotation = Vec3::new(0.0, 0.0, 0.0);

        // Cámara completamente fija en el mundo, NO sigue a la nave
    let aspect = framebuffer_width as f32 / framebuffer_height as f32;
    let camera = Camera::new(
        Vec3::new(0.0, 25.0, 60.0), // posición fija de la cámara en el mundo
        Vec3::new(0.0, 0.0, 0.0),   // mira SIEMPRE al origen del mundo
        aspect,
    );


    while window.is_open() {
        if window.is_key_down(Key::Escape) {
            break;
        }

        handle_input(&window, &mut ship_position, &mut ship_rotation, &mut ship_scale);

        framebuffer.clear();

        // Matriz de modelo: offset para centrar + posición de la nave en el mundo
        let model_translation = ship_position + model_offset;
        let model_matrix = create_model_matrix(model_translation, ship_scale, ship_rotation);

        let view_matrix = camera.view_matrix();
        let projection_matrix = camera.projection_matrix();

        let uniforms = Uniforms {
            model_matrix,
            view_matrix,
            projection_matrix,
            screen_width: framebuffer_width as f32,
            screen_height: framebuffer_height as f32,
        };

        framebuffer.set_current_color(0xFFDDDD);
        render(&mut framebuffer, &uniforms, &obj);

        window
            .update_with_buffer(&framebuffer.buffer, framebuffer_width, framebuffer_height)
            .unwrap();

        std::thread::sleep(frame_delay);
    }
}

fn handle_input(
    window: &Window,
    ship_position: &mut Vec3,
    ship_rotation: &mut Vec3,
    ship_scale: &mut f32,
) {
    let rot_speed = PI / 90.0;   // velocidad de giro (yaw)
    let move_speed = 0.5;        // velocidad de movimiento hacia adelante/atrás

    // ======== ROTACIÓN LOCAL (YAW) ========
    // A = girar a la izquierda, D = girar a la derecha
    if window.is_key_down(Key::A) {
        ship_rotation.y -= rot_speed;
    }
    if window.is_key_down(Key::D) {
        ship_rotation.y += rot_speed;
    }

    // ======== ORIENTACIÓN COMPLETA DE LA NAVE ========
    // Usamos la MISMA matriz que para el modelo, pero sin traslación ni escala rara.
    // Esto nos da la base (X,Y,Z) local en coordenadas de mundo.
    let orientation = create_model_matrix(Vec3::new(0.0, 0.0, 0.0), 1.0, *ship_rotation);

    // Tercera columna = eje Z local en espacio global
    let mut forward = Vec3::new(
        orientation[(0, 2)],
        orientation[(1, 2)],
        orientation[(2, 2)],
    );

    // Queremos movimiento tipo "barco": solo en el plano XZ
    forward.y = 0.0;
    if forward.magnitude() > 1e-6 {
        forward = forward.normalize();
    }

    // ======== MOVIMIENTO LOCAL Z (FORWARD/BACK) ========
    if window.is_key_down(Key::W) {
        *ship_position += forward * move_speed;
    }
    if window.is_key_down(Key::S) {
        *ship_position -= forward * move_speed;
    }

    // (Opcional) Escala de la nave
    if window.is_key_down(Key::Z) {
        *ship_scale *= 1.02;
    }
    if window.is_key_down(Key::X) {
        *ship_scale *= 0.98;
    }
}
