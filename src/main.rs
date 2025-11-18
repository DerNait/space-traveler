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

/// Render ahora:
/// - llama al vertex shader que puede devolver None (culling)
/// - rasteriza triángulo por triángulo, sin acumular todos los fragments en un mega Vec
fn render(framebuffer: &mut Framebuffer, uniforms: &Uniforms, obj: &Obj) {
    let (positions, normals, uvs) = obj.mesh_buffers();

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

        let v0_opt = vertex_shader(&Vertex::new(p0, n0, t0), uniforms);
        let v1_opt = vertex_shader(&Vertex::new(p1, n1, t1), uniforms);
        let v2_opt = vertex_shader(&Vertex::new(p2, n2, t2), uniforms);

        if let (Some(v0), Some(v1), Some(v2)) = (v0_opt, v1_opt, v2_opt) {
            let screen_w = framebuffer.width as i32;
            let screen_h = framebuffer.height as i32;

            for fragment in triangle(&v0, &v1, &v2, screen_w, screen_h) {
                let x_i32 = fragment.position.x as i32;
                let y_i32 = fragment.position.y as i32;

                if x_i32 >= 0
                    && y_i32 >= 0
                    && x_i32 < framebuffer.width as i32
                    && y_i32 < framebuffer.height as i32
                {
                    let x = x_i32 as usize;
                    let y = y_i32 as usize;
                    framebuffer.point(x, y, fragment.depth, fragment.color.to_hex());
                }
            }
        }
    });
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

    // ===== Modelos =====
    let ship_obj = Obj::load("assets/models/Pelican.obj").expect("Failed to load ship obj");
    let planet_obj = Obj::load("assets/models/Planet.obj").expect("Failed to load planet obj");

    // ===== Normalización de tamaño para la nave =====
    let (min_v, max_v) = ship_obj.bounds();
    let size = max_v - min_v;
    let center = (min_v + max_v) * 0.5;

    let max_extent = size.x.abs().max(size.y.abs()).max(size.z.abs());
    let mut ship_scale = if max_extent > 0.0 { 5.0 / max_extent } else { 1.0 };

    // Offset para que el modelo quede centrado alrededor del origen
    let model_offset = -center * ship_scale;

    // ===== Normalización de tamaño para el planeta =====
    let (p_min, p_max) = planet_obj.bounds();
    let p_size = p_max - p_min;
    let p_center = (p_min + p_max) * 0.5;
    let p_max_extent = p_size.x.abs().max(p_size.y.abs()).max(p_size.z.abs());
    let planet_scale = if p_max_extent > 0.0 { 8.0 / p_max_extent } else { 1.0 };
    let planet_model_offset = -p_center * planet_scale;

    // ===== Posiciones de planetas en el mundo =====
    // Todos alejados de la nave (0,0,0) y al frente (z < 0)
    let planet_positions: Vec<Vec3> = vec![
        Vec3::new(   0.0,   0.0,  -180.0),
        Vec3::new( 120.0,  30.0,  -260.0),
        Vec3::new(-150.0, -20.0,  -320.0),
        Vec3::new(  60.0, -40.0,  -420.0),
        Vec3::new(-220.0,  10.0,  -520.0),
        Vec3::new( 180.0,  25.0,  -650.0),
    ];

    // ===== Nave en el mundo =====
    let mut ship_position = Vec3::new(0.0, 0.0, 0.0);

    // Modelo girado 180° en Z para verse bien
    let mut ship_rotation = Vec3::new(0.0, 0.0, PI);

    // ===== Cámara =====
    let aspect = framebuffer_width as f32 / framebuffer_height as f32;
    let mut camera = Camera::new(
        Vec3::new(0.0, 15.0, 60.0),
        ship_position,
        aspect,
    );

    // Parámetros base de cámara detrás de la nave
    let cam_back_dist: f32 = 30.0;
    let cam_height: f32 = 12.0;
    let cam_radius: f32 = (cam_back_dist * cam_back_dist + cam_height * cam_height).sqrt();

    // Offsets de órbita (sobre la posición "detrás" base)
    let mut orbit_yaw: f32 = 0.0;
    let mut orbit_pitch: f32 = 0.0;
    let orbit_speed: f32 = PI / 180.0 * 2.0; // 2° por frame

    let max_orbit_pitch: f32 = PI / 3.0;
    let min_orbit_pitch: f32 = -PI / 6.0;

    while window.is_open() {
        if window.is_key_down(Key::Escape) {
            break;
        }

        // Movimiento/rotación de la nave
        handle_input(&window, &mut ship_position, &mut ship_rotation, &mut ship_scale);

        // ===== Recalcular forward de la nave (en plano XZ) =====
        let orientation = create_model_matrix(
            Vec3::new(0.0, 0.0, 0.0),
            1.0,
            ship_rotation,
        );

        let mut forward = Vec3::new(
            orientation[(0, 2)],
            orientation[(1, 2)],
            orientation[(2, 2)],
        );
        forward.y = 0.0;
        if forward.magnitude() > 1e-6 {
            forward = forward.normalize();
        }

        // ===== Controles de órbita IJKL =====
        if window.is_key_down(Key::J) {
            orbit_yaw -= orbit_speed;
        }
        if window.is_key_down(Key::L) {
            orbit_yaw += orbit_speed;
        }
        if window.is_key_down(Key::I) {
            orbit_pitch += orbit_speed;
        }
        if window.is_key_down(Key::K) {
            orbit_pitch -= orbit_speed;
        }

        if orbit_pitch > max_orbit_pitch { orbit_pitch = max_orbit_pitch; }
        if orbit_pitch < min_orbit_pitch { orbit_pitch = min_orbit_pitch; }

        // SHIFT = reset a posición “detrás de la nave”
        if window.is_key_down(Key::LeftShift) || window.is_key_down(Key::RightShift) {
            orbit_yaw = 0.0;
            orbit_pitch = 0.0;
        }

        // ===== Base: cámara detrás de la nave =====
        let base_offset = -forward * cam_back_dist + Vec3::new(0.0, cam_height, 0.0);

        // Dirección base normalizada
        let base_dir = base_offset / cam_radius;
        let base_pitch = base_dir.y.asin();
        let base_yaw = base_dir.x.atan2(base_dir.z);

        // Aplicar offsets de órbita
        let yaw = base_yaw + orbit_yaw;
        let pitch = (base_pitch + orbit_pitch)
            .clamp(-PI * 0.49, PI * 0.49); // evita voltear la cámara por completo

        let cos_p = pitch.cos();
        let sin_p = pitch.sin();
        let sin_y = yaw.sin();
        let cos_y = yaw.cos();

        let final_offset = Vec3::new(
            cam_radius * sin_y * cos_p,
            cam_radius * sin_p,
            cam_radius * cos_y * cos_p,
        );

        camera.position = ship_position + final_offset;
        camera.target = ship_position;

        framebuffer.clear();

        // ===== Matrices de vista/proyección =====
        let view_matrix = camera.view_matrix();
        let projection_matrix = camera.projection_matrix();

        // ===== Render nave =====
        let model_translation = ship_position + model_offset;
        let ship_model_matrix = create_model_matrix(model_translation, ship_scale, ship_rotation);

        let ship_uniforms = Uniforms {
            model_matrix: ship_model_matrix,
            view_matrix,
            projection_matrix,
            screen_width: framebuffer_width as f32,
            screen_height: framebuffer_height as f32,
        };

        framebuffer.set_current_color(0xFFDDDD);
        render(&mut framebuffer, &ship_uniforms, &ship_obj);

        // ===== Render planetas (están en mundo, no pegados a la cámara) =====
        for planet_pos in &planet_positions {
            let planet_translation = *planet_pos + planet_model_offset;
            let planet_model_matrix = create_model_matrix(
                planet_translation,
                planet_scale,
                Vec3::new(0.0, 0.0, 0.0),
            );

            let planet_uniforms = Uniforms {
                model_matrix: planet_model_matrix,
                view_matrix,
                projection_matrix,
                screen_width: framebuffer_width as f32,
                screen_height: framebuffer_height as f32,
            };

            render(&mut framebuffer, &planet_uniforms, &planet_obj);
        }

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
    if window.is_key_down(Key::A) {
        ship_rotation.y -= rot_speed;
    }
    if window.is_key_down(Key::D) {
        ship_rotation.y += rot_speed;
    }

    // ======== ORIENTACIÓN COMPLETA DE LA NAVE ========
    let orientation = create_model_matrix(Vec3::new(0.0, 0.0, 0.0), 1.0, *ship_rotation);

    // Tercera columna = eje Z local en espacio global
    let mut forward = Vec3::new(
        orientation[(0, 2)],
        orientation[(1, 2)],
        orientation[(2, 2)],
    );

    // Movimiento tipo "barco": solo en el plano XZ
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

    // Escala opcional de la nave
    if window.is_key_down(Key::Z) {
        *ship_scale *= 1.02;
    }
    if window.is_key_down(Key::X) {
        *ship_scale *= 0.98;
    }
}
