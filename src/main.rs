use nalgebra_glm::{Vec3, Mat4};
use minifb::{Key, Window, WindowOptions};
use std::time::{Duration, Instant};
use std::f32::consts::PI;

mod framebuffer;
mod triangle;
mod line;
mod vertex;
mod obj;
mod color;
mod fragment;
mod shaders;
mod noise;
mod scene;

use framebuffer::Framebuffer;
use vertex::Vertex;
use obj::Obj;
use triangle::triangle_with_shader;
use shaders::{vertex_shader, Uniforms, FragmentShader};
use scene::create_solar_system;

// ===================== Cámara simple (ortográfica sobre tu "espacio pantalla") =====================

struct Camera {
    position: Vec3,   // traslación en tu espacio actual (px)
    rotation: Vec3,   // pitch (x), yaw (y), roll (z)
}

impl Camera {
    fn new() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 0.0),
            rotation: Vec3::new(0.0, 0.0, 0.0),
        }
    }

    fn view_matrix(&self) -> Mat4 {
        let (sx, cx) = self.rotation.x.sin_cos();
        let (sy, cy) = self.rotation.y.sin_cos();
        let (sz, cz) = self.rotation.z.sin_cos();

        let rx = Mat4::new(
            1.0, 0.0,  0.0, 0.0,
            0.0, cx,  -sx, 0.0,
            0.0, sx,   cx, 0.0,
            0.0, 0.0,  0.0, 1.0,
        );
        let ry = Mat4::new(
             cy, 0.0, sy, 0.0,
             0.0, 1.0, 0.0, 0.0,
            -sy, 0.0, cy, 0.0,
             0.0, 0.0, 0.0, 1.0,
        );
        let rz = Mat4::new(
            cz, -sz, 0.0, 0.0,
            sz,  cz, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );

        // vista = R^T * T(-pos)  (inversa de la pose de cámara)
        let r = rz * ry * rx;
        let t = Mat4::new(
            1.0, 0.0, 0.0, -self.position.x,
            0.0, 1.0, 0.0, -self.position.y,
            0.0, 0.0, 1.0, -self.position.z,
            0.0, 0.0, 0.0,  1.0,
        );

        r.transpose() * t
    }
}

// ===================== Helpers de modelo =====================

pub struct ModelMatrices { pub base: Mat4, pub overlay: Mat4 }

pub fn create_model_matrix(translation: Vec3, scale: f32, rotation: Vec3) -> Mat4 {
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
        0.0,    0.0,  1.0, 0.0,
        0.0,    0.0,  0.0, 1.0,
    );

    let rotation_matrix = rotation_matrix_z * rotation_matrix_y * rotation_matrix_x;

    let transform_matrix = Mat4::new(
        scale, 0.0,   0.0,   translation.x,
        0.0,   scale, 0.0,   translation.y,
        0.0,   0.0,   scale, translation.z,
        0.0,   0.0,   0.0,   1.0,
    );

    transform_matrix * rotation_matrix
}

// ===================== Render passes =====================

fn render_pass(
    framebuffer: &mut Framebuffer,
    uniforms: &Uniforms,
    obj: &Obj,
    shader: &dyn FragmentShader
) {
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

        all_fragments.extend(triangle_with_shader(&v0, &v1, &v2, shader, uniforms));
    });

    for fragment in all_fragments {
        let x = fragment.position.x as usize;
        let y = fragment.position.y as usize;
        if x < framebuffer.width && y < framebuffer.height {
            framebuffer.draw_rgba(x, y, fragment.depth, fragment.color.to_hex(), 1.0);
        }
    }
}

fn render_pass_alpha(
    framebuffer: &mut Framebuffer,
    uniforms: &Uniforms,
    obj: &Obj,
    shader: &dyn FragmentShader,
    z_bias: f32,  // << sesgo de profundidad (negativo = más cerca de la cámara)
) {
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

        all_fragments.extend(crate::triangle::triangle_with_shader(&v0, &v1, &v2, shader, uniforms));
    });

    for frag in all_fragments {
        let x = frag.position.x as usize;
        let y = frag.position.y as usize;
        if x < framebuffer.width && y < framebuffer.height {
            framebuffer.draw_rgba(x, y, frag.depth + z_bias, frag.color.to_hex(), frag.alpha);
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
        "Sistema Epsilon Eridani - Mini Renderizador 3D",
        window_width,
        window_height,
        WindowOptions::default(),
    ).unwrap();

    window.set_position(500, 500);
    window.update();

    framebuffer.set_background_color(0x000000);

    // ===== Carga de modelos =====
    let obj = Obj::load("assets/models/Planet.obj").expect("Failed to load obj");
    let rings_obj = Obj::load("assets/models/PlanetRing.obj").expect("Failed to load rings obj");
    let moon_obj = Obj::load("assets/models/Planet.obj").expect("Failed to load moon sphere");

    // ===== Ajuste de escala base para los planetas =====
    let (min_v, max_v) = obj.bounds();
    let size = max_v - min_v;

    let target_w = framebuffer_width as f32 * 0.8;
    let target_h = framebuffer_height as f32 * 0.8;
    let sx = if size.x.abs() < 1e-6 { 1.0 } else { target_w / size.x.abs() };
    let sy = if size.y.abs() < 1e-6 { 1.0 } else { target_h / size.y.abs() };
    let base_scale = (sx.min(sy)) * 0.12; // escala más pequeña para cada planeta

    // Posición del sol (centro de la pantalla)
    let sun_translation = Vec3::new(
        (framebuffer_width as f32) * 0.5,
        (framebuffer_height as f32) * 0.5,
        0.0
    );

    // ===== Cámara =====
    let mut camera = Camera::new();

    // ===== Crear el sistema solar completo desde scene.rs =====
    let solar_system = create_solar_system(obj.clone(), moon_obj.clone());

    // ===== Medimos los semiejes del anillo (en espacio OBJ) =====
    let (rmin, rmax) = rings_obj.bounds();
    let ext = rmax - rmin;

    let (ring_a, ring_b, ring_plane_xy) = {
        let ex = ext.x.abs();
        let ey = ext.y.abs();
        let ez = ext.z.abs();

        if ez <= ex.min(ey) {
            let a = ((rmax.x - rmin.x).abs() * 0.5).max(1e-6);
            let b = ((rmax.y - rmin.y).abs() * 0.5).max(1e-6);
            (a, b, true)
        } else if ey <= ex.min(ez) {
            let a = ((rmax.x - rmin.x).abs() * 0.5).max(1e-6);
            let b = ((rmax.z - rmin.z).abs() * 0.5).max(1e-6);
            (a, b, false)
        } else {
            let a = ((rmax.y - rmin.y).abs() * 0.5).max(1e-6);
            let b = ((rmax.z - rmin.z).abs() * 0.5).max(1e-6);
            (a, b, true)
        }
    };

    // ===== Estado de visualización =====
    let time_origin = Instant::now();

    let mut show_rings = true;
    let mut show_moons = true;

    let mut prev_z = false;
    let mut prev_x = false;

    while window.is_open() {
        if window.is_key_down(Key::Escape) { break; }

        handle_input_camera(&window, &mut camera);

        // ==== Toggles con Z (anillos), X (lunas) ====
        let z_down = window.is_key_down(Key::Z);
        if z_down && !prev_z {
            show_rings = !show_rings;
        }
        prev_z = z_down;

        let x_down = window.is_key_down(Key::X);
        if x_down && !prev_x {
            show_moons = !show_moons;
        }
        prev_x = x_down;

        framebuffer.clear();

        let elapsed = time_origin.elapsed().as_secs_f32();

        // ===== SOL (estrella en el centro) =====
        let sun_scale = base_scale * 2.0;
        let sun_rotation = Vec3::new(0.0, elapsed * 0.15, 0.0);
        let sun_model = create_model_matrix(sun_translation, sun_scale, sun_rotation);
        let sun_uniforms = Uniforms {
            model_matrix: sun_model,
            view_matrix: camera.view_matrix(),
            time: elapsed,
            seed: 6666,
            ring_a, ring_b, ring_plane_xy,
        };
        render_pass(&mut framebuffer, &sun_uniforms, &obj, &solar_system.sun_shader);

        // ===== PLANETAS orbitando el sol =====
        for (idx, planet) in solar_system.planets.iter().enumerate() {
            let planet_model = planet.model_matrix(sun_translation, elapsed);
            let planet_uniforms = Uniforms {
                model_matrix: planet_model,
                view_matrix: camera.view_matrix(),
                time: elapsed,
                seed: planet.seed,
                ring_a, ring_b, ring_plane_xy,
            };

            // Renderizar superficie del planeta
            render_pass(&mut framebuffer, &planet_uniforms, &planet.obj, &*planet.shader);

            // Si tiene nubes, renderizarlas como overlay
            if let Some(ref cloud_shader) = planet.cloud_shader {
                let overlay_scale = planet.scale * 1.02;
                let planet_translation = planet.translation(sun_translation, elapsed);
                let rotation = Vec3::new(0.0, elapsed * planet.rotation_speed, 0.0);
                let cloud_model = create_model_matrix(planet_translation, overlay_scale, rotation);
                let cloud_uniforms = Uniforms {
                    model_matrix: cloud_model,
                    view_matrix: camera.view_matrix(),
                    time: elapsed,
                    seed: planet.seed + 1000,
                    ring_a, ring_b, ring_plane_xy,
                };
                render_pass_alpha(&mut framebuffer, &cloud_uniforms, &planet.obj, cloud_shader.as_ref(), 0.0);
            }

            // Anillos para planetas con anillos
            if show_rings {
                // Urano (índice 3)
                if idx == 3 {
                    let planet_translation = planet.translation(sun_translation, elapsed);
                    let rings_tilt = Vec3::new(1.35, elapsed * planet.rotation_speed, 0.0);
                    let rings_scale = planet.scale * 1.05;
                    let rings_matrix = create_model_matrix(planet_translation, rings_scale, rings_tilt);
                    let rings_uniforms = Uniforms {
                        model_matrix: rings_matrix,
                        view_matrix: camera.view_matrix(),
                        time: elapsed,
                        seed: planet.seed + 2000,
                        ring_a, ring_b, ring_plane_xy,
                    };
                    render_pass(&mut framebuffer, &rings_uniforms, &rings_obj, &solar_system.uranus_rings_shader);
                }
                // Saturno (índice 4)
                else if idx == 4 {
                    let planet_translation = planet.translation(sun_translation, elapsed);
                    let rings_tilt = Vec3::new(0.35, elapsed * planet.rotation_speed, 0.05);
                    let rings_scale = planet.scale * 1.2;
                    let rings_matrix = create_model_matrix(planet_translation, rings_scale, rings_tilt);
                    let rings_uniforms = Uniforms {
                        model_matrix: rings_matrix,
                        view_matrix: camera.view_matrix(),
                        time: elapsed,
                        seed: planet.seed + 2000,
                        ring_a, ring_b, ring_plane_xy,
                    };
                    render_pass(&mut framebuffer, &rings_uniforms, &rings_obj, &solar_system.saturn_rings_shader);
                }
            }

            // Renderizar lunas de cada planeta
            if show_moons {
                let planet_translation = planet.translation(sun_translation, elapsed);
                for m in &planet.moons {
                    let mm = m.model_matrix(planet_translation, planet.scale, elapsed);
                    let mu = Uniforms {
                        model_matrix: mm,
                        view_matrix: camera.view_matrix(),
                        time: elapsed,
                        seed: m.seed,
                        ring_a, ring_b, ring_plane_xy,
                    };
                    render_pass(&mut framebuffer, &mu, &m.obj, &*m.shader);
                }
            }
        }

        window
            .update_with_buffer(&framebuffer.buffer, framebuffer_width, framebuffer_height)
            .unwrap();

        std::thread::sleep(frame_delay);
    }
}

// ===================== Input de cámara =====================

fn handle_input_camera(window: &Window, camera: &mut Camera) {
    // mover cámara
    if window.is_key_down(Key::Right) { camera.position.x += 10.0; }
    if window.is_key_down(Key::Left)  { camera.position.x -= 10.0; }
    if window.is_key_down(Key::Up)    { camera.position.y -= 10.0; }
    if window.is_key_down(Key::Down)  { camera.position.y += 10.0; }

    // zoom con Z en profundidad
    if window.is_key_down(Key::S)     { camera.position.z += 5.0; }
    if window.is_key_down(Key::A)     { camera.position.z -= 5.0; }

    // rotar cámara
    if window.is_key_down(Key::Q) { camera.rotation.x -= PI / 20.0; }
    if window.is_key_down(Key::W) { camera.rotation.x += PI / 20.0; }
    if window.is_key_down(Key::E) { camera.rotation.y -= PI / 20.0; }
    if window.is_key_down(Key::R) { camera.rotation.y += PI / 20.0; }
    if window.is_key_down(Key::T) { camera.rotation.z -= PI / 20.0; }
    if window.is_key_down(Key::Y) { camera.rotation.z += PI / 20.0; }
}
