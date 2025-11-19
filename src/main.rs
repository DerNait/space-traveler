use nalgebra_glm::{Vec3, Mat4, normalize};
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
mod camera;
mod noise; 
mod scene;
mod texture;

use texture::Texture; 
use framebuffer::Framebuffer;
use vertex::Vertex;
use obj::Obj;
use triangle::triangle_with_shader;
use line::line; 
use shaders::{
    vertex_shader, Uniforms, FragmentShader, ProceduralLayerShader, NoiseParams, NoiseType, FlowParams, ColorStop, AlphaMode, VoronoiDistance
};
use camera::Camera;
use color::Color;
use scene::{create_solar_system, SceneData, PlanetConfig}; // Importamos PlanetConfig también
use shaders::SkyboxShader;

// ===================== ESTADOS DE LA VISTA =====================
#[derive(PartialEq)]
enum ViewMode {
    SolarSystem,                  // Vista general orbitando
    Warp { target_index: usize }, // Vista enfocada en un planeta específico
}

// ===================== FUNCIONES DE MATRICES Y RENDER BASE =====================

fn create_model_matrix(translation: Vec3, scale: f32, rotation: Vec3) -> Mat4 {
    let (sin_x, cos_x) = rotation.x.sin_cos();
    let (sin_y, cos_y) = rotation.y.sin_cos();
    let (sin_z, cos_z) = rotation.z.sin_cos();
    let rx = Mat4::new(1.0,0.0,0.0,0.0, 0.0,cos_x,-sin_x,0.0, 0.0,sin_x,cos_x,0.0, 0.0,0.0,0.0,1.0);
    let ry = Mat4::new(cos_y,0.0,sin_y,0.0, 0.0,1.0,0.0,0.0, -sin_y,0.0,cos_y,0.0, 0.0,0.0,0.0,1.0);
    let rz = Mat4::new(cos_z,-sin_z,0.0,0.0, sin_z,cos_z,0.0,0.0, 0.0,0.0,1.0,0.0, 0.0,0.0,0.0,1.0);
    let rotation_matrix = rz * ry * rx;
    let transform_matrix = Mat4::new(scale,0.0,0.0,translation.x, 0.0,scale,0.0,translation.y, 0.0,0.0,scale,translation.z, 0.0,0.0,0.0,1.0);
    transform_matrix * rotation_matrix
}

fn render_orbit(framebuffer: &mut Framebuffer, uniforms: &Uniforms, radius: f32, segments: usize) {
    let mut prev_vertex: Option<Vertex> = None;
    for i in 0..=segments {
        let angle = (i as f32 / segments as f32) * 2.0 * PI;
        let pos = Vec3::new(radius * angle.cos(), 0.0, radius * angle.sin());
        let v = Vertex::new_with_color(pos, Color::new(60, 80, 100));
        let orbit_uniforms = Uniforms { model_matrix: Mat4::identity(), ..*uniforms };
        if let Some(transformed_v) = vertex_shader(&v, &orbit_uniforms) {
            if let Some(prev) = prev_vertex {
                line(&prev, &transformed_v, framebuffer);
            }
            prev_vertex = Some(transformed_v);
        } else { prev_vertex = None; }
    }
}

fn render(framebuffer: &mut Framebuffer, uniforms: &Uniforms, obj: &Obj, shader: &dyn FragmentShader) {
    let (positions, normals, uvs) = obj.mesh_buffers();
    obj.for_each_face(|i0, i1, i2| {
        let p0 = positions[i0]; let p1 = positions[i1]; let p2 = positions[i2];
        let n0 = normals.get(i0).cloned().unwrap_or(Vec3::new(0.0, 1.0, 0.0));
        let n1 = normals.get(i1).cloned().unwrap_or(Vec3::new(0.0, 1.0, 0.0));
        let n2 = normals.get(i2).cloned().unwrap_or(Vec3::new(0.0, 1.0, 0.0));
        let t0 = uvs.get(i0).cloned().unwrap_or(nalgebra_glm::Vec2::new(0.0, 0.0));
        let t1 = uvs.get(i1).cloned().unwrap_or(nalgebra_glm::Vec2::new(0.0, 0.0));
        let t2 = uvs.get(i2).cloned().unwrap_or(nalgebra_glm::Vec2::new(0.0, 0.0));
        let v0_opt = vertex_shader(&Vertex::new(p0, n0, t0), uniforms);
        let v1_opt = vertex_shader(&Vertex::new(p1, n1, t1), uniforms);
        let v2_opt = vertex_shader(&Vertex::new(p2, n2, t2), uniforms);
        if let (Some(v0), Some(v1), Some(v2)) = (v0_opt, v1_opt, v2_opt) {
            triangle_with_shader(&v0, &v1, &v2, shader, uniforms, framebuffer);
        }
    });
}

fn render_alpha(framebuffer: &mut Framebuffer, uniforms: &Uniforms, obj: &Obj, shader: &dyn FragmentShader, z_bias: f32) {
    let (positions, normals, uvs) = obj.mesh_buffers();
    obj.for_each_face(|i0, i1, i2| {
        let p0 = positions[i0]; let p1 = positions[i1]; let p2 = positions[i2];
        let n0 = normals.get(i0).cloned().unwrap_or(Vec3::new(0.0, 1.0, 0.0));
        let n1 = normals.get(i1).cloned().unwrap_or(Vec3::new(0.0, 1.0, 0.0));
        let n2 = normals.get(i2).cloned().unwrap_or(Vec3::new(0.0, 1.0, 0.0));
        let t0 = uvs.get(i0).cloned().unwrap_or(nalgebra_glm::Vec2::new(0.0, 0.0));
        let t1 = uvs.get(i1).cloned().unwrap_or(nalgebra_glm::Vec2::new(0.0, 0.0));
        let t2 = uvs.get(i2).cloned().unwrap_or(nalgebra_glm::Vec2::new(0.0, 0.0));
        let v0_opt = vertex_shader(&Vertex::new(p0, n0, t0), uniforms);
        let v1_opt = vertex_shader(&Vertex::new(p1, n1, t1), uniforms);
        let v2_opt = vertex_shader(&Vertex::new(p2, n2, t2), uniforms);
        if let (Some(mut v0), Some(mut v1), Some(mut v2)) = (v0_opt, v1_opt, v2_opt) {
            v0.transformed_position.z += z_bias;
            v1.transformed_position.z += z_bias;
            v2.transformed_position.z += z_bias;
            triangle_with_shader(&v0, &v1, &v2, shader, uniforms, framebuffer);
        }
    });
}

// ===================== MAIN =====================

fn main() {
    let window_width = 1300;
    let window_height = 800;
    let framebuffer_width = 1300;
    let framebuffer_height = 800;
    let frame_delay = Duration::from_millis(16);

    let mut framebuffer = Framebuffer::new(framebuffer_width, framebuffer_height);
    let mut window = Window::new("Sistema Solar - WARP MODE", window_width, window_height, WindowOptions::default()).unwrap();
    window.set_position(200, 200);
    framebuffer.set_background_color(0x000000);

    // 1. Carga de Modelos
    let ship_obj = Obj::load("assets/models/Pelican.obj").expect("Failed to load ship");
    let planet_obj = Obj::load("assets/models/Planet.obj").expect("Failed to load planet");
    let rings_obj = Obj::load("assets/models/PlanetRing.obj").expect("Failed to load rings");
    let moon_obj = Obj::load("assets/models/Planet.obj").expect("Failed to load moon");

    let skybox_obj = Obj::load("assets/models/Planet.obj").expect("Failed to load skybox cube");
    let t_posx = Texture::load("assets/skybox/posx.png");
    let t_negx = Texture::load("assets/skybox/negx.png");
    let t_posy = Texture::load("assets/skybox/posy.png");
    let t_negy = Texture::load("assets/skybox/negy.png");
    let t_posz = Texture::load("assets/skybox/negz.png");
    let t_negz = Texture::load("assets/skybox/posz.png");

    // 2. Cálculos de Escala
    let (min_v, max_v) = ship_obj.bounds();
    let center = (min_v + max_v) * 0.5;
    let max_extent = (max_v - min_v).x.abs().max((max_v - min_v).y.abs()).max((max_v - min_v).z.abs());
    let mut ship_scale = if max_extent > 0.0 { 5.0 / max_extent } else { 1.0 };
    let model_offset = -center * ship_scale;

    let (p_min, p_max) = planet_obj.bounds();
    let p_center = (p_min + p_max) * 0.5;
    let p_ext = (p_max - p_min).x.abs().max((p_max - p_min).y.abs()).max((p_max - p_min).z.abs());
    let base_planet_scale = if p_ext > 0.0 { 1.0 / p_ext } else { 1.0 }; 
    let planet_offset = -p_center * base_planet_scale;

    let (rmin, rmax) = rings_obj.bounds();
    let ext = rmax - rmin;
    let ring_radius_x = (ext.x.abs() / 2.0).max(1.0);
    let ring_radius_z = (ext.z.abs() / 2.0).max(1.0);

    // 3. Inicializar Escena
    let scene_data = create_solar_system(&moon_obj);

    // 4. Shader de la Nave
    let ship_shader = ProceduralLayerShader {
        noise: NoiseParams { kind: NoiseType::Value, scale: 1.0, octaves: 1, lacunarity: 0.0, gain: 0.0, cell_size: 0.0, w1:0.0,w2:0.0,w3:0.0,w4:0.0, dist: VoronoiDistance::Euclidean, animate_time: false, time_speed: 0.0, animate_spin: false, spin_speed: 0.0, ring_swirl_amp: 0.0, ring_swirl_freq: 0.0, band_frequency: 0.0, band_contrast: 0.0, lat_shear: 0.0, turb_scale: 0.0, turb_octaves: 0, turb_lacunarity: 0.0, turb_gain: 0.0, turb_amp: 0.0, flow: FlowParams::default() },
        color_stops: vec![ColorStop{ threshold: 0.0, color: Color::from_hex(0xAAAAAA) }],
        color_hardness: 0.0, lighting_enabled: true, light_dir: normalize(&Vec3::new(0.5, 1.0, 1.0)), light_min: 0.2, light_max: 1.0, alpha_mode: AlphaMode::Opaque
    };

    // 5. Configuración Inicial de Nave y Cámara
    let default_ship_pos = Vec3::new(0.0, 50.0, 400.0);
    let mut ship_position = default_ship_pos; 
    let mut ship_rotation = Vec3::new(0.0, 0.0, PI); 
    
    // ESTADO INICIAL WARP
    let mut view_mode = ViewMode::SolarSystem;
    let warp_scale_factor = 10.0; // Factor de escala para que el planeta se vea grande en Warp

    let aspect = framebuffer_width as f32 / framebuffer_height as f32;
    let mut camera = Camera::new(Vec3::new(0.0, 15.0, 60.0), ship_position, aspect);
    
    // Órbita de cámara
    let cam_back_dist: f32 = 40.0;
    let cam_height: f32 = 15.0;
    let cam_radius: f32 = (cam_back_dist * cam_back_dist + cam_height * cam_height).sqrt();
    let mut orbit_yaw: f32 = 0.0;
    let mut orbit_pitch: f32 = 0.0;
    let orbit_speed: f32 = PI / 180.0 * 2.0;

    let start_time = Instant::now();

    // ===================== BUCLE =====================
    while window.is_open() {
        if window.is_key_down(Key::Escape) { break; }
        let time_secs = start_time.elapsed().as_secs_f32();

        // --- INPUT DE WARP (TELETRANSPORTE) ---
        // Regresar al sistema solar completo
        if window.is_key_pressed(Key::Backspace, minifb::KeyRepeat::No) {
            view_mode = ViewMode::SolarSystem;
            ship_position = default_ship_pos; 
            ship_rotation = Vec3::new(0.0, 0.0, PI);
        }

        // Ir a un planeta específico (Teclas 0 a 5)
        let mut target_warp = None;
        if window.is_key_pressed(Key::Key0, minifb::KeyRepeat::No) { target_warp = Some(0); } // Sol
        if window.is_key_pressed(Key::Key1, minifb::KeyRepeat::No) { target_warp = Some(1); } // Tierra
        if window.is_key_pressed(Key::Key2, minifb::KeyRepeat::No) { target_warp = Some(2); } // Marte
        if window.is_key_pressed(Key::Key3, minifb::KeyRepeat::No) { target_warp = Some(3); } // Jupiter
        if window.is_key_pressed(Key::Key4, minifb::KeyRepeat::No) { target_warp = Some(4); } // Saturno
        if window.is_key_pressed(Key::Key5, minifb::KeyRepeat::No) { target_warp = Some(5); } // Urano

        if let Some(index) = target_warp {
            if index < scene_data.planets.len() {
                view_mode = ViewMode::Warp { target_index: index };
                
                // CAMBIO AQUÍ:
                // Al hacer warp, movemos la nave frente al planeta.
                let target_planet_scale = scene_data.planets[index].scale * warp_scale_factor;
                
                // Posición: en Z positivo
                ship_position = Vec3::new(0.0, 0.0, target_planet_scale * 3.0);
                
                // Rotación: Mantenemos PI en Z (corrección de modelo) y agregamos PI en Y para girar 180 grados
                ship_rotation = Vec3::new(0.0, PI, PI); 
            }
        }

        // --- INPUT DE NAVE ---
        handle_input(&window, &mut ship_position, &mut ship_rotation, &mut ship_scale);

        // --- CÁLCULO DE CÁMARA ---
        let orientation = create_model_matrix(Vec3::new(0.0,0.0,0.0), 1.0, ship_rotation);
        let mut forward = Vec3::new(orientation[(0,2)], orientation[(1,2)], orientation[(2,2)]);
        forward.y = 0.0;
        if forward.magnitude() > 1e-6 { forward = forward.normalize(); }

        if window.is_key_down(Key::J) { orbit_yaw -= orbit_speed; }
        if window.is_key_down(Key::L) { orbit_yaw += orbit_speed; }
        if window.is_key_down(Key::I) { orbit_pitch += orbit_speed; }
        if window.is_key_down(Key::K) { orbit_pitch -= orbit_speed; }
        orbit_pitch = orbit_pitch.clamp(-PI/3.0, PI/3.0);
        if window.is_key_down(Key::LeftShift) { orbit_yaw = 0.0; orbit_pitch = 0.0; }

        let base_offset = -forward * cam_back_dist + Vec3::new(0.0, cam_height, 0.0);
        let base_dir = base_offset / cam_radius;
        let base_pitch = base_dir.y.asin();
        let base_yaw = base_dir.x.atan2(base_dir.z);
        let yaw = base_yaw + orbit_yaw;
        let pitch = (base_pitch + orbit_pitch).clamp(-PI*0.49, PI*0.49);
        let final_offset = Vec3::new(
            cam_radius * yaw.sin() * pitch.cos(),
            cam_radius * pitch.sin(),
            cam_radius * yaw.cos() * pitch.cos()
        );
        camera.position = ship_position + final_offset;
        camera.target = ship_position;

        framebuffer.clear();
        let view_matrix = camera.view_matrix();
        let projection_matrix = camera.projection_matrix();

        // --- RENDER SKYBOX ---
        let skybox_pos = camera.position;
        let skybox_scale = -1500.0; 
        let skybox_model = create_model_matrix(skybox_pos, skybox_scale, Vec3::new(0.0, 0.0, 0.0));
        let skybox_uniforms = Uniforms {
            model_matrix: skybox_model,
            view_matrix, projection_matrix, screen_width: framebuffer_width as f32, screen_height: framebuffer_height as f32,
            time: 0.0, seed: 0, ring_a: 0.0, ring_b: 0.0, ring_plane_xy: false
        };
        let skybox_shader_instance = SkyboxShader {
            tex_pos_x: &t_posx, tex_neg_x: &t_negx, tex_pos_y: &t_posy, tex_negy_y: &t_negy, tex_pos_z: &t_posz, tex_neg_z: &t_negz,
        };
        render(&mut framebuffer, &skybox_uniforms, &skybox_obj, &skybox_shader_instance);

        // --- RENDER NAVE ---
        let ship_model = create_model_matrix(ship_position + model_offset, ship_scale, ship_rotation);
        let u_ship = Uniforms {
            model_matrix: ship_model, view_matrix, projection_matrix,
            screen_width: framebuffer_width as f32, screen_height: framebuffer_height as f32,
            time: time_secs, seed: 0, ring_a: 0.0, ring_b: 0.0, ring_plane_xy: false
        };
        render(&mut framebuffer, &u_ship, &ship_obj, &ship_shader);

        // --- RENDER PLANETAS (DEPENDIENDO DEL MODO) ---
        match view_mode {
            // Modo 1: Sistema Solar Completo
            ViewMode::SolarSystem => {
                for (i, planet) in scene_data.planets.iter().enumerate() {
                    // Cálculo de órbita
                    let angle = planet.orbit_offset + time_secs * planet.orbit_speed;
                    let px = planet.dist_from_sun * angle.cos();
                    let pz = planet.dist_from_sun * angle.sin();
                    let translation = Vec3::new(px, 0.0, pz);

                    // Dibujar líneas de órbita solo en modo sistema
                    if planet.dist_from_sun > 0.1 {
                        let u_orbit = Uniforms { model_matrix: Mat4::identity(), view_matrix, projection_matrix, screen_width: framebuffer_width as f32, screen_height: framebuffer_height as f32, time: 0.0, seed: 0, ring_a: 0.0, ring_b: 0.0, ring_plane_xy: false };
                        render_orbit(&mut framebuffer, &u_orbit, planet.dist_from_sun, 128);
                    }

                    // Escala y Rotación normal
                    let scale_final = base_planet_scale * planet.scale;
                    let rot_planet = Vec3::new(0.0, time_secs * planet.rotation_speed, 0.0);

                    // Renderizado centralizado
                    render_planet_system(
                        &mut framebuffer, 
                        &uniforms_base(view_matrix, projection_matrix, framebuffer_width, framebuffer_height),
                        planet, translation, scale_final, rot_planet, planet_offset,
                        &planet_obj, &rings_obj, &scene_data, i, time_secs,
                        ring_radius_x, ring_radius_z
                    );
                }
            },

            // Modo 2: Warp (Un solo planeta, centrado y grande)
            ViewMode::Warp { target_index } => {
                let planet = &scene_data.planets[target_index];

                // Posición forzada al centro (0,0,0)
                let translation = Vec3::new(0.0, 0.0, 0.0);
                
                // Escala aumentada por el factor warp
                let scale_final = base_planet_scale * planet.scale * warp_scale_factor;
                
                // Rotación sigue activa
                let rot_planet = Vec3::new(0.0, time_secs * planet.rotation_speed, 0.0);

                // Renderizado centralizado
                render_planet_system(
                    &mut framebuffer, 
                    &uniforms_base(view_matrix, projection_matrix, framebuffer_width, framebuffer_height),
                    planet, translation, scale_final, rot_planet, planet_offset,
                    &planet_obj, &rings_obj, &scene_data, target_index, time_secs,
                    ring_radius_x, ring_radius_z
                );
            }
        }

        window.update_with_buffer(&framebuffer.buffer, framebuffer_width, framebuffer_height).unwrap();
        std::thread::sleep(frame_delay);
    }
}

fn handle_input(window: &Window, pos: &mut Vec3, rot: &mut Vec3, _scale: &mut f32) {
    let rot_speed = PI / 90.0;
    let move_speed = 1.0;
    if window.is_key_down(Key::A) { rot.y -= rot_speed; }
    if window.is_key_down(Key::D) { rot.y += rot_speed; }
    let orientation = create_model_matrix(Vec3::new(0.0,0.0,0.0), 1.0, *rot);
    let mut forward = Vec3::new(orientation[(0,2)], orientation[(1,2)], orientation[(2,2)]);
    forward.y = 0.0;
    if forward.magnitude() > 1e-6 { forward = forward.normalize(); }
    if window.is_key_down(Key::W) { *pos += forward * move_speed; }
    if window.is_key_down(Key::S) { *pos -= forward * move_speed; }
}

// ===================== HELPER DE RENDERIZADO =====================

fn uniforms_base(view: Mat4, proj: Mat4, w: usize, h: usize) -> Uniforms {
    Uniforms {
        model_matrix: Mat4::identity(), view_matrix: view, projection_matrix: proj,
        screen_width: w as f32, screen_height: h as f32,
        time: 0.0, seed: 0, ring_a: 0.0, ring_b: 0.0, ring_plane_xy: false
    }
}

// Función que renderiza el planeta, sus nubes, anillos y lunas
fn render_planet_system(
    framebuffer: &mut Framebuffer, base_uniforms: &Uniforms,
    planet: &PlanetConfig, translation: Vec3, scale_final: f32, rotation: Vec3, planet_offset: Vec3,
    planet_obj: &Obj, rings_obj: &Obj, scene_data: &SceneData, 
    planet_index: usize, time_secs: f32, ring_rad_x: f32, ring_rad_z: f32
) {
    // 1. CUERPO DEL PLANETA
    let model = create_model_matrix(translation + planet_offset * planet.scale, scale_final, rotation);
    let uniforms = Uniforms {
        model_matrix: model, time: time_secs, seed: planet_index as i32 * 999, 
        ..*base_uniforms
    };
    render(framebuffer, &uniforms, planet_obj, &*scene_data.planet_shaders[planet.shader_index]);

    // 2. ATMÓSFERA (NUBES)
    if planet.has_atmosphere {
         if let Some(cloud_idx) = planet.atmosphere_shader_index {
             if let Some(cloud_shader) = &scene_data.cloud_shaders[cloud_idx] {
                let cloud_model = create_model_matrix(translation + planet_offset * planet.scale, scale_final * 1.02, rotation);
                let u_clouds = Uniforms { model_matrix: cloud_model, ..uniforms };
                render_alpha(framebuffer, &u_clouds, planet_obj, &**cloud_shader, 0.01);
             }
         }
    }

    // 3. ANILLOS
    if planet.has_rings {
        if let Some(ring_idx) = planet.ring_shader_index {
            if let Some(ring_shader) = &scene_data.ring_shaders[ring_idx] {
                let animated_ring_rot = Vec3::new(planet.ring_tilt.x, planet.ring_tilt.y + time_secs * 0.05, planet.ring_tilt.z);
                let ring_model = create_model_matrix(translation + planet_offset * planet.scale, scale_final * 1.2, animated_ring_rot);
                let u_rings = Uniforms { 
                    model_matrix: ring_model, ring_a: ring_rad_x, ring_b: ring_rad_z, ring_plane_xy: false, 
                    ..uniforms 
                };
                render(framebuffer, &u_rings, rings_obj, &**ring_shader);
            }
        }
    }

    // 4. LUNAS ASOCIADAS
    for m in &scene_data.moons {
        if m.parent_index == planet_index {
            let moon_model = m.model_matrix(translation, scale_final, time_secs);
            let u_moon = Uniforms {
                model_matrix: moon_model, time: time_secs, seed: m.seed,
                ..uniforms
            };
            render(framebuffer, &u_moon, &m.obj, &*m.shader);
        }
    }
}