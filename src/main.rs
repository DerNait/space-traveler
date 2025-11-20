// main.rs
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
    vertex_shader, Uniforms, FragmentShader, ProceduralLayerShader, NoiseParams, NoiseType, FlowParams, ColorStop, AlphaMode, VoronoiDistance, TextureShader
};
use camera::Camera;
use color::Color;
use scene::{create_solar_system, SceneData, PlanetConfig}; 
use shaders::SkyboxShader;

// ===================== ESTADOS DE LA VISTA =====================
#[derive(PartialEq, Clone, Copy)]
enum ViewMode {
    SolarSystem,              // Vista general orbitando
    Warp { target_index: usize }, // Vista enfocada en un planeta específico
}

// ===================== ESTRUCTURA DE LA NAVE Y FÍSICA =====================

struct Ship {
    pub position: Vec3,
    pub rotation: Vec3, // x: pitch, y: yaw, z: roll
    pub speed: f32,
    pub max_speed: f32,
    pub acceleration: f32,
    pub friction: f32,
    pub turn_speed: f32, 
}

impl Ship {
    fn new(position: Vec3) -> Self {
        Self {
            position,
            rotation: Vec3::new(0.0, 0.0, 0.0), 
            speed: 0.0,
            max_speed: 5.0,      
            acceleration: 0.1,   
            friction: 0.05,      
            turn_speed: 0.04, 
        }
    }
}

fn lerp_angle(start: f32, end: f32, t: f32) -> f32 {
    let mut delta = end - start;
    while delta > PI { delta -= 2.0 * PI; }
    while delta < -PI { delta += 2.0 * PI; }
    start + delta * t
}

// ===================== MATRICES Y RENDER =====================

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

// ===================== COLLISION SYSTEM (FIXED) =====================

fn check_planet_collision(position: &mut Vec3, planets: &[PlanetConfig], time: f32, base_planet_scale: f32) -> bool {
    let mut collided = false;
    
    // AUMENTADO: Antes era 2.0. Ahora 5.0 para cubrir el tamaño de la nave.
    // Esto evita que la "nariz" de la nave entre al planeta antes que el centro.
    let buffer_radius = 5.0; 

    for planet in planets {
        // Ignorar el Sol (0,0) si estamos muy lejos o si el shader index es 0 (Sol)
        // Esto es opcional, pero ayuda si el sol es gigante.
        
        // 1. Recalcular posición
        let angle = planet.orbit_offset + time * planet.orbit_speed;
        let px = planet.dist_from_sun * angle.cos();
        let pz = planet.dist_from_sun * angle.sin();
        let planet_pos = Vec3::new(px, 0.0, pz);

        // 2. Definir radio.
        // AUMENTADO: De 0.55 a 0.65. 
        // El cálculo es: EscalaVisual = Scale * Base. 
        // Si OBJ tiene diametro 1, radio es 0.5.
        // 0.5 * 1.3 (margen seguridad 30%) = 0.65 aprox.
        let visual_scale_factor = 0.65; 
        let planet_radius = (planet.scale * base_planet_scale * visual_scale_factor) + buffer_radius;

        // 3. Distancia
        let dist_vec = *position - planet_pos;
        let distance = dist_vec.magnitude();

        // 4. Resolver
        if distance < planet_radius {
            collided = true;
            let direction = normalize(&dist_vec);
            // Empujamos FUERTE hacia afuera
            *position = planet_pos + (direction * planet_radius);
        }
    }
    collided
}

// ===================== MAIN =====================

fn main() {
    let window_width = 1300;
    let window_height = 800;
    let framebuffer_width = 1300;
    let framebuffer_height = 800;
    let frame_delay = Duration::from_millis(16);

    let mut framebuffer = Framebuffer::new(framebuffer_width, framebuffer_height);
    let mut window = Window::new("Sistema Solar - WARP FLIGHT", window_width, window_height, WindowOptions::default()).unwrap();
    window.set_position(200, 200);
    framebuffer.set_background_color(0x000000);

    // 1. Carga de Modelos
    let ship_obj = Obj::load("assets/models/Pelican.obj").expect("Failed to load ship");
    let planet_obj = Obj::load("assets/models/Planet.obj").expect("Failed to load planet");
    let rings_obj = Obj::load("assets/models/PlanetRing.obj").expect("Failed to load rings");
    let moon_obj = Obj::load("assets/models/Planet.obj").expect("Failed to load moon");
    let warp_bubble_obj = Obj::load("assets/models/Planet.obj").expect("Failed to load warp bubble");
    let skybox_obj = Obj::load("assets/models/Planet.obj").expect("Failed to load skybox cube");
    
    let t_posx = Texture::load("assets/skybox/posx.png");
    let t_negx = Texture::load("assets/skybox/negx.png");
    let t_posy = Texture::load("assets/skybox/posy.png");
    let t_negy = Texture::load("assets/skybox/negy.png");
    let t_posz = Texture::load("assets/skybox/negz1.png");
    let t_negz = Texture::load("assets/skybox/posz.png");

    let t_ship = Texture::load("assets/models/Pelican-Textures.png");
    let ship_shader = TextureShader {
        texture: &t_ship,
        light_dir: normalize(&Vec3::new(0.5, 1.0, 1.0)),
        ambient: 0.2,
        diffuse: 0.8,
    };

    // 2. Cálculos de Escala
    let (min_v, max_v) = ship_obj.bounds();
    let center = (min_v + max_v) * 0.5;
    let max_extent = (max_v - min_v).x.abs().max((max_v - min_v).y.abs()).max((max_v - min_v).z.abs());
    let ship_scale = if max_extent > 0.0 { 5.0 / max_extent } else { 1.0 };
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

    // 4. Shaders
    /* let ship_shader = ProceduralLayerShader {
        noise: NoiseParams { kind: NoiseType::Value, scale: 1.0, octaves: 1, lacunarity: 0.0, gain: 0.0, cell_size: 0.0, w1:0.0,w2:0.0,w3:0.0,w4:0.0, dist: VoronoiDistance::Euclidean, animate_time: false, time_speed: 0.0, animate_spin: false, spin_speed: 0.0, ring_swirl_amp: 0.0, ring_swirl_freq: 0.0, band_frequency: 0.0, band_contrast: 0.0, lat_shear: 0.0, turb_scale: 0.0, turb_octaves: 0, turb_lacunarity: 0.0, turb_gain: 0.0, turb_amp: 0.0, flow: FlowParams::default() },
        color_stops: vec![ColorStop{ threshold: 0.0, color: Color::from_hex(0xAAAAAA) }],
        color_hardness: 0.0, lighting_enabled: true, light_dir: normalize(&Vec3::new(0.5, 1.0, 1.0)), light_min: 0.2, light_max: 1.0, alpha_mode: AlphaMode::Opaque
    }; */

    let warp_shader = ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::Voronoi, scale: 0.8, octaves: 1, lacunarity: 2.0, gain: 0.5, cell_size: 1.0, w1: 1.0, w2: 0.0, w3: 0.0, w4: 0.0, 
            dist: VoronoiDistance::Manhattan, animate_time: true, time_speed: 2.0, animate_spin: true, spin_speed: 1.0,
            ring_swirl_amp: 0.0, ring_swirl_freq: 0.0, band_frequency: 0.0, band_contrast: 0.0, lat_shear: 0.0, turb_scale: 0.0, turb_octaves: 0, turb_lacunarity: 0.0, turb_gain: 0.0, turb_amp: 0.0, flow: FlowParams::default()
        },
        color_stops: vec![
            ColorStop { threshold: 0.0, color: Color::from_hex(0x001133) },
            ColorStop { threshold: 0.4, color: Color::from_hex(0x0044AA) },
            ColorStop { threshold: 0.8, color: Color::from_hex(0x00FFFF) },
            ColorStop { threshold: 1.0, color: Color::from_hex(0xFFFFFF) },
        ],
        color_hardness: 0.1, lighting_enabled: false, light_dir: Vec3::new(0.0, 1.0, 0.0), light_min: 1.0, light_max: 1.0,
        alpha_mode: AlphaMode::Threshold { threshold: 0.2, sharpness: 2.0, coverage_bias: 0.0, invert: false }
    };

    // 5. Configuración Inicial
    let default_ship_pos = Vec3::new(0.0, 50.0, 400.0);
    let mut ship = Ship::new(default_ship_pos);
    let mut view_mode = ViewMode::SolarSystem;
    let warp_scale_factor = 10.0;

    // Variables Warp
    let mut is_warping = false;
    let mut warp_start_time: Option<Instant> = None;
    let mut pending_target_index: Option<usize> = None;
    let mut pending_return = false;
    let mut has_switched_scene = false;

    // Tiempos Warp
    let warp_enter_duration = 1.5;  
    let warp_fade_in = 0.5;        
    let warp_hold_white = 1.0;     
    let warp_fade_out = 0.5;       
    let warp_exit_duration = 2.0;   
    let warp_total_duration = warp_enter_duration + warp_fade_in + warp_hold_white + warp_fade_out + warp_exit_duration; 

    // Cámara
    let aspect = framebuffer_width as f32 / framebuffer_height as f32;
    let mut camera = Camera::new(Vec3::new(0.0, 15.0, 60.0), ship.position, aspect);
    let mut orbit_yaw: f32 = 0.0; 
    let mut orbit_pitch: f32 = 0.0;
    let orbit_sens: f32 = 0.06; 
    let cam_dist: f32 = 60.0;

    let start_time = Instant::now();

    while window.is_open() {
        if window.is_key_down(Key::Escape) { break; }
        let time_secs = start_time.elapsed().as_secs_f32();

        // ================== FÍSICA Y LOGICA ==================

        // 1. Control Orbit (Inputs)
        if window.is_key_down(Key::J) { orbit_yaw += orbit_sens; }
        if window.is_key_down(Key::L) { orbit_yaw -= orbit_sens; }
        if window.is_key_down(Key::I) { orbit_pitch += orbit_sens; }
        if window.is_key_down(Key::K) { orbit_pitch -= orbit_sens; }
        orbit_pitch = orbit_pitch.clamp(-PI/2.5, PI/2.5); 

        // --- CÁMARA CINEMÁTICA SUAVE (ESPACIO) ---
        if window.is_key_down(Key::Space) {
            // Calculamos el ángulo objetivo
            let target_yaw = ship.position.x.atan2(ship.position.z);
            let target_pitch = -0.2;
            
            // Factor de suavidad (0.0 a 1.0). 
            // 0.1 es muy suave, 0.3 es rápido.
            let smooth_factor = 0.3;

            // Usamos lerp_angle para evitar que la cámara de vueltas locas (360 grados)
            orbit_yaw = lerp_angle(orbit_yaw, target_yaw, smooth_factor);
            
            // Interpolación lineal simple para el pitch
            orbit_pitch = orbit_pitch + (target_pitch - orbit_pitch) * smooth_factor;
        }

        // 2. Control Nave (Velocidad)
        if window.is_key_down(Key::W) { ship.speed += ship.acceleration; }
        else {
            if ship.speed > 0.0 { ship.speed -= ship.friction; }
            if ship.speed < 0.0 { ship.speed += ship.friction; }
        }
        if window.is_key_down(Key::S) { ship.speed -= ship.acceleration * 1.5; }
        ship.speed = ship.speed.clamp(-ship.max_speed * 0.5, ship.max_speed);
        if ship.speed.abs() < 0.01 { ship.speed = 0.0; }

        // 3. Calcular POSICIÓN BASE de la Cámara (Orbital)
        let cam_x = cam_dist * orbit_yaw.sin() * orbit_pitch.cos();
        let cam_y = cam_dist * orbit_pitch.sin();
        let cam_z = cam_dist * orbit_yaw.cos() * orbit_pitch.cos();
        let cam_offset = Vec3::new(cam_x, cam_y, cam_z);

        let mut shake = Vec3::new(0.0, 0.0, 0.0);
        if is_warping {
            shake = Vec3::new((time_secs * 30.0).sin(), (time_secs * 25.0).cos(), 0.0) * 0.5;
        }

        // Asignar posición provisional de la cámara
        camera.position = ship.position + cam_offset + shake;
        camera.target = ship.position; 

        // 4. Movimiento Nave
        let target_dir = normalize(&(ship.position - camera.position)); 
        let target_yaw = target_dir.x.atan2(target_dir.z);
        let target_pitch = -target_dir.y.asin(); 

        // Banking
        let mut turn_diff = target_yaw - ship.rotation.y;
        while turn_diff > PI { turn_diff -= 2.0 * PI; }
        while turn_diff < -PI { turn_diff += 2.0 * PI; }
        let max_roll = PI / 4.0; 
        let target_roll = -turn_diff.clamp(-1.0, 1.0) * max_roll;

        ship.rotation.y = lerp_angle(ship.rotation.y, target_yaw, ship.turn_speed);
        ship.rotation.x = lerp_angle(ship.rotation.x, target_pitch, ship.turn_speed);
        ship.rotation.z = ship.rotation.z + (target_roll - ship.rotation.z) * 0.1; 

        let (s_sin_y, s_cos_y) = ship.rotation.y.sin_cos();
        let (s_sin_p, s_cos_p) = ship.rotation.x.sin_cos();
        let forward = Vec3::new(s_sin_y * s_cos_p, -s_sin_p, s_cos_y * s_cos_p).normalize();

        ship.position += forward * ship.speed;

        // ================== SISTEMA DE COLISIÓN ==================
        // Ejecutamos esto DESPUÉS de mover la nave y DESPUÉS de calcular la cámara inicial
       if !is_warping {
            match view_mode {
                // CASO 1: Sistema Solar (Múltiples planetas orbitando)
                ViewMode::SolarSystem => {
                    // A. Colisión NAVE
                    let hit = check_planet_collision(&mut ship.position, &scene_data.planets, time_secs, base_planet_scale);
                    if hit { ship.speed = 0.0; }

                    // B. Colisión CÁMARA
                    check_planet_collision(&mut camera.position, &scene_data.planets, time_secs, base_planet_scale);
                },

                // CASO 2: Warp (Un solo planeta gigante en el centro 0,0,0)
                ViewMode::Warp { target_index } => {
                    let planet = &scene_data.planets[target_index];
                    let center = Vec3::new(0.0, 0.0, 0.0);
                    
                    // Calculamos el radio con la escala de Warp aplicada
                    // Scale * Base * WarpFactor * AjusteVisual + Buffer
                    let visual_factor = 0.65; // Mismo ajuste visual que en solar
                    let buffer = 5.0;         // Mismo buffer de nave
                    
                    let warp_radius = (planet.scale * base_planet_scale * warp_scale_factor * visual_factor) + buffer;

                    // A. Colisión NAVE vs Planeta Warp
                    let hit = check_simple_collision(&mut ship.position, center, warp_radius);
                    if hit { ship.speed = 0.0; }

                    // B. Colisión CÁMARA vs Planeta Warp
                    check_simple_collision(&mut camera.position, center, warp_radius);
                }
            }
        }

        // ================== INPUT WARP ==================
        if !is_warping {
            if window.is_key_pressed(Key::Backspace, minifb::KeyRepeat::No) {
                is_warping = true;
                warp_start_time = Some(Instant::now());
                pending_return = true;
                pending_target_index = None;
                has_switched_scene = false; 
            }
            let mut target_warp = None;
            if window.is_key_pressed(Key::Key0, minifb::KeyRepeat::No) { target_warp = Some(0); }
            if window.is_key_pressed(Key::Key1, minifb::KeyRepeat::No) { target_warp = Some(1); }
            if window.is_key_pressed(Key::Key2, minifb::KeyRepeat::No) { target_warp = Some(2); }
            if window.is_key_pressed(Key::Key3, minifb::KeyRepeat::No) { target_warp = Some(3); }
            if window.is_key_pressed(Key::Key4, minifb::KeyRepeat::No) { target_warp = Some(4); }
            if window.is_key_pressed(Key::Key5, minifb::KeyRepeat::No) { target_warp = Some(5); }

            if let Some(index) = target_warp {
                if index < scene_data.planets.len() {
                    is_warping = true;
                    warp_start_time = Some(Instant::now());
                    pending_target_index = Some(index);
                    pending_return = false;
                    has_switched_scene = false; 
                }
            }
        }

        // ================== ANIMACIÓN WARP ==================
        let mut warp_bubble_scale = 0.0;
        let mut white_screen_alpha = 0.0;

        if is_warping {
            if let Some(start_t) = warp_start_time {
                let elapsed = start_t.elapsed().as_secs_f32();

                if elapsed < warp_enter_duration {
                    let t = elapsed / warp_enter_duration;
                    warp_bubble_scale = t * t * (3.0 - 2.0 * t) * 15.0; 
                    ship.speed = ship.max_speed * 2.0;
                } else if elapsed < (warp_enter_duration + warp_fade_in) {
                    warp_bubble_scale = 15.0; 
                    ship.speed = ship.max_speed * 3.0;
                    let t = (elapsed - warp_enter_duration) / warp_fade_in;
                    white_screen_alpha = t.clamp(0.0, 1.0);
                } else if elapsed < (warp_enter_duration + warp_fade_in + warp_hold_white) {
                    warp_bubble_scale = 15.0;
                    white_screen_alpha = 1.0;
                    ship.speed = 0.0; 

                    if !has_switched_scene {
                        if pending_return {
                            view_mode = ViewMode::SolarSystem;
                            ship.position = default_ship_pos;
                            ship.speed = 0.0; 
                            ship.rotation = Vec3::new(0.0, 0.0, 0.0); 
                            orbit_yaw = PI; orbit_pitch = 0.0; 
                        } else if let Some(idx) = pending_target_index {
                            view_mode = ViewMode::Warp { target_index: idx };
                            let target_planet_scale = scene_data.planets[idx].scale * warp_scale_factor;
                            ship.position = Vec3::new(0.0, 0.0, target_planet_scale * 8.0);
                            ship.rotation = Vec3::new(0.0, 0.0, 0.0); 
                            ship.speed = 0.0;
                        }
                        has_switched_scene = true;
                    }
                } else if elapsed < (warp_enter_duration + warp_fade_in + warp_hold_white + warp_fade_out) {
                    warp_bubble_scale = 15.0;
                    let t = (elapsed - (warp_enter_duration + warp_fade_in + warp_hold_white)) / warp_fade_out;
                    white_screen_alpha = 1.0 - t.clamp(0.0, 1.0);
                } else if elapsed < warp_total_duration {
                    white_screen_alpha = 0.0;
                    let t = (elapsed - (warp_total_duration - warp_exit_duration)) / warp_exit_duration;
                    let shrink = 1.0 - t;
                    warp_bubble_scale = (shrink * shrink) * 15.0;
                } else {
                    is_warping = false;
                    warp_start_time = None;
                    warp_bubble_scale = 0.0;
                    has_switched_scene = false;
                    white_screen_alpha = 0.0;
                }
            }
        }

        // ================== RENDERIZADO ==================
        framebuffer.clear();
        
        // IMPORTANTE: Tomamos las matrices AQUÍ, después de resolver colisiones.
        // Si la colisión modificó camera.position, view_matrix usará la posición corregida.
        let view_matrix = camera.view_matrix();
        let projection_matrix = camera.projection_matrix();

        let moon_dist_factor = match view_mode {
            ViewMode::Warp { .. } => 4.0, 
            _ => 1.0,
        };

        // 1. SKYBOX
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

        // 2. NAVE
        let ship_model = create_model_matrix(ship.position + model_offset, ship_scale, ship.rotation);
        let u_ship = Uniforms {
            model_matrix: ship_model, view_matrix, projection_matrix,
            screen_width: framebuffer_width as f32, screen_height: framebuffer_height as f32,
            time: time_secs, seed: 0, ring_a: 0.0, ring_b: 0.0, ring_plane_xy: false
        };
        render(&mut framebuffer, &u_ship, &ship_obj, &ship_shader);

        // 3. WARP BUBBLE
        if is_warping && warp_bubble_scale > 0.1 {
            let warp_rot = Vec3::new(time_secs, time_secs * 0.5, 0.0);
            let warp_model = create_model_matrix(ship.position, warp_bubble_scale, warp_rot);
            let u_warp = Uniforms { model_matrix: warp_model, ..u_ship };
            render_alpha(&mut framebuffer, &u_warp, &warp_bubble_obj, &warp_shader, 0.0);
        }

        // 4. PLANETAS
        let hide_universe = (is_warping && white_screen_alpha >= 0.99) || (is_warping && warp_bubble_scale > 50.0); 
        
        if !hide_universe {
            match view_mode {
                ViewMode::SolarSystem => {
                    for (i, planet) in scene_data.planets.iter().enumerate() {
                        let angle = planet.orbit_offset + time_secs * planet.orbit_speed;
                        let px = planet.dist_from_sun * angle.cos();
                        let pz = planet.dist_from_sun * angle.sin();
                        let translation = Vec3::new(px, 0.0, pz);

                        if planet.dist_from_sun > 0.1 {
                            let u_orbit = Uniforms { model_matrix: Mat4::identity(), view_matrix, projection_matrix, screen_width: framebuffer_width as f32, screen_height: framebuffer_height as f32, time: 0.0, seed: 0, ring_a: 0.0, ring_b: 0.0, ring_plane_xy: false };
                            render_orbit(&mut framebuffer, &u_orbit, planet.dist_from_sun, 128);
                        }

                        let scale_final = base_planet_scale * planet.scale;
                        let rot_planet = Vec3::new(0.0, time_secs * planet.rotation_speed, 0.0);

                        render_planet_system(&mut framebuffer, &uniforms_base(view_matrix, projection_matrix, framebuffer_width, framebuffer_height),
                            planet, translation, scale_final, rot_planet, planet_offset,
                            &planet_obj, &rings_obj, &scene_data, i, time_secs, ring_radius_x, ring_radius_z, moon_dist_factor);
                    }
                },

                ViewMode::Warp { target_index } => {
                    let planet = &scene_data.planets[target_index];
                    let translation = Vec3::new(0.0, 0.0, 0.0);
                    let scale_final = base_planet_scale * planet.scale * warp_scale_factor;
                    let rot_planet = Vec3::new(0.0, time_secs * planet.rotation_speed, 0.0);

                    render_planet_system(&mut framebuffer, &uniforms_base(view_matrix, projection_matrix, framebuffer_width, framebuffer_height),
                        planet, translation, scale_final, rot_planet, planet_offset,
                        &planet_obj, &rings_obj, &scene_data, target_index, time_secs, ring_radius_x, ring_radius_z, moon_dist_factor);
                }
            }
        }

        // --- POST PROCESO: WHITE FLASH ---
        if white_screen_alpha > 0.0 {
            let alpha_int = (white_screen_alpha * 255.0) as u32;
            if alpha_int >= 255 {
                framebuffer.buffer.fill(0xFFFFFF);
            } else {
                let inv_alpha = 255 - alpha_int;
                for pixel in framebuffer.buffer.iter_mut() {
                    let r = (*pixel >> 16) & 0xFF;
                    let g = (*pixel >> 8) & 0xFF;
                    let b = *pixel & 0xFF;
                    let nr = (r * inv_alpha / 255) + alpha_int;
                    let ng = (g * inv_alpha / 255) + alpha_int;
                    let nb = (b * inv_alpha / 255) + alpha_int;
                    *pixel = (nr << 16) | (ng << 8) | nb;
                }
            }
        }

        window.update_with_buffer(&framebuffer.buffer, framebuffer_width, framebuffer_height).unwrap();
        std::thread::sleep(frame_delay);
    }
}

// ===================== HELPER DE RENDERIZADO =====================
fn uniforms_base(view: Mat4, proj: Mat4, w: usize, h: usize) -> Uniforms {
    Uniforms {
        model_matrix: Mat4::identity(), view_matrix: view, projection_matrix: proj,
        screen_width: w as f32, screen_height: h as f32,
        time: 0.0, seed: 0, ring_a: 0.0, ring_b: 0.0, ring_plane_xy: false
    }
}

fn render_planet_system(
    framebuffer: &mut Framebuffer, base_uniforms: &Uniforms,
    planet: &PlanetConfig, translation: Vec3, scale_final: f32, rotation: Vec3, planet_offset: Vec3,
    planet_obj: &Obj, rings_obj: &Obj, scene_data: &SceneData, 
    planet_index: usize, time_secs: f32, ring_rad_x: f32, ring_rad_z: f32,
    moon_dist_mult: f32 
) {
    let model = create_model_matrix(translation + planet_offset * planet.scale, scale_final, rotation);
    let uniforms = Uniforms {
        model_matrix: model, time: time_secs, seed: planet_index as i32 * 999, 
        ..*base_uniforms
    };
    render(framebuffer, &uniforms, planet_obj, &*scene_data.planet_shaders[planet.shader_index]);

    if planet.has_atmosphere {
         if let Some(cloud_idx) = planet.atmosphere_shader_index {
             if let Some(cloud_shader) = &scene_data.cloud_shaders[cloud_idx] {
                let cloud_model = create_model_matrix(translation + planet_offset * planet.scale, scale_final * 1.02, rotation);
                let u_clouds = Uniforms { model_matrix: cloud_model, ..uniforms };
                render_alpha(framebuffer, &u_clouds, planet_obj, &**cloud_shader, 0.01);
             }
         }
    }

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

    for m in &scene_data.moons {
        if m.parent_index == planet_index {
            let mut moon_model = m.model_matrix(translation, scale_final, time_secs);
            let mx = moon_model[(0, 3)];
            let my = moon_model[(1, 3)];
            let mz = moon_model[(2, 3)];
            let dx = mx - translation.x;
            let dy = my - translation.y;
            let dz = mz - translation.z;
            moon_model[(0, 3)] = translation.x + dx * moon_dist_mult;
            moon_model[(1, 3)] = translation.y + dy * moon_dist_mult;
            moon_model[(2, 3)] = translation.z + dz * moon_dist_mult;
            let u_moon = Uniforms {
                model_matrix: moon_model, time: time_secs, seed: m.seed,
                ..uniforms
            };
            render(framebuffer, &u_moon, &m.obj, &*m.shader);
        }
    }
}

fn check_simple_collision(position: &mut Vec3, center: Vec3, radius: f32) -> bool {
    let dist_vec = *position - center;
    let distance = dist_vec.magnitude();

    if distance < radius {
        let direction = normalize(&dist_vec);
        // Empujamos hacia afuera
        *position = center + (direction * radius);
        return true;
    }
    false
}