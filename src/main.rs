use nalgebra_glm::{Vec3, Mat4, normalize};
use minifb::{Key, Window, WindowOptions};
use std::time::{Duration, Instant};
use std::f32::consts::PI;

mod framebuffer;
mod triangle;
mod line; // Necesario para dibujar las órbitas
mod vertex;
mod obj;
mod color;
mod fragment;
mod shaders;
mod camera;
mod noise; 

use framebuffer::Framebuffer;
use vertex::Vertex;
use obj::Obj;
use triangle::triangle_with_shader;
use line::line; // Importamos la función de línea
use shaders::{
    vertex_shader, Uniforms, FragmentShader,
    ProceduralLayerShader, NoiseParams, NoiseType, VoronoiDistance,
    ColorStop, AlphaMode, FlowParams
};
use camera::Camera;
use color::Color;

// ===================== ESTRUCTURAS DE CONFIGURACIÓN =====================

struct PlanetConfig {
    dist_from_sun: f32,
    orbit_speed: f32,
    orbit_offset: f32,
    scale: f32,
    rotation_speed: f32,
    shader_index: usize,
    has_rings: bool,
    ring_shader_index: Option<usize>,
    ring_tilt: Vec3,
    has_atmosphere: bool,
    atmosphere_shader_index: Option<usize>,
}

struct Moon {
    obj: Obj,
    scale_rel: f32,
    orbit_px: f32,      // Radio órbita relativo al padre
    orbit_speed: f32,
    phase0: f32,
    tilt: Vec3,
    shader: Box<dyn FragmentShader + Send + Sync>,
    seed: i32,
    parent_index: usize, // Índice del planeta en el vector de planetas
}

impl Moon {
    fn model_matrix(&self, planet_pos: Vec3, planet_scale: f32, t: f32) -> Mat4 {
        let angle = self.phase0 + t * self.orbit_speed;
        let dist = self.orbit_px * planet_scale * 4.0; 
        
        let dx = angle.cos() * dist;
        let dz = angle.sin() * dist;

        let tr = Vec3::new(planet_pos.x + dx, planet_pos.y, planet_pos.z + dz);
        let spin = Vec3::new(self.tilt.x, self.tilt.y + t * 0.5, self.tilt.z);
        create_model_matrix(tr, planet_scale * self.scale_rel, spin)
    }
}

// ===================== MATRICES =====================

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

// ===================== RENDERIZADO =====================

fn render_orbit(framebuffer: &mut Framebuffer, uniforms: &Uniforms, radius: f32, segments: usize) {
    let mut prev_vertex: Option<Vertex> = None;
    for i in 0..=segments {
        let angle = (i as f32 / segments as f32) * 2.0 * PI;
        let pos = Vec3::new(radius * angle.cos(), 0.0, radius * angle.sin());
        let v = Vertex::new_with_color(pos, Color::new(60, 80, 100));
        
        // Usamos model matrix identidad para la órbita
        let mut orbit_uniforms = Uniforms { model_matrix: Mat4::identity(), ..*uniforms };

        if let Some(transformed_v) = vertex_shader(&v, &orbit_uniforms) {
            if let Some(prev) = prev_vertex {
                let fragments = line(&prev, &transformed_v);
                for frag in fragments {
                    let x = frag.position.x as usize;
                    let y = frag.position.y as usize;
                    if x < framebuffer.width && y < framebuffer.height {
                        framebuffer.point(x, y, frag.depth, frag.color.to_hex());
                    }
                }
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
            let fragments = triangle_with_shader(&v0, &v1, &v2, shader, uniforms);
            for frag in fragments {
                let x = frag.position.x as usize;
                let y = frag.position.y as usize;
                if x < framebuffer.width && y < framebuffer.height {
                    framebuffer.draw_rgba(x, y, frag.depth, frag.color.to_hex(), frag.alpha);
                }
            }
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

        if let (Some(v0), Some(v1), Some(v2)) = (v0_opt, v1_opt, v2_opt) {
            let fragments = triangle_with_shader(&v0, &v1, &v2, shader, uniforms);
            for frag in fragments {
                let x = frag.position.x as usize;
                let y = frag.position.y as usize;
                if x < framebuffer.width && y < framebuffer.height {
                    framebuffer.draw_rgba(x, y, frag.depth + z_bias, frag.color.to_hex(), frag.alpha);
                }
            }
        }
    });
}

// ===================== MAIN =====================

fn main() {
    let window_width = 800;
    let window_height = 600;
    let framebuffer_width = 800;
    let framebuffer_height = 600;
    let frame_delay = Duration::from_millis(16);

    let mut framebuffer = Framebuffer::new(framebuffer_width, framebuffer_height);
    let mut window = Window::new("Sistema Solar Completo", window_width, window_height, WindowOptions::default()).unwrap();
    window.set_position(200, 200);
    framebuffer.set_background_color(0x000000);

    // ===== Carga de Modelos =====
    let ship_obj = Obj::load("assets/models/Pelican.obj").expect("Failed to load ship");
    let planet_obj = Obj::load("assets/models/Planet.obj").expect("Failed to load planet");
    let rings_obj = Obj::load("assets/models/PlanetRing.obj").expect("Failed to load rings");
    let moon_obj = Obj::load("assets/models/Planet.obj").expect("Failed to load moon");

    // ===== Escalas Base =====
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
    let ring_radius_x = (ext.x.abs() / 2.0).max(1e-6);
    let ring_radius_y = (ext.y.abs() / 2.0).max(1e-6);

    // ===== SHADERS =====

    // 0. SOL
    let sun_shader = ProceduralLayerShader {
        noise: NoiseParams { 
            kind: NoiseType::BandedGas, scale: 1.0, octaves: 16, lacunarity: 2.2, gain: 0.52, cell_size: 0.35, w1: 1.0, w2: 0.0, w3:0.0, w4:0.0, dist: VoronoiDistance::Euclidean, 
            animate_time: false, time_speed: 0.0, animate_spin: true, spin_speed: 0.12,
            ring_swirl_amp: 0.0, ring_swirl_freq: 8.0, band_frequency: 20.0, band_contrast: 0.05, lat_shear: 0.05, turb_scale: 8.0, turb_octaves: 4, turb_lacunarity: 2.0, turb_gain: 0.55, turb_amp: 1.4,
            flow: FlowParams { enabled: true, flow_scale: 2.5, strength: 0.09, time_speed: 0.7, jets_base_speed: 0.04, jets_frequency: 5.0, phase_amp: 2.0 }
        },
        color_stops: vec![
            ColorStop{ threshold: 0.00, color: Color::from_hex(0xA33600) },
            ColorStop{ threshold: 0.25, color: Color::from_hex(0x7A1400) },
            ColorStop{ threshold: 0.50, color: Color::from_hex(0xE04800) },
            ColorStop{ threshold: 0.75, color: Color::from_hex(0xFFC640) },
            ColorStop{ threshold: 1.00, color: Color::from_hex(0xFFF9D0) },
        ],
        color_hardness: 0.15, lighting_enabled: false, light_dir: Vec3::new(0.0, 1.0, 0.0), light_min: 1.0, light_max: 1.0, alpha_mode: AlphaMode::Opaque
    };

    // 1. TIERRA
    let terrain_shader = ProceduralLayerShader {
        noise: NoiseParams { kind: NoiseType::Perlin, scale: 3.0, octaves: 5, lacunarity: 2.0, gain: 0.5, cell_size: 0.35, w1: 1.0, w2: 0.0, w3: 0.0, w4: 0.0, dist: VoronoiDistance::Euclidean, animate_time: false, time_speed: 1.0, animate_spin: false, spin_speed: 0.0, ring_swirl_amp: 0.0, ring_swirl_freq: 0.0, band_frequency: 0.0, band_contrast: 0.0, lat_shear: 0.0, turb_scale: 0.0, turb_octaves: 0, turb_lacunarity: 0.0, turb_gain: 0.0, turb_amp: 0.0, flow: FlowParams::default() },
        color_stops: vec![
            ColorStop { threshold: 0.35, color: Color::from_hex(0x1B3494) }, ColorStop { threshold: 0.48, color: Color::from_hex(0x203FB0) }, ColorStop { threshold: 0.50, color: Color::from_hex(0x4B87DB) },
            ColorStop { threshold: 0.51, color: Color::from_hex(0xA4957F) }, ColorStop { threshold: 0.52, color: Color::from_hex(0x88B04B) }, ColorStop { threshold: 0.60, color: Color::from_hex(0x668736) }, ColorStop { threshold: 0.70, color: Color::from_hex(0x597A2A) },
        ],
        color_hardness: 0.25, lighting_enabled: true, light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)), light_min: 0.35, light_max: 1.0, alpha_mode: AlphaMode::Opaque
    };
    let clouds_shader = ProceduralLayerShader {
        noise: NoiseParams { kind: NoiseType::Value, scale: 5.0, octaves: 3, lacunarity: 2.0, gain: 0.5, cell_size: 0.25, w1: 1.0, w2: 0.0, w3: 0.0, w4: 0.0, dist: VoronoiDistance::Euclidean, animate_time: true, time_speed: 0.1, animate_spin: true, spin_speed: 0.05, ring_swirl_amp: 0.0, ring_swirl_freq: 8.0, band_frequency: 0.0, band_contrast: 0.0, lat_shear: 0.0, turb_scale: 0.0, turb_octaves: 0, turb_lacunarity: 0.0, turb_gain: 0.0, turb_amp: 0.0, flow: FlowParams::default() },
        color_stops: vec![ ColorStop { threshold: 0.0, color: Color::from_hex(0xEDEDED) }, ColorStop { threshold: 1.0, color: Color::from_hex(0xFFFFFF) } ],
        color_hardness: 0.0, lighting_enabled: true, light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)), light_min: 0.8, light_max: 1.0,
        alpha_mode: AlphaMode::Threshold { threshold: 0.50, sharpness: 6.0, coverage_bias: 0.05, invert: false },
    };

    // 2. MARTE
    let mars_shader = ProceduralLayerShader {
        noise: NoiseParams { kind: NoiseType::Perlin, scale: 4.0, octaves: 4, lacunarity: 2.0, gain: 0.5, cell_size: 0.40, w1: 1.0, w2: 0.0, w3:0.0, w4:0.0, dist: VoronoiDistance::Euclidean, animate_time: false, time_speed: 0.0, animate_spin: false, spin_speed: 0.0, ring_swirl_amp: 0.0, ring_swirl_freq: 8.0, band_frequency: 0.0, band_contrast: 0.0, lat_shear: 0.0, turb_scale: 6.0, turb_octaves: 3, turb_lacunarity: 2.0, turb_gain: 0.5, turb_amp: 0.2, flow: FlowParams::default() },
        color_stops: vec![ ColorStop { threshold: 0.10, color: Color::from_hex(0x3D1F1A) }, ColorStop { threshold: 0.35, color: Color::from_hex(0x6B2F26) }, ColorStop { threshold: 0.60, color: Color::from_hex(0xA94432) }, ColorStop { threshold: 0.85, color: Color::from_hex(0xD27A4A) }, ColorStop { threshold: 1.00, color: Color::from_hex(0xF3C9A2) } ],
        color_hardness: 0.35, lighting_enabled: true, light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)), light_min: 0.35, light_max: 1.05, alpha_mode: AlphaMode::Opaque
    };
    let mars_clouds_shader = ProceduralLayerShader {
        noise: NoiseParams { kind: NoiseType::Value, scale: 2.0, octaves: 3, lacunarity: 2.0, gain: 0.5, cell_size: 0.25, w1: 1.0, w2: 0.0, w3:0.0, w4:0.0, dist: VoronoiDistance::Euclidean, animate_time: true, time_speed: 0.05, animate_spin: true, spin_speed: 0.02, ring_swirl_amp: 0.0, ring_swirl_freq: 8.0, band_frequency: 0.0, band_contrast: 0.0, lat_shear: 0.0, turb_scale: 0.0, turb_octaves: 0, turb_lacunarity: 0.0, turb_gain: 0.0, turb_amp: 0.0, flow: FlowParams::default() },
        color_stops: vec![ ColorStop { threshold: 0.0, color: Color::from_hex(0xEDEDED) }, ColorStop { threshold: 1.0, color: Color::from_hex(0xFFFFFF) } ],
        color_hardness: 0.0, lighting_enabled: true, light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)), light_min: 0.8, light_max: 1.0,
        alpha_mode: AlphaMode::Threshold { threshold: 0.60, sharpness: 6.0, coverage_bias: 0.05, invert: false },
    };

    // 3. JUPITER (GASEOSO)
    let jupiter_shader = ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::BandedGas, scale: 1.0, octaves: 4, lacunarity: 2.0, gain: 0.5, cell_size: 0.0, w1:0.0,w2:0.0,w3:0.0,w4:0.0, dist: VoronoiDistance::Euclidean,
            animate_time: false, time_speed: 0.0, animate_spin: false, spin_speed: 0.0, ring_swirl_amp: 0.0, ring_swirl_freq: 0.0,
            band_frequency: 6.0, band_contrast: 1.0, lat_shear: 0.2, turb_scale: 10.0, turb_octaves: 4, turb_lacunarity: 2.0, turb_gain: 0.55, turb_amp: 0.35,
            flow: FlowParams { enabled: true, flow_scale: 3.0, strength: 0.04, time_speed: 0.6, jets_base_speed: 0.12, jets_frequency: 6.0, phase_amp: 3.0 }
        },
        color_stops: vec![ ColorStop { threshold: 0.0, color: Color::from_hex(0x734D1E) }, ColorStop { threshold: 0.4, color: Color::from_hex(0xB58C5A) }, ColorStop { threshold: 0.8, color: Color::from_hex(0xE4D1B5) } ],
        color_hardness: 0.3, lighting_enabled: true, light_dir: normalize(&Vec3::new(1.0, 0.5, 1.0)), light_min: 0.4, light_max: 1.0, alpha_mode: AlphaMode::Opaque
    };

    // 4. SATURNO
    let saturn_shader = ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::BandedGas, scale: 1.0, octaves: 4, lacunarity: 2.0, gain: 0.5, cell_size: 0.35, w1: 1.0, w2: 0.0, w3:0.0, w4:0.0, dist: VoronoiDistance::Euclidean,
            animate_time: false, time_speed: 0.0, animate_spin: false, spin_speed: 0.0,
            ring_swirl_amp: 0.0, ring_swirl_freq: 8.0, band_frequency: 5.0, band_contrast: 1.0, lat_shear: 0.10, turb_scale: 1.0, turb_octaves: 4, turb_lacunarity: 2.0, turb_gain: 0.5, turb_amp: 0.35,
            flow: FlowParams { enabled: true, flow_scale: 3.0, strength: 0.04, time_speed: 0.25, jets_base_speed: 0.1, jets_frequency: 6.0, phase_amp: 3.0 }
        },
        color_stops: vec![
            ColorStop { threshold: 0.00, color: Color::from_hex(0xEDD198) }, ColorStop { threshold: 0.32, color: Color::from_hex(0xD5BE8A) }, ColorStop { threshold: 0.72, color: Color::from_hex(0xF6E6C4) }, ColorStop { threshold: 1.00, color: Color::from_hex(0xFFF9E6) },
        ],
        color_hardness: 0.24, lighting_enabled: true, light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)), light_min: 0.50, light_max: 1.08, alpha_mode: AlphaMode::Opaque
    };
    let saturn_rings_shader = ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::RadialGradient { inner: 0.70, outer: 1.0, invert: false, bias: 0.10, gamma: 1.0 },
            scale: 1.0, octaves: 1, lacunarity: 2.0, gain: 0.5, cell_size: 1.0, w1: 1.0, w2: 0.0, w3:0.0, w4:0.0, dist: VoronoiDistance::Euclidean,
            animate_time: false, time_speed: 0.0, animate_spin: true, spin_speed: 1.2, ring_swirl_amp: 0.05, ring_swirl_freq: 10.0, band_frequency: 0.0, band_contrast: 0.0, lat_shear: 0.0, turb_scale: 0.0, turb_octaves: 0, turb_lacunarity: 0.0, turb_gain: 0.0, turb_amp: 0.0, flow: FlowParams::default()
        },
        color_stops: vec![ ColorStop { threshold: 0.00, color: Color::from_hex(0x0E0F12) }, ColorStop { threshold: 0.50, color: Color::from_hex(0xDCCEB3) }, ColorStop { threshold: 1.00, color: Color::from_hex(0x0A0C10) } ],
        color_hardness: 0.0, lighting_enabled: false, light_dir: Vec3::new(0.0,1.0,0.0), light_min: 1.0, light_max: 1.0, alpha_mode: AlphaMode::Opaque
    };

    // 5. URANO
    let uranus_shader = ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::BandedGas, scale: 1.0, octaves: 3, lacunarity: 2.0, gain: 0.55, cell_size: 0.35, w1: 1.0, w2: 0.0, w3:0.0, w4:0.0, dist: VoronoiDistance::Euclidean,
            animate_time: false, time_speed: 0.0, animate_spin: false, spin_speed: 0.0, ring_swirl_amp: 0.0, ring_swirl_freq: 8.0, band_frequency: 3.0, band_contrast: 0.5, lat_shear: 0.05, turb_scale: 4.0, turb_octaves: 3, turb_lacunarity: 2.0, turb_gain: 0.5, turb_amp: 0.15, flow: FlowParams::default()
        },
        color_stops: vec![ ColorStop { threshold: 0.00, color: Color::from_hex(0x0A2F4F) }, ColorStop { threshold: 0.60, color: Color::from_hex(0x42A8C8) }, ColorStop { threshold: 1.00, color: Color::from_hex(0xC5F6FF) } ],
        color_hardness: 0.25, lighting_enabled: true, light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)), light_min: 0.45, light_max: 1.1, alpha_mode: AlphaMode::Opaque
    };
    let uranus_rings_shader = ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::RadialGradient { inner: 0.82, outer: 1.0, invert: false, bias: 0.08, gamma: 1.1 },
            scale: 1.0, octaves: 1, lacunarity: 2.0, gain: 0.5, cell_size: 1.0, w1: 1.0, w2: 0.0, w3:0.0, w4:0.0, dist: VoronoiDistance::Euclidean,
            animate_time: false, time_speed: 0.0, animate_spin: true, spin_speed: 1.6, ring_swirl_amp: 0.03, ring_swirl_freq: 14.0, band_frequency: 0.0, band_contrast: 0.0, lat_shear: 0.0, turb_scale: 0.0, turb_octaves: 0, turb_lacunarity: 0.0, turb_gain: 0.0, turb_amp: 0.0, flow: FlowParams::default()
        },
        color_stops: vec![ ColorStop { threshold: 0.00, color: Color::from_hex(0x2E7E9E) } ],
        color_hardness: 0.0, lighting_enabled: false, light_dir: Vec3::new(0.0,1.0,0.0), light_min: 1.0, light_max: 1.0, alpha_mode: AlphaMode::Opaque
    };

    // Shader Nave
    let ship_shader = ProceduralLayerShader {
        noise: NoiseParams { kind: NoiseType::Value, scale: 1.0, octaves: 1, lacunarity: 0.0, gain: 0.0, cell_size: 0.0, w1:0.0,w2:0.0,w3:0.0,w4:0.0, dist: VoronoiDistance::Euclidean, animate_time: false, time_speed: 0.0, animate_spin: false, spin_speed: 0.0, ring_swirl_amp: 0.0, ring_swirl_freq: 0.0, band_frequency: 0.0, band_contrast: 0.0, lat_shear: 0.0, turb_scale: 0.0, turb_octaves: 0, turb_lacunarity: 0.0, turb_gain: 0.0, turb_amp: 0.0, flow: FlowParams::default() },
        color_stops: vec![ColorStop{ threshold: 0.0, color: Color::from_hex(0xAAAAAA) }],
        color_hardness: 0.0, lighting_enabled: true, light_dir: normalize(&Vec3::new(0.5, 1.0, 1.0)), light_min: 0.2, light_max: 1.0, alpha_mode: AlphaMode::Opaque
    };

    // Vectores de acceso
    let shaders = vec![ &sun_shader, &terrain_shader, &mars_shader, &jupiter_shader, &saturn_shader, &uranus_shader ];
    let ring_shaders = vec![ None, None, None, None, Some(&saturn_rings_shader), Some(&uranus_rings_shader) ];
    let cloud_shaders = vec![ None, Some(&clouds_shader), Some(&mars_clouds_shader), None, None, None ];

    // ===== SISTEMA SOLAR (Configuración) =====
    let solar_system = vec![
        // 0. SOL (Centro)
        PlanetConfig { dist_from_sun: 0.0,   orbit_speed: 0.0,  orbit_offset: 0.0, scale: 20.0, rotation_speed: 0.05, shader_index: 0, has_rings: false, ring_shader_index: None, ring_tilt: Vec3::new(0.0,0.0,0.0), has_atmosphere: false, atmosphere_shader_index: None },
        // 1. TIERRA (Cerca)
        PlanetConfig { dist_from_sun: 60.0,  orbit_speed: 0.4,  orbit_offset: 0.0, scale: 5.0,  rotation_speed: 0.5,  shader_index: 1, has_rings: false, ring_shader_index: None, ring_tilt: Vec3::new(0.0,0.0,0.0), has_atmosphere: true,  atmosphere_shader_index: Some(1) },
        // 2. MARTE
        PlanetConfig { dist_from_sun: 85.0,  orbit_speed: 0.3,  orbit_offset: 2.0, scale: 4.0,  rotation_speed: 0.4,  shader_index: 2, has_rings: false, ring_shader_index: None, ring_tilt: Vec3::new(0.0,0.0,0.0), has_atmosphere: true,  atmosphere_shader_index: Some(2) },
        // 3. JUPITER (Grande, Gaseoso)
        PlanetConfig { dist_from_sun: 120.0, orbit_speed: 0.2,  orbit_offset: 4.0, scale: 14.0, rotation_speed: 0.3,  shader_index: 3, has_rings: false, ring_shader_index: None, ring_tilt: Vec3::new(0.0,0.0,0.0), has_atmosphere: false, atmosphere_shader_index: None },
        // 4. SATURNO (Anillos)
        PlanetConfig { dist_from_sun: 160.0, orbit_speed: 0.15, orbit_offset: 1.0, scale: 11.0, rotation_speed: 0.2,  shader_index: 4, has_rings: true,  ring_shader_index: Some(4), ring_tilt: Vec3::new(0.3, 0.0, 0.1), has_atmosphere: false, atmosphere_shader_index: None },
        // 5. URANO (Lejos)
        PlanetConfig { dist_from_sun: 200.0, orbit_speed: 0.1,  orbit_offset: 5.5, scale: 9.0,  rotation_speed: 0.1,  shader_index: 5, has_rings: true,  ring_shader_index: Some(5), ring_tilt: Vec3::new(1.3, 0.0, 0.0), has_atmosphere: false, atmosphere_shader_index: None },
    ];

    // Lunas (Orbitan a la Tierra - Índice 1 en el vector nuevo)
    let moon_shader_rocky = ProceduralLayerShader {
        noise: NoiseParams { kind: NoiseType::Voronoi, scale: 2.6, octaves: 3, lacunarity: 2.0, gain: 0.5, cell_size: 0.4, w1:1.0, w2:1.0, w3:1.0, w4:0.0, dist: VoronoiDistance::Euclidean, animate_time: false, time_speed: 0.0, animate_spin: false, spin_speed: 0.0, ring_swirl_amp: 0.0, ring_swirl_freq: 0.0, band_frequency: 0.0, band_contrast: 0.0, lat_shear: 0.0, turb_scale: 0.0, turb_octaves: 0, turb_lacunarity: 0.0, turb_gain: 0.0, turb_amp: 0.0, flow: FlowParams::default() },
        color_stops: vec![ ColorStop{threshold:0.25, color:Color::from_hex(0x5E5347)}, ColorStop{threshold:0.85, color:Color::from_hex(0xCBB79F)} ],
        color_hardness: 0.25, lighting_enabled: true, light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)), light_min: 0.35, light_max: 1.05, alpha_mode: AlphaMode::Opaque
    };
    let moons: Vec<Moon> = vec![
        Moon { obj: moon_obj.clone(), scale_rel: 0.25, orbit_px: 15.0, orbit_speed: 1.5, phase0: 0.0, tilt: Vec3::new(0.05, 0.0, 0.05), shader: Box::new(moon_shader_rocky), seed: 8888, parent_index: 1 }
    ];

    // ===== NAVE & CAMARA =====
    let mut ship_position = Vec3::new(0.0, 30.0, 280.0); // Lejos para ver el sistema
    let mut ship_rotation = Vec3::new(0.0, 0.0, PI); 
    let aspect = framebuffer_width as f32 / framebuffer_height as f32;
    let mut camera = Camera::new(Vec3::new(0.0, 15.0, 60.0), ship_position, aspect);
    
    let cam_back_dist: f32 = 40.0;
    let cam_height: f32 = 15.0;
    let cam_radius: f32 = (cam_back_dist * cam_back_dist + cam_height * cam_height).sqrt();
    let mut orbit_yaw: f32 = 0.0;
    let mut orbit_pitch: f32 = 0.0;
    let orbit_speed: f32 = PI / 180.0 * 2.0;

    let start_time = Instant::now();

    while window.is_open() {
        if window.is_key_down(Key::Escape) { break; }
        let time_secs = start_time.elapsed().as_secs_f32();

        handle_input(&window, &mut ship_position, &mut ship_rotation, &mut ship_scale);

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

        // Render Nave
        let ship_model = create_model_matrix(ship_position + model_offset, ship_scale, ship_rotation);
        let u_ship = Uniforms {
            model_matrix: ship_model, view_matrix, projection_matrix,
            screen_width: framebuffer_width as f32, screen_height: framebuffer_height as f32,
            time: time_secs, seed: 0, ring_a: 0.0, ring_b: 0.0, ring_plane_xy: false
        };
        render(&mut framebuffer, &u_ship, &ship_obj, &ship_shader);

        // Render Sistema Solar
        for (i, planet) in solar_system.iter().enumerate() {
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
            let model = create_model_matrix(translation + planet_offset * planet.scale, scale_final, rot_planet);
            let uniforms = Uniforms {
                model_matrix: model, view_matrix, projection_matrix,
                screen_width: framebuffer_width as f32, screen_height: framebuffer_height as f32,
                time: time_secs, seed: i as i32 * 999, ring_a: 0.0, ring_b: 0.0, ring_plane_xy: false
            };
            render(&mut framebuffer, &uniforms, &planet_obj, shaders[planet.shader_index]);

            if planet.has_atmosphere {
                 if let Some(cloud_idx) = planet.atmosphere_shader_index {
                     if let Some(cloud_shader) = cloud_shaders[cloud_idx] {
                        let cloud_model = create_model_matrix(translation + planet_offset * planet.scale, scale_final * 1.02, rot_planet);
                        let u_clouds = Uniforms { model_matrix: cloud_model, ..uniforms };
                        render_alpha(&mut framebuffer, &u_clouds, &planet_obj, cloud_shader, 0.01);
                     }
                 }
            }

            if planet.has_rings {
                if let Some(ring_idx) = planet.ring_shader_index {
                     if let Some(ring_shader) = ring_shaders[ring_idx] {
                        let ring_model = create_model_matrix(translation + planet_offset * planet.scale, scale_final * 1.2, planet.ring_tilt);
                        let u_rings = Uniforms { 
                            model_matrix: ring_model, ring_a: ring_radius_x, ring_b: ring_radius_y, ring_plane_xy: true, ..uniforms 
                        };
                        render(&mut framebuffer, &u_rings, &rings_obj, ring_shader);
                     }
                }
            }

            for m in &moons {
                if m.parent_index == i {
                    let moon_model = m.model_matrix(translation, scale_final, time_secs);
                    let u_moon = Uniforms {
                        model_matrix: moon_model, view_matrix, projection_matrix,
                        screen_width: framebuffer_width as f32, screen_height: framebuffer_height as f32,
                        time: time_secs, seed: m.seed, ring_a: 0.0, ring_b: 0.0, ring_plane_xy: false
                    };
                    render(&mut framebuffer, &u_moon, &m.obj, &*m.shader);
                }
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