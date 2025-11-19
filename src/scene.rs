use nalgebra_glm::{Vec3, Mat4, normalize};
use crate::obj::Obj;
use crate::color::Color;
use crate::shaders::{
    FragmentShader, ProceduralLayerShader, NoiseParams, 
    NoiseType, VoronoiDistance, ColorStop, AlphaMode, FlowParams
};
use std::f32::consts::PI;

// ===================== ESTRUCTURAS DE DATOS =====================

pub struct PlanetConfig {
    pub dist_from_sun: f32,
    pub orbit_speed: f32,
    pub orbit_offset: f32,
    pub scale: f32,
    pub rotation_speed: f32,
    pub shader_index: usize,
    pub has_rings: bool,
    pub ring_shader_index: Option<usize>,
    pub ring_tilt: Vec3,
    pub has_atmosphere: bool,
    pub atmosphere_shader_index: Option<usize>,
}

pub struct Moon {
    pub obj: Obj,
    pub scale_rel: f32,
    pub orbit_px: f32,
    pub orbit_speed: f32,
    pub phase0: f32,
    pub tilt: Vec3,
    pub shader: Box<dyn FragmentShader + Send + Sync>,
    pub seed: i32,
    pub parent_index: usize,
}

impl Moon {
    pub fn model_matrix(&self, planet_pos: Vec3, planet_scale: f32, t: f32) -> Mat4 {
        let angle = self.phase0 + t * self.orbit_speed;
        // Distancia relativa al borde del planeta
        let dist = planet_scale + (self.orbit_px * 0.5); 
        
        let dx = angle.cos() * dist;
        let dz = angle.sin() * dist;

        let tr = Vec3::new(planet_pos.x + dx, planet_pos.y, planet_pos.z + dz);
        // Rotación de la luna sobre sí misma
        let spin_angle = t * 0.5;
        
        // Matriz manual para evitar dependencia circular con main::create_model_matrix
        // (Copiamos la lógica de rotación básica aquí o usamos nalgebra directo)
        let (sin_x, cos_x) = self.tilt.x.sin_cos();
        let (sin_y, cos_y) = (self.tilt.y + spin_angle).sin_cos();
        let (sin_z, cos_z) = self.tilt.z.sin_cos();

        let rx = Mat4::new(1.0,0.0,0.0,0.0, 0.0,cos_x,-sin_x,0.0, 0.0,sin_x,cos_x,0.0, 0.0,0.0,0.0,1.0);
        let ry = Mat4::new(cos_y,0.0,sin_y,0.0, 0.0,1.0,0.0,0.0, -sin_y,0.0,cos_y,0.0, 0.0,0.0,0.0,1.0);
        let rz = Mat4::new(cos_z,-sin_z,0.0,0.0, sin_z,cos_z,0.0,0.0, 0.0,0.0,1.0,0.0, 0.0,0.0,0.0,1.0);
        
        let rotation_matrix = rz * ry * rx;
        let scale_final = planet_scale * self.scale_rel;
        let transform_matrix = Mat4::new(scale_final,0.0,0.0,tr.x, 0.0,scale_final,0.0,tr.y, 0.0,0.0,scale_final,tr.z, 0.0,0.0,0.0,1.0);
        
        transform_matrix * rotation_matrix
    }
}

// Contenedor para devolver todo lo necesario al Main
pub struct SceneData {
    pub planets: Vec<PlanetConfig>,
    pub moons: Vec<Moon>,
    pub planet_shaders: Vec<Box<dyn FragmentShader + Send + Sync>>,
    pub ring_shaders: Vec<Option<Box<dyn FragmentShader + Send + Sync>>>,
    pub cloud_shaders: Vec<Option<Box<dyn FragmentShader + Send + Sync>>>,
}

// ===================== GENERADOR DE ESCENA =====================

pub fn create_solar_system(moon_obj: &Obj) -> SceneData {
    
    // ------------------- SHADERS -------------------

    // 0. SOL
    let sun_shader = ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::BandedGas,

            // Base de ruido
            scale: 1.0,
            octaves: 16,
            lacunarity: 2.2,
            gain: 0.52,
            cell_size: 0.35,
            w1: 1.0, w2: 1.0, w3: 1.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,

            // El "tiempo" principal lo vamos a manejar con el flow,
            // aquí lo dejamos neutro
            animate_time: false,
            time_speed: 0.0,

            // Ligero spin global para que todo gire despacio
            animate_spin: true,
            spin_speed: 0.12,

            // No usamos swirl de anillos aquí
            ring_swirl_amp: 0.0,
            ring_swirl_freq: 8.0,

            // Bandas base (se rompen con turbulencia + flow)
            band_frequency: 20.0,   // pocas bandas grandes
            band_contrast: 0.05,   // no tan duras, más "lava"

            lat_shear: 0.05,       // un pelín de inclinación

            // Turbulencia fuerte para romper las bandas
            turb_scale: 8.0,
            turb_octaves: 4,
            turb_lacunarity: 2.0,
            turb_gain: 0.55,
            turb_amp: 1.4,         // alto = formas muy retorcidas

            // Flowmap: aquí está la magia de la "lava" que fluye
            flow: FlowParams {
                enabled: true,
                flow_scale: 2.5,   // tamaño de los remolinos
                strength: 0.09,    // cuánto se desplazan lon/lat
                time_speed: 0.7,   // velocidad de animación del flow
                jets_base_speed: 0.04,
                jets_frequency: 5.0,
                phase_amp: 2.0,    // qué tan fuerte se siente la transición de fases
            },
        },
        color_stops: vec![
            ColorStop { threshold: 0.00, color: Color::from_hex(0xA33600) }, // sombras muy oscuras
            ColorStop { threshold: 0.25, color: Color::from_hex(0x7A1400) }, // rojo profundo
            ColorStop { threshold: 0.50, color: Color::from_hex(0xE04800) }, // naranja brillante
            ColorStop { threshold: 0.75, color: Color::from_hex(0xFFC640) }, // amarillo intenso
            ColorStop { threshold: 1.00, color: Color::from_hex(0xFFF9D0) }, // casi blanco
        ],
        color_hardness: 0.15,         // bordes algo marcados entre "placas" de lava
        lighting_enabled: false,      // autoiluminado
        light_dir: normalize(&Vec3::new(0.0, 1.0, 0.0)),
        light_min: 1.0,
        light_max: 1.0,
        alpha_mode: AlphaMode::Opaque,
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

    // 3. JUPITER
    let gas_palette = vec![
        ColorStop { threshold: 0.00, color: Color::from_hex(0x734D1E) },
        ColorStop { threshold: 0.20, color: Color::from_hex(0x6E5034) },
        ColorStop { threshold: 0.40, color: Color::from_hex(0xB58C5A) },
        ColorStop { threshold: 0.60, color: Color::from_hex(0xD9B98B) },
        ColorStop { threshold: 0.80, color: Color::from_hex(0xA57442) },
        ColorStop { threshold: 1.00, color: Color::from_hex(0xE4D1B5) },
    ];

    let jupiter_shader = ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::BandedGas,
            scale: 1.0, octaves: 4, lacunarity: 2.0, gain: 0.5,
            cell_size: 0.35, w1: 1.0, w2: 1.0, w3: 1.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,
            animate_time: false, time_speed: 0.0,
            animate_spin: false,  spin_speed: 0.0,

            ring_swirl_amp: 0.0,
            ring_swirl_freq: 8.0,

            band_frequency: 4.0,
            band_contrast: 1.0,
            lat_shear: 0.2,
            turb_scale: 10.0,
            turb_octaves: 4,
            turb_lacunarity: 2.0,
            turb_gain: 0.55,
            turb_amp: 0.35,
            flow: FlowParams {
                enabled: true,
                flow_scale: 3.0,
                strength: 0.04,
                time_speed: 0.6,
                jets_base_speed: 0.12,
                jets_frequency: 6.0,
                phase_amp: 3.0,
            },
        },
        color_stops: gas_palette,
        color_hardness: 0.35,
        lighting_enabled: true,
        light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)),
        light_min: 0.45,
        light_max: 1.05,
        alpha_mode: AlphaMode::Opaque,
    };

    // 4. SATURNO
    let saturn_shader = ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::BandedGas,
            scale: 1.0,
            octaves: 4,
            lacunarity: 2.0,
            gain: 0.5,
            cell_size: 0.35,
            w1: 1.0, w2: 1.0, w3: 1.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,

            animate_time: false,
            time_speed: 0.0,
            animate_spin: false,
            spin_speed: 0.0,

            ring_swirl_amp: 0.0,
            ring_swirl_freq: 8.0,

            band_frequency: 5.0,
            band_contrast: 1.0,

            lat_shear: 0.10,

            turb_scale: 1.0,
            turb_octaves: 4,
            turb_lacunarity: 2.0,
            turb_gain: 0.5,
            turb_amp: 0.35,

            flow: FlowParams {
                enabled: true,
                flow_scale: 3.0,
                strength: 0.04,
                time_speed: 0.25,
                jets_base_speed: 0.1,
                jets_frequency: 6.0,
                phase_amp: 3.0,
            },
        },
        color_stops: vec![
            // zona de sombras ligeramente marrón
            ColorStop { threshold: 0.00, color: Color::from_hex(0xEDD198) },
            ColorStop { threshold: 0.18, color: Color::from_hex(0xB48D5A) },

            // crema suave bastante ancho
            ColorStop { threshold: 0.32, color: Color::from_hex(0xD5BE8A) },

            // franja anaranjada finita
            ColorStop { threshold: 0.36, color: Color::from_hex(0xE5A95C) },

            // vuelve a crema más claro
            ColorStop { threshold: 0.44, color: Color::from_hex(0xEFD49C) },

            // otra zona amplia crema muy clara
            ColorStop { threshold: 0.58, color: Color::from_hex(0xF4E0B6) },

            // tramo final casi todo crema muy luminoso
            ColorStop { threshold: 0.72, color: Color::from_hex(0xF6E6C4) },
            ColorStop { threshold: 0.86, color: Color::from_hex(0xFBF0D9) },
            ColorStop { threshold: 1.00, color: Color::from_hex(0xFFF9E6) },
        ],
        // Un poquito más de dureza para que se lean las franjas
        color_hardness: 0.24,

        lighting_enabled: true,
        light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)),
        light_min: 0.50,
        light_max: 1.08,
        alpha_mode: AlphaMode::Opaque,
    };
    let saturn_rings_shader = ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::RadialGradient {
                inner: 0.70,
                outer: 1.0,
                invert: false,
                bias: 0.10,
                gamma: 1.0,
            },
            scale: 1.0, octaves: 1, lacunarity: 2.0, gain: 0.5,
            cell_size: 1.0, w1: 1.0, w2: 0.0, w3: 0.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,
            animate_time: false, time_speed: 0.0,
            animate_spin: true, spin_speed: 1.2,

            ring_swirl_amp: 0.05,
            ring_swirl_freq: 10.0,

            band_frequency: 0.0,
            band_contrast: 0.0,
            lat_shear: 0.0,
            turb_scale: 0.0,
            turb_octaves: 0,
            turb_lacunarity: 0.0,
            turb_gain: 0.0,
            turb_amp: 0.0,
            flow: FlowParams::default(),
        },
        color_stops: vec![
            ColorStop { threshold: 0.00, color: Color::from_hex(0x0E0F12) },
            ColorStop { threshold: 0.12, color: Color::from_hex(0xDCCEB3) },
            ColorStop { threshold: 0.28, color: Color::from_hex(0x2A2E36) },
            ColorStop { threshold: 0.45, color: Color::from_hex(0xEFE8D8) },
            ColorStop { threshold: 0.62, color: Color::from_hex(0xC0B29A) },
            ColorStop { threshold: 0.80, color: Color::from_hex(0x1F2329) },
            ColorStop { threshold: 1.00, color: Color::from_hex(0x0A0C10) },
        ],
        color_hardness: 0.0,
        lighting_enabled: false,
        light_dir: normalize(&Vec3::new(0.0, 1.0, 0.0)),
        light_min: 1.0, light_max: 1.0,
        alpha_mode: AlphaMode::Opaque,
    };

    // 5. URANO
    let uranus_shader = ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::BandedGas,
            scale: 1.0, octaves: 3, lacunarity: 2.0, gain: 0.55,
            cell_size: 0.35, w1: 1.0, w2: 1.0, w3: 1.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,
            animate_time: false, time_speed: 0.0,
            animate_spin: false,  spin_speed: 0.0,

            ring_swirl_amp: 0.0,
            ring_swirl_freq: 8.0,

            band_frequency: 3.0,
            band_contrast: 0.5,
            lat_shear: 0.05,
            turb_scale: 4.0,
            turb_octaves: 3,
            turb_lacunarity: 2.0,
            turb_gain: 0.5,
            turb_amp: 0.15,
            flow: FlowParams::default(),
        },
        color_stops: vec![
            ColorStop { threshold: 0.00, color: Color::from_hex(0x0A2F4F) },
            ColorStop { threshold: 0.30, color: Color::from_hex(0x145D7A) },
            ColorStop { threshold: 0.60, color: Color::from_hex(0x42A8C8) },
            ColorStop { threshold: 0.85, color: Color::from_hex(0x7FD8E6) },
            ColorStop { threshold: 1.00, color: Color::from_hex(0xC5F6FF) },
        ],
        color_hardness: 0.25,
        lighting_enabled: true,
        light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)),
        light_min: 0.45,
        light_max: 1.1,
        alpha_mode: AlphaMode::Opaque,
    };
    let uranus_rings_shader = ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::RadialGradient {
                inner: 0.82,   // anillo más estrecho, más “fino”
                outer: 1.0,
                invert: false,
                bias: 0.08,
                gamma: 1.1,
            },
            scale: 1.0, octaves: 1, lacunarity: 2.0, gain: 0.5,
            cell_size: 1.0, w1: 1.0, w2: 0.0, w3: 0.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,
            animate_time: false, time_speed: 0.0,
            animate_spin: true, spin_speed: 1.6, // un pelín más rápido

            // pequeñísimo swirl para que no sea perfectamente estático
            ring_swirl_amp: 0.03,
            ring_swirl_freq: 14.0,

            band_frequency: 0.0,
            band_contrast: 0.0,
            lat_shear: 0.0,
            turb_scale: 0.0,
            turb_octaves: 0,
            turb_lacunarity: 0.0,
            turb_gain: 0.0,
            turb_amp: 0.0,
            flow: FlowParams::default(),
        },
        // Alternamos oscuro / azul / cian con thresholds muy pegados
        // para que algunas franjas sean bien delgaditas
        color_stops: vec![
            ColorStop { threshold: 0.00, color: Color::from_hex(0x02070B) }, // casi negro azulado
            ColorStop { threshold: 0.10, color: Color::from_hex(0x0E2230) }, // azul muy oscuro
            ColorStop { threshold: 0.14, color: Color::from_hex(0x1C5670) }, // franja azul más clara (delgada)
            ColorStop { threshold: 0.18, color: Color::from_hex(0x050A10) }, // gap oscuro
            ColorStop { threshold: 0.24, color: Color::from_hex(0x2E7E9E) }, // teal
            ColorStop { threshold: 0.28, color: Color::from_hex(0x08141C) }, // gap
            ColorStop { threshold: 0.34, color: Color::from_hex(0x58B4D3) }, // cian claro
            ColorStop { threshold: 0.38, color: Color::from_hex(0x091822) }, // gap
            ColorStop { threshold: 0.46, color: Color::from_hex(0x8ADBF0) }, // franja muy clara
            ColorStop { threshold: 0.52, color: Color::from_hex(0x0A1820) }, // gap
            ColorStop { threshold: 0.60, color: Color::from_hex(0x5DB3CD) },
            ColorStop { threshold: 0.70, color: Color::from_hex(0x0A1218) },
            ColorStop { threshold: 0.85, color: Color::from_hex(0x9FE5F7) }, // casi blanco azulado
            ColorStop { threshold: 1.00, color: Color::from_hex(0x02060A) }, // fondo
        ],
        color_hardness: 0.30, // más dureza para que se noten las franjas finas
        lighting_enabled: false,
        light_dir: normalize(&Vec3::new(0.0, 1.0, 0.0)),
        light_min: 1.0,
        light_max: 1.0,
        alpha_mode: AlphaMode::Opaque,
    };

    // SHADER DE LUNAS
    let moon_shader = ProceduralLayerShader {
        noise: NoiseParams { kind: NoiseType::Voronoi, scale: 2.6, octaves: 3, lacunarity: 2.0, gain: 0.5, cell_size: 0.4, w1:1.0, w2:1.0, w3:1.0, w4:0.0, dist: VoronoiDistance::Euclidean, animate_time: false, time_speed: 0.0, animate_spin: false, spin_speed: 0.0, ring_swirl_amp: 0.0, ring_swirl_freq: 0.0, band_frequency: 0.0, band_contrast: 0.0, lat_shear: 0.0, turb_scale: 0.0, turb_octaves: 0, turb_lacunarity: 0.0, turb_gain: 0.0, turb_amp: 0.0, flow: FlowParams::default() },
        color_stops: vec![ ColorStop{threshold:0.25, color:Color::from_hex(0x5E5347)}, ColorStop{threshold:0.85, color:Color::from_hex(0xCBB79F)} ],
        color_hardness: 0.25, lighting_enabled: true, light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)), light_min: 0.35, light_max: 1.05, alpha_mode: AlphaMode::Opaque
    };

    // ------------------- VECTORES DE DATOS -------------------

    let planet_shaders: Vec<Box<dyn FragmentShader + Send + Sync>> = vec![
        Box::new(sun_shader),      // 0
        Box::new(terrain_shader),  // 1
        Box::new(mars_shader),     // 2
        Box::new(jupiter_shader),  // 3
        Box::new(saturn_shader),   // 4
        Box::new(uranus_shader),   // 5
    ];

    let ring_shaders: Vec<Option<Box<dyn FragmentShader + Send + Sync>>> = vec![
        None,
        None,
        None,
        None,
        Some(Box::new(saturn_rings_shader)), // Saturno
        Some(Box::new(uranus_rings_shader)), // Urano
    ];

    let cloud_shaders: Vec<Option<Box<dyn FragmentShader + Send + Sync>>> = vec![
        None,
        Some(Box::new(clouds_shader)),      // Tierra
        Some(Box::new(mars_clouds_shader)), // Marte
        None,
        None,
        None,
    ];

    // ------------------- CONFIGURACIÓN PLANETAS -------------------
    // He aumentado la distancia entre planetas (dist_from_sun) como pediste.
    let planets = vec![
        // 0. SOL
        PlanetConfig { dist_from_sun: 0.0,   orbit_speed: 0.0,  orbit_offset: 0.0, scale: 20.0, rotation_speed: 0.05, shader_index: 0, has_rings: false, ring_shader_index: None, ring_tilt: Vec3::new(0.0,0.0,0.0), has_atmosphere: false, atmosphere_shader_index: None },
        // 1. TIERRA
        PlanetConfig { dist_from_sun: 120.0,  orbit_speed: 0.4,  orbit_offset: 0.0, scale: 5.0,  rotation_speed: 0.5,  shader_index: 1, has_rings: false, ring_shader_index: None, ring_tilt: Vec3::new(0.0,0.0,0.0), has_atmosphere: true,  atmosphere_shader_index: Some(1) },
        // 2. MARTE
        PlanetConfig { dist_from_sun: 180.0,  orbit_speed: 0.3,  orbit_offset: 2.0, scale: 4.0,  rotation_speed: 0.4,  shader_index: 2, has_rings: false, ring_shader_index: None, ring_tilt: Vec3::new(0.0,0.0,0.0), has_atmosphere: true,  atmosphere_shader_index: Some(2) },
        // 3. JUPITER
        PlanetConfig { dist_from_sun: 280.0, orbit_speed: 0.2,  orbit_offset: 4.0, scale: 14.0, rotation_speed: 0.3,  shader_index: 3, has_rings: false, ring_shader_index: None, ring_tilt: Vec3::new(0.0,0.0,0.0), has_atmosphere: false, atmosphere_shader_index: None },
        // 4. SATURNO
        PlanetConfig { dist_from_sun: 400.0, orbit_speed: 0.15, orbit_offset: 1.0, scale: 11.0, rotation_speed: 0.2,  shader_index: 4, has_rings: true,  ring_shader_index: Some(4), ring_tilt: Vec3::new(0.3, 0.0, 0.1), has_atmosphere: false, atmosphere_shader_index: None },
        // 5. URANO
        PlanetConfig { dist_from_sun: 550.0, orbit_speed: 0.1,  orbit_offset: 5.5, scale: 9.0,  rotation_speed: 0.1,  shader_index: 5, has_rings: true,  ring_shader_index: Some(5), ring_tilt: Vec3::new(1.3, 0.0, 0.0), has_atmosphere: false, atmosphere_shader_index: None },
    ];

    // ------------------- LUNAS -------------------
    // Creamos instancias frescas del shader para cada luna clonando el objeto (pero es un Box, asi que clonamos los params y rehacemos el box)
    // Para simplificar en el refactor, usaremos el mismo shader genérico.

    let moons = vec![
        // Luna de la Tierra (index 1) - Muy cerca
        Moon { obj: moon_obj.clone(), scale_rel: 0.25, orbit_px: 2.0, orbit_speed: 1.5, phase0: 0.0, tilt: Vec3::new(0.05, 0.0, 0.05), shader: Box::new(moon_shader), seed: 8888, parent_index: 1 },
        // Luna de Júpiter (index 3) - Un poco mas lejos
        Moon { obj: moon_obj.clone(), scale_rel: 0.20, orbit_px: 3.0, orbit_speed: 0.8, phase0: 2.0, tilt: Vec3::new(0.0, 0.0, 0.0), shader: Box::new(
             ProceduralLayerShader {
                noise: NoiseParams { kind: NoiseType::Voronoi, scale: 3.0, octaves: 3, lacunarity: 2.0, gain: 0.5, cell_size: 0.4, w1:1.0, w2:1.0, w3:1.0, w4:0.0, dist: VoronoiDistance::Euclidean, animate_time: false, time_speed: 0.0, animate_spin: false, spin_speed: 0.0, ring_swirl_amp: 0.0, ring_swirl_freq: 0.0, band_frequency: 0.0, band_contrast: 0.0, lat_shear: 0.0, turb_scale: 0.0, turb_octaves: 0, turb_lacunarity: 0.0, turb_gain: 0.0, turb_amp: 0.0, flow: FlowParams::default() },
                color_stops: vec![ ColorStop{threshold:0.25, color:Color::from_hex(0xAAAAAA)}, ColorStop{threshold:0.85, color:Color::from_hex(0xFFFFFF)} ],
                color_hardness: 0.25, lighting_enabled: true, light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)), light_min: 0.35, light_max: 1.05, alpha_mode: AlphaMode::Opaque
            }
        ), seed: 7777, parent_index: 3 }
    ];

    SceneData {
        planets,
        moons,
        planet_shaders,
        ring_shaders,
        cloud_shaders,
    }
}