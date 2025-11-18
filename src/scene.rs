// scene.rs - Configuración completa del sistema solar

use nalgebra_glm::{Vec3, Mat4};
use nalgebra_glm::normalize;
use crate::obj::Obj;
use crate::shaders::{
    ProceduralLayerShader, NoiseParams, NoiseType, VoronoiDistance,
    ColorStop, AlphaMode, FlowParams, FragmentShader,
};
use crate::color::Color;

// ===================== Estructuras de planetas y lunas =====================

pub struct Planet {
    pub obj: Obj,
    pub scale: f32,
    pub orbit_distance: f32,
    pub orbit_speed: f32,
    pub phase0: f32,
    pub rotation_speed: f32,
    pub shader: Box<dyn FragmentShader + Send + Sync>,
    pub cloud_shader: Option<Box<dyn FragmentShader + Send + Sync>>,
    pub seed: i32,
    pub moons: Vec<Moon>,
}

impl Planet {
    pub fn model_matrix(&self, sun_translation: Vec3, t: f32) -> Mat4 {
        let angle = self.phase0 + t * self.orbit_speed;
        let dx = angle.cos() * self.orbit_distance;
        let dz = angle.sin() * self.orbit_distance;

        let translation = Vec3::new(
            sun_translation.x + dx,
            sun_translation.y,
            sun_translation.z + dz
        );

        let rotation = Vec3::new(0.0, t * self.rotation_speed, 0.0);
        crate::create_model_matrix(translation, self.scale, rotation)
    }

    pub fn translation(&self, sun_translation: Vec3, t: f32) -> Vec3 {
        let angle = self.phase0 + t * self.orbit_speed;
        let dx = angle.cos() * self.orbit_distance;
        let dz = angle.sin() * self.orbit_distance;
        
        Vec3::new(
            sun_translation.x + dx,
            sun_translation.y,
            sun_translation.z + dz
        )
    }
}

pub struct Moon {
    pub obj: Obj,
    pub scale_rel: f32,
    pub orbit_px: f32,
    pub orbit_depth_px: f32,
    pub orbit_speed: f32,
    pub phase0: f32,
    pub tilt: Vec3,
    pub shader: Box<dyn FragmentShader + Send + Sync>,
    pub seed: i32,
}

impl Moon {
    pub fn model_matrix(&self, planet_translation: Vec3, planet_scale: f32, t: f32) -> Mat4 {
        let angle = self.phase0 + t * self.orbit_speed;
        let dx = angle.cos() * self.orbit_px;
        let dz = angle.sin() * self.orbit_depth_px;

        let tr = Vec3::new(
            planet_translation.x + dx,
            planet_translation.y + 0.0,
            planet_translation.z + dz
        );

        let spin = Vec3::new(self.tilt.x, self.tilt.y + t * 0.35, self.tilt.z);
        crate::create_model_matrix(tr, planet_scale * self.scale_rel, spin)
    }
}

// ===================== Configuración de shaders =====================

pub fn create_terrain_shader() -> ProceduralLayerShader {
    ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::Perlin,
            scale: 3.0,
            octaves: 5,
            lacunarity: 2.0,
            gain: 0.5,
            cell_size: 0.35,
            w1: 1.0, w2: 0.0, w3: 0.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,
            animate_time: false,
            time_speed: 1.0,
            animate_spin: false,
            spin_speed: 0.0,
            ring_swirl_amp: 0.0,
            ring_swirl_freq: 8.0,
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
            ColorStop { threshold: 0.35, color: Color::from_hex(0x1B3494) },
            ColorStop { threshold: 0.48, color: Color::from_hex(0x203FB0) },
            ColorStop { threshold: 0.50, color: Color::from_hex(0x4B87DB) },
            ColorStop { threshold: 0.51, color: Color::from_hex(0xA4957F) },
            ColorStop { threshold: 0.52, color: Color::from_hex(0x88B04B) },
            ColorStop { threshold: 0.60, color: Color::from_hex(0x668736) },
            ColorStop { threshold: 0.70, color: Color::from_hex(0x597A2A) },
        ],
        color_hardness: 0.25,
        lighting_enabled: true,
        light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)),
        light_min: 0.35,
        light_max: 1.0,
        alpha_mode: AlphaMode::Opaque,
    }
}

pub fn create_clouds_shader() -> ProceduralLayerShader {
    ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::Value,
            scale: 5.0,
            octaves: 3,
            lacunarity: 2.0,
            gain: 0.5,
            cell_size: 0.25,
            w1: 1.0, w2: 0.0, w3: 0.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,
            animate_time: true,
            time_speed: 0.1,
            animate_spin: true,
            spin_speed: 0.05,
            ring_swirl_amp: 0.0,
            ring_swirl_freq: 8.0,
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
            ColorStop { threshold: 0.0, color: Color::from_hex(0xEDEDED) },
            ColorStop { threshold: 1.0, color: Color::from_hex(0xFFFFFF) },
        ],
        color_hardness: 0.0,
        lighting_enabled: true,
        light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)),
        light_min: 0.8,
        light_max: 1.0,
        alpha_mode: AlphaMode::Threshold {
            threshold: 0.50,
            sharpness: 6.0,
            coverage_bias: 0.05,
            invert: false,
        },
    }
}

pub fn create_gas_giant_shader() -> ProceduralLayerShader {
    ProceduralLayerShader {
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
        color_stops: vec![
            ColorStop { threshold: 0.00, color: Color::from_hex(0x734D1E) },
            ColorStop { threshold: 0.20, color: Color::from_hex(0x6E5034) },
            ColorStop { threshold: 0.40, color: Color::from_hex(0xB58C5A) },
            ColorStop { threshold: 0.60, color: Color::from_hex(0xD9B98B) },
            ColorStop { threshold: 0.80, color: Color::from_hex(0xA57442) },
            ColorStop { threshold: 1.00, color: Color::from_hex(0xE4D1B5) },
        ],
        color_hardness: 0.35,
        lighting_enabled: true,
        light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)),
        light_min: 0.45,
        light_max: 1.05,
        alpha_mode: AlphaMode::Opaque,
    }
}

pub fn create_mars_shader() -> ProceduralLayerShader {
    ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::Perlin,
            scale: 4.0,
            octaves: 4,
            lacunarity: 2.0,
            gain: 0.5,
            cell_size: 0.40,
            w1: 1.0, w2: 0.0, w3: 0.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,
            animate_time: false,
            time_speed: 0.0,
            animate_spin: false,
            spin_speed: 0.0,
            ring_swirl_amp: 0.0,
            ring_swirl_freq: 8.0,
            band_frequency: 0.0,
            band_contrast: 0.0,
            lat_shear: 0.0,
            turb_scale: 6.0,
            turb_octaves: 3,
            turb_lacunarity: 2.0,
            turb_gain: 0.5,
            turb_amp: 0.2,
            flow: FlowParams::default(),
        },
        color_stops: vec![
            ColorStop { threshold: 0.10, color: Color::from_hex(0x3D1F1A) },
            ColorStop { threshold: 0.35, color: Color::from_hex(0x6B2F26) },
            ColorStop { threshold: 0.60, color: Color::from_hex(0xA94432) },
            ColorStop { threshold: 0.85, color: Color::from_hex(0xD27A4A) },
            ColorStop { threshold: 1.00, color: Color::from_hex(0xF3C9A2) },
        ],
        color_hardness: 0.35,
        lighting_enabled: true,
        light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)),
        light_min: 0.35,
        light_max: 1.05,
        alpha_mode: AlphaMode::Opaque,
    }
}

pub fn create_mars_clouds_shader() -> ProceduralLayerShader {
    ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::Value,
            scale: 2.0,
            octaves: 3,
            lacunarity: 2.0,
            gain: 0.5,
            cell_size: 0.25,
            w1: 1.0, w2: 0.0, w3: 0.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,
            animate_time: true,
            time_speed: 0.05,
            animate_spin: true,
            spin_speed: 0.02,
            ring_swirl_amp: 0.0,
            ring_swirl_freq: 8.0,
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
            ColorStop { threshold: 0.0, color: Color::from_hex(0xEDEDED) },
            ColorStop { threshold: 1.0, color: Color::from_hex(0xFFFFFF) },
        ],
        color_hardness: 0.0,
        lighting_enabled: true,
        light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)),
        light_min: 0.8,
        light_max: 1.0,
        alpha_mode: AlphaMode::Threshold {
            threshold: 0.60,
            sharpness: 6.0,
            coverage_bias: 0.05,
            invert: false,
        },
    }
}

pub fn create_uranus_shader() -> ProceduralLayerShader {
    ProceduralLayerShader {
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
    }
}

pub fn create_saturn_shader() -> ProceduralLayerShader {
    ProceduralLayerShader {
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
            ColorStop { threshold: 0.00, color: Color::from_hex(0xEDD198) },
            ColorStop { threshold: 0.18, color: Color::from_hex(0xB48D5A) },
            ColorStop { threshold: 0.32, color: Color::from_hex(0xD5BE8A) },
            ColorStop { threshold: 0.36, color: Color::from_hex(0xE5A95C) },
            ColorStop { threshold: 0.44, color: Color::from_hex(0xEFD49C) },
            ColorStop { threshold: 0.58, color: Color::from_hex(0xF4E0B6) },
            ColorStop { threshold: 0.72, color: Color::from_hex(0xF6E6C4) },
            ColorStop { threshold: 0.86, color: Color::from_hex(0xFBF0D9) },
            ColorStop { threshold: 1.00, color: Color::from_hex(0xFFF9E6) },
        ],
        color_hardness: 0.24,
        lighting_enabled: true,
        light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)),
        light_min: 0.50,
        light_max: 1.08,
        alpha_mode: AlphaMode::Opaque,
    }
}

pub fn create_star_shader() -> ProceduralLayerShader {
    ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::BandedGas,
            scale: 1.0,
            octaves: 16,
            lacunarity: 2.2,
            gain: 0.52,
            cell_size: 0.35,
            w1: 1.0, w2: 1.0, w3: 1.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,
            animate_time: false,
            time_speed: 0.0,
            animate_spin: true,
            spin_speed: 0.12,
            ring_swirl_amp: 0.0,
            ring_swirl_freq: 8.0,
            band_frequency: 20.0,
            band_contrast: 0.05,
            lat_shear: 0.05,
            turb_scale: 8.0,
            turb_octaves: 4,
            turb_lacunarity: 2.0,
            turb_gain: 0.55,
            turb_amp: 1.4,
            flow: FlowParams {
                enabled: true,
                flow_scale: 2.5,
                strength: 0.09,
                time_speed: 0.7,
                jets_base_speed: 0.04,
                jets_frequency: 5.0,
                phase_amp: 2.0,
            },
        },
        color_stops: vec![
            ColorStop { threshold: 0.00, color: Color::from_hex(0xA33600) },
            ColorStop { threshold: 0.25, color: Color::from_hex(0x7A1400) },
            ColorStop { threshold: 0.50, color: Color::from_hex(0xE04800) },
            ColorStop { threshold: 0.75, color: Color::from_hex(0xFFC640) },
            ColorStop { threshold: 1.00, color: Color::from_hex(0xFFF9D0) },
        ],
        color_hardness: 0.15,
        lighting_enabled: false,
        light_dir: normalize(&Vec3::new(0.0, 1.0, 0.0)),
        light_min: 1.0,
        light_max: 1.0,
        alpha_mode: AlphaMode::Opaque,
    }
}

pub fn create_saturn_rings_shader() -> ProceduralLayerShader {
    ProceduralLayerShader {
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
    }
}

pub fn create_uranus_rings_shader() -> ProceduralLayerShader {
    ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::RadialGradient {
                inner: 0.82,
                outer: 1.0,
                invert: false,
                bias: 0.08,
                gamma: 1.1,
            },
            scale: 1.0, octaves: 1, lacunarity: 2.0, gain: 0.5,
            cell_size: 1.0, w1: 1.0, w2: 0.0, w3: 0.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,
            animate_time: false, time_speed: 0.0,
            animate_spin: true, spin_speed: 1.6,
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
        color_stops: vec![
            ColorStop { threshold: 0.00, color: Color::from_hex(0x02070B) },
            ColorStop { threshold: 0.10, color: Color::from_hex(0x0E2230) },
            ColorStop { threshold: 0.14, color: Color::from_hex(0x1C5670) },
            ColorStop { threshold: 0.18, color: Color::from_hex(0x050A10) },
            ColorStop { threshold: 0.24, color: Color::from_hex(0x2E7E9E) },
            ColorStop { threshold: 0.28, color: Color::from_hex(0x08141C) },
            ColorStop { threshold: 0.34, color: Color::from_hex(0x58B4D3) },
            ColorStop { threshold: 0.38, color: Color::from_hex(0x091822) },
            ColorStop { threshold: 0.46, color: Color::from_hex(0x8ADBF0) },
            ColorStop { threshold: 0.52, color: Color::from_hex(0x0A1820) },
            ColorStop { threshold: 0.60, color: Color::from_hex(0x5DB3CD) },
            ColorStop { threshold: 0.70, color: Color::from_hex(0x0A1218) },
            ColorStop { threshold: 0.85, color: Color::from_hex(0x9FE5F7) },
            ColorStop { threshold: 1.00, color: Color::from_hex(0x02060A) },
        ],
        color_hardness: 0.30,
        lighting_enabled: false,
        light_dir: normalize(&Vec3::new(0.0, 1.0, 0.0)),
        light_min: 1.0,
        light_max: 1.0,
        alpha_mode: AlphaMode::Opaque,
    }
}

fn create_moon_shader_grey() -> ProceduralLayerShader {
    ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::Perlin,
            scale: 4.0, octaves: 4, lacunarity: 2.0, gain: 0.5,
            cell_size: 0.35, w1: 1.0, w2: 0.0, w3: 0.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,
            animate_time: false, time_speed: 0.0,
            animate_spin: false, spin_speed: 0.0,
            ring_swirl_amp: 0.0,
            ring_swirl_freq: 8.0,
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
            ColorStop { threshold: 0.30, color: Color::from_hex(0x4D4D4D) },
            ColorStop { threshold: 0.55, color: Color::from_hex(0x7A7A7A) },
            ColorStop { threshold: 0.80, color: Color::from_hex(0xB5B5B5) },
        ],
        color_hardness: 0.35,
        lighting_enabled: true,
        light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)),
        light_min: 0.35,
        light_max: 1.05,
        alpha_mode: AlphaMode::Opaque,
    }
}

fn create_moon_shader_voronoi() -> ProceduralLayerShader {
    ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::Voronoi,
            scale: 2.6, octaves: 3, lacunarity: 2.0, gain: 0.5,
            cell_size: 0.40, w1: 1.0, w2: 1.0, w3: 1.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,
            animate_time: false, time_speed: 0.0,
            animate_spin: false, spin_speed: 0.0,
            ring_swirl_amp: 0.0,
            ring_swirl_freq: 8.0,
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
            ColorStop { threshold: 0.25, color: Color::from_hex(0x5E5347) },
            ColorStop { threshold: 0.55, color: Color::from_hex(0x8C7B68) },
            ColorStop { threshold: 0.85, color: Color::from_hex(0xCBB79F) },
        ],
        color_hardness: 0.25,
        lighting_enabled: true,
        light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)),
        light_min: 0.35,
        light_max: 1.05,
        alpha_mode: AlphaMode::Opaque,
    }
}

fn create_moon_shader_rocky() -> ProceduralLayerShader {
    ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::Perlin,
            scale: 3.5, octaves: 5, lacunarity: 2.2, gain: 0.5,
            cell_size: 0.30, w1: 1.0, w2: 0.0, w3: 0.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,
            animate_time: false, time_speed: 0.0,
            animate_spin: false, spin_speed: 0.0,
            ring_swirl_amp: 0.0,
            ring_swirl_freq: 8.0,
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
            ColorStop { threshold: 0.20, color: Color::from_hex(0x3A2F28) },
            ColorStop { threshold: 0.50, color: Color::from_hex(0x6B5E52) },
            ColorStop { threshold: 0.80, color: Color::from_hex(0x9E8D7C) },
        ],
        color_hardness: 0.30,
        lighting_enabled: true,
        light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)),
        light_min: 0.35,
        light_max: 1.05,
        alpha_mode: AlphaMode::Opaque,
    }
}

fn create_moon_shader_icy() -> ProceduralLayerShader {
    ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::Value,
            scale: 3.0, octaves: 4, lacunarity: 2.0, gain: 0.5,
            cell_size: 0.35, w1: 1.0, w2: 0.0, w3: 0.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,
            animate_time: false, time_speed: 0.0,
            animate_spin: false, spin_speed: 0.0,
            ring_swirl_amp: 0.0,
            ring_swirl_freq: 8.0,
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
            ColorStop { threshold: 0.30, color: Color::from_hex(0xB8D4E8) },
            ColorStop { threshold: 0.60, color: Color::from_hex(0xD9E8F5) },
            ColorStop { threshold: 0.85, color: Color::from_hex(0xF0F8FF) },
        ],
        color_hardness: 0.20,
        lighting_enabled: true,
        light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)),
        light_min: 0.40,
        light_max: 1.1,
        alpha_mode: AlphaMode::Opaque,
    }
}

// ===================== Configuración de la escena completa =====================

pub struct SolarSystem {
    pub sun_shader: ProceduralLayerShader,
    pub saturn_rings_shader: ProceduralLayerShader,
    pub uranus_rings_shader: ProceduralLayerShader,
    pub planets: Vec<Planet>,
}

pub fn create_solar_system(planet_obj: Obj, moon_obj: Obj) -> SolarSystem {
    let base_scale = 18.0; // Escala base para planetas

    let planets = vec![
        // Tierra con 2 lunas
        Planet {
            obj: planet_obj.clone(),
            scale: base_scale * 1.0,
            orbit_distance: 250.0,
            orbit_speed: 0.15,
            phase0: 0.0,
            rotation_speed: 0.20,
            shader: Box::new(create_terrain_shader()),
            cloud_shader: Some(Box::new(create_clouds_shader())),
            seed: 1337,
            moons: vec![
                Moon {
                    obj: moon_obj.clone(),
                    scale_rel: 0.27,
                    orbit_px: 45.0,
                    orbit_depth_px: 35.0,
                    orbit_speed: 0.8,
                    phase0: 0.0,
                    tilt: Vec3::new(0.05, 0.0, 0.05),
                    shader: Box::new(create_moon_shader_grey()),
                    seed: 8888,
                },
                Moon {
                    obj: moon_obj.clone(),
                    scale_rel: 0.20,
                    orbit_px: 65.0,
                    orbit_depth_px: 55.0,
                    orbit_speed: 0.5,
                    phase0: 3.14,
                    tilt: Vec3::new(0.15, 0.0, -0.05),
                    shader: Box::new(create_moon_shader_voronoi()),
                    seed: 9999,
                },
            ],
        },
        // Marte con 2 pequeñas lunas (Fobos y Deimos)
        Planet {
            obj: planet_obj.clone(),
            scale: base_scale * 0.53,
            orbit_distance: 340.0,
            orbit_speed: 0.10,
            phase0: 2.5,
            rotation_speed: 0.18,
            shader: Box::new(create_mars_shader()),
            cloud_shader: Some(Box::new(create_mars_clouds_shader())),
            seed: 3003,
            moons: vec![
                Moon {
                    obj: moon_obj.clone(),
                    scale_rel: 0.15,
                    orbit_px: 30.0,
                    orbit_depth_px: 25.0,
                    orbit_speed: 1.2,
                    phase0: 0.5,
                    tilt: Vec3::new(0.02, 0.0, 0.03),
                    shader: Box::new(create_moon_shader_rocky()),
                    seed: 7777,
                },
                Moon {
                    obj: moon_obj.clone(),
                    scale_rel: 0.12,
                    orbit_px: 42.0,
                    orbit_depth_px: 38.0,
                    orbit_speed: 0.85,
                    phase0: 2.0,
                    tilt: Vec3::new(0.08, 0.0, -0.02),
                    shader: Box::new(create_moon_shader_grey()),
                    seed: 7778,
                },
            ],
        },
        // Júpiter con 4 lunas galileanas
        Planet {
            obj: planet_obj.clone(),
            scale: base_scale * 2.2,
            orbit_distance: 500.0,
            orbit_speed: 0.06,
            phase0: 1.2,
            rotation_speed: 0.35,
            shader: Box::new(create_gas_giant_shader()),
            cloud_shader: None,
            seed: 2001,
            moons: vec![
                Moon {
                    obj: moon_obj.clone(),
                    scale_rel: 0.18,
                    orbit_px: 80.0,
                    orbit_depth_px: 60.0,
                    orbit_speed: 0.9,
                    phase0: 0.0,
                    tilt: Vec3::new(0.03, 0.0, 0.02),
                    shader: Box::new(create_moon_shader_voronoi()),
                    seed: 6661,
                },
                Moon {
                    obj: moon_obj.clone(),
                    scale_rel: 0.16,
                    orbit_px: 100.0,
                    orbit_depth_px: 85.0,
                    orbit_speed: 0.65,
                    phase0: 1.5,
                    tilt: Vec3::new(0.05, 0.0, 0.04),
                    shader: Box::new(create_moon_shader_rocky()),
                    seed: 6662,
                },
                Moon {
                    obj: moon_obj.clone(),
                    scale_rel: 0.25,
                    orbit_px: 125.0,
                    orbit_depth_px: 110.0,
                    orbit_speed: 0.48,
                    phase0: 3.0,
                    tilt: Vec3::new(0.02, 0.0, -0.03),
                    shader: Box::new(create_moon_shader_icy()),
                    seed: 6663,
                },
                Moon {
                    obj: moon_obj.clone(),
                    scale_rel: 0.22,
                    orbit_px: 150.0,
                    orbit_depth_px: 135.0,
                    orbit_speed: 0.35,
                    phase0: 4.7,
                    tilt: Vec3::new(0.07, 0.0, 0.05),
                    shader: Box::new(create_moon_shader_grey()),
                    seed: 6664,
                },
            ],
        },
        // Saturno con 3 lunas principales
        Planet {
            obj: planet_obj.clone(),
            scale: base_scale * 1.9,
            orbit_distance: 720.0,
            orbit_speed: 0.04,
            phase0: 5.5,
            rotation_speed: 0.30,
            shader: Box::new(create_saturn_shader()),
            cloud_shader: None,
            seed: 5005,
            moons: vec![
                Moon {
                    obj: moon_obj.clone(),
                    scale_rel: 0.28,
                    orbit_px: 140.0,
                    orbit_depth_px: 120.0,
                    orbit_speed: 0.55,
                    phase0: 0.8,
                    tilt: Vec3::new(0.04, 0.0, 0.03),
                    shader: Box::new(create_moon_shader_icy()),
                    seed: 5551,
                },
                Moon {
                    obj: moon_obj.clone(),
                    scale_rel: 0.20,
                    orbit_px: 170.0,
                    orbit_depth_px: 150.0,
                    orbit_speed: 0.42,
                    phase0: 2.5,
                    tilt: Vec3::new(0.06, 0.0, -0.04),
                    shader: Box::new(create_moon_shader_rocky()),
                    seed: 5552,
                },
                Moon {
                    obj: moon_obj.clone(),
                    scale_rel: 0.17,
                    orbit_px: 195.0,
                    orbit_depth_px: 175.0,
                    orbit_speed: 0.35,
                    phase0: 4.2,
                    tilt: Vec3::new(0.03, 0.0, 0.05),
                    shader: Box::new(create_moon_shader_voronoi()),
                    seed: 5553,
                },
            ],
        },
        // Urano con 2 lunas
        Planet {
            obj: planet_obj.clone(),
            scale: base_scale * 1.4,
            orbit_distance: 880.0,
            orbit_speed: 0.03,
            phase0: 4.0,
            rotation_speed: 0.25,
            shader: Box::new(create_uranus_shader()),
            cloud_shader: None,
            seed: 4004,
            moons: vec![
                Moon {
                    obj: moon_obj.clone(),
                    scale_rel: 0.22,
                    orbit_px: 90.0,
                    orbit_depth_px: 75.0,
                    orbit_speed: 0.6,
                    phase0: 1.0,
                    tilt: Vec3::new(0.05, 0.0, 0.04),
                    shader: Box::new(create_moon_shader_grey()),
                    seed: 4441,
                },
                Moon {
                    obj: moon_obj.clone(),
                    scale_rel: 0.19,
                    orbit_px: 115.0,
                    orbit_depth_px: 100.0,
                    orbit_speed: 0.45,
                    phase0: 3.8,
                    tilt: Vec3::new(0.08, 0.0, -0.03),
                    shader: Box::new(create_moon_shader_icy()),
                    seed: 4442,
                },
            ],
        },
    ];

    SolarSystem {
        sun_shader: create_star_shader(),
        saturn_rings_shader: create_saturn_rings_shader(),
        uranus_rings_shader: create_uranus_rings_shader(),
        planets,
    }
}
