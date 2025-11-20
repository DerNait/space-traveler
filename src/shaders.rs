// shaders.rs
use nalgebra_glm::{Vec2, Vec3, Mat3, Mat4, Vec4, dot};
use crate::vertex::Vertex;
use crate::color::Color;
use crate::noise;
use crate::texture::Texture;

pub struct Uniforms {
    pub model_matrix: Mat4,
    pub view_matrix:  Mat4,
    pub projection_matrix: Mat4,
    pub screen_width: f32,
    pub screen_height: f32,

    pub time: f32,
    pub seed: i32,
    // Parámetros para anillos
    pub ring_a: f32, // Radio mayor del modelo
    pub ring_b: f32, // Radio menor del modelo
    pub ring_plane_xy: bool, // true = plano XY, false = plano XZ
}

pub struct FragAttrs {
    pub obj_pos: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
    pub depth: f32,
}

pub trait FragmentShader {
    fn shade(&self, frag: &FragAttrs, uniforms: &Uniforms) -> (Color, f32);
}

pub fn vertex_shader(vertex: &Vertex, uniforms: &Uniforms) -> Option<Vertex> {
    let position = Vec4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0);
    let world_pos = uniforms.model_matrix * position;
    let view_pos = uniforms.view_matrix * world_pos;

    if view_pos.z >= 0.0 { return None; } // Culling

    let clip_pos = uniforms.projection_matrix * view_pos;
    let w = clip_pos.w;
    if w.abs() < 1e-5 { return None; }

    let ndc = clip_pos / w;
    let half_w = uniforms.screen_width * 0.5;
    let half_h = uniforms.screen_height * 0.5;
    let screen_x = ndc.x * half_w + half_w;
    let screen_y = -ndc.y * half_h + half_h;
    let depth = -view_pos.z;

    let model_mat3 = Mat3::new(
        uniforms.model_matrix[(0, 0)], uniforms.model_matrix[(0, 1)], uniforms.model_matrix[(0, 2)],
        uniforms.model_matrix[(1, 0)], uniforms.model_matrix[(1, 1)], uniforms.model_matrix[(1, 2)],
        uniforms.model_matrix[(2, 0)], uniforms.model_matrix[(2, 1)], uniforms.model_matrix[(2, 2)],
    );
    let normal_matrix = model_mat3.transpose().try_inverse().unwrap_or(Mat3::identity());
    let transformed_normal = (normal_matrix * vertex.normal).normalize();

    Some(Vertex {
        position: vertex.position,
        normal: vertex.normal,
        tex_coords: vertex.tex_coords,
        color: vertex.color,
        transformed_position: Vec3::new(screen_x, screen_y, depth),
        transformed_normal,
    })
}

// ==================== RUIDO ====================

#[derive(Clone, Copy)]
pub enum RingPlane { XY, XZ, YZ }

#[derive(Clone, Copy)]
pub enum NoiseType {
    Value, Perlin, Voronoi, BandedGas,
    RadialGradient { inner: f32, outer: f32, invert: bool, bias: f32, gamma: f32 },
    UVRadialGradient { center: Vec2, invert: bool, bias: f32, gamma: f32 },
    UVRadialGradientObj { plane: RingPlane, radius_max: f32, invert: bool, bias: f32, gamma: f32 },
}

#[derive(Clone, Copy)]
pub enum VoronoiDistance { Euclidean, Manhattan, Chebyshev }

#[derive(Clone)]
pub struct FlowParams {
    pub enabled: bool,
    pub flow_scale: f32, pub strength: f32, pub time_speed: f32,
    pub jets_base_speed: f32, pub jets_frequency: f32, pub phase_amp: f32,
}
impl Default for FlowParams {
    fn default() -> Self { Self { enabled: false, flow_scale: 2.0, strength: 0.0, time_speed: 0.0, jets_base_speed: 0.0, jets_frequency: 1.0, phase_amp: 0.0 } }
}

#[derive(Clone)]
pub struct NoiseParams {
    pub kind: NoiseType,
    pub scale: f32, pub octaves: u32, pub lacunarity: f32, pub gain: f32, pub cell_size: f32,
    pub w1: f32, pub w2: f32, pub w3: f32, pub w4: f32,
    pub dist: VoronoiDistance,
    pub animate_time: bool, pub time_speed: f32,
    pub animate_spin: bool, pub spin_speed: f32,
    pub ring_swirl_amp: f32, pub ring_swirl_freq: f32,
    pub band_frequency: f32, pub band_contrast: f32, pub lat_shear: f32,
    pub turb_scale: f32, pub turb_octaves: u32, pub turb_lacunarity: f32, pub turb_gain: f32, pub turb_amp: f32,
    pub flow: FlowParams,
}

#[derive(Clone)]
pub struct ColorStop { pub threshold: f32, pub color: Color }

#[derive(Clone)]
pub enum AlphaMode {
    Opaque,
    Threshold { threshold: f32, sharpness: f32, coverage_bias: f32, invert: bool },
    Constant(f32),
}

pub struct ProceduralLayerShader {
    pub noise: NoiseParams,
    pub color_stops: Vec<ColorStop>,
    pub color_hardness: f32,
    pub lighting_enabled: bool,
    pub light_dir: Vec3,
    pub light_min: f32,
    pub light_max: f32,
    pub alpha_mode: AlphaMode,
}

#[inline] fn rotate_y(v: Vec3, angle: f32) -> Vec3 {
    let (s, c) = angle.sin_cos();
    Vec3::new(c * v.x + s * v.z, v.y, -s * v.x + c * v.z)
}
#[inline] fn to_spherical(p: Vec3) -> (f32, f32) { (p.z.atan2(p.x), p.y.asin()) }
#[inline] fn wrap_pi(mut x: f32) -> f32 {
    while x <= -std::f32::consts::PI { x += 2.0 * std::f32::consts::PI; }
    while x >  std::f32::consts::PI  { x -= 2.0 * std::f32::consts::PI; }
    x
}
#[inline] fn shape_bias_gamma(mut x: f32, bias: f32, gamma: f32) -> f32 {
    let b = bias.clamp(0.0, 1.0);
    x = x * (1.0 - b) + x * x * b;
    x.powf(gamma.max(1e-6))
}

fn sample_color_stops(stops: &[ColorStop], x01: f32, hardness: f32) -> Color {
    if stops.is_empty() { return Color::from_hex(0xFF00FF); }
    let x = x01.clamp(0.0, 1.0);
    if stops.len() == 1 { return stops[0].color; }
    let mut i = 0usize;
    while i + 1 < stops.len() && x > stops[i + 1].threshold { i += 1; }
    if i + 1 >= stops.len() { return stops[stops.len() - 1].color; }
    let a = &stops[i];
    let b = &stops[i + 1];
    let span = (b.threshold - a.threshold).max(1e-6);
    let mut t = (x - a.threshold) / span;
    let h = hardness.clamp(0.0, 1.0);
    if h >= 0.999 { t = if t < 0.5 { 0.0 } else { 1.0 }; } 
    else { let ss = t * t * (3.0 - 2.0 * t); t = ss * (1.0 - h) + t * h; }
    Color::lerp(a.color, b.color, t)
}

fn eval_bands_core(lon: f32, lat: f32, uniforms: &Uniforms, p: &NoiseParams) -> f32 {
    let (sln, cln) = lon.sin_cos();
    let turb = {
        let f = noise::fbm_perlin_3proj(
            Vec3::new(cln, lat, sln), p.turb_scale.max(1e-6), 0.0, p.turb_octaves, p.turb_lacunarity, p.turb_gain, uniforms.seed ^ 0x4444
        );
        (f * 2.0 - 1.0)
    };
    let shear_term = p.lat_shear * sln;
    let phase = p.band_frequency * (lat + shear_term + p.turb_amp * turb);
    let s = 0.5 + 0.5 * phase.sin();
    let c = p.band_contrast.clamp(0.0, 1.0);
    let sc = s * s * (3.0 - 2.0 * s);
    sc * c + s * (1.0 - c)
}

fn eval_noise(p_obj_unit_in: Vec3, uniforms: &Uniforms, params: &NoiseParams) -> f32 {
    let spin = if params.animate_spin { uniforms.time * params.spin_speed } else { 0.0 };
    let mut p = rotate_y(p_obj_unit_in, spin);
    let (mut lon, mut lat) = to_spherical(p);

    if params.flow.enabled {
        let u = lon * params.flow.flow_scale;
        let v = lat * params.flow.flow_scale;
        let a = 2.0 * std::f32::consts::PI * noise::value_noise_2d(u, v, uniforms.seed ^ 0xABCD);
        let (sa, ca) = a.sin_cos();
        let lat_cos = lat.cos().abs().max(0.15);
        let dlon = ca * params.flow.strength * lat_cos;
        let dlat = sa * params.flow.strength * 0.5;
        let jets = params.flow.jets_base_speed * (params.flow.jets_frequency * lat).sin();
        let t = uniforms.time * params.flow.time_speed;
        let phase1 = t.fract();
        let phase2 = (phase1 + 0.5).fract();
        let flow_mix = ((phase1 - 0.5).abs() * 2.0).clamp(0.0, 1.0);
        let amp = params.flow.phase_amp.max(0.0);
        let lon_a = wrap_pi(lon + (dlon + jets) * phase1 * amp);
        let lat_a = (lat + dlat * phase1 * amp).clamp(-1.57, 1.57);
        let lon_b = wrap_pi(lon + (dlon + jets) * phase2 * amp);
        let lat_b = (lat + dlat * phase2 * amp).clamp(-1.57, 1.57);
        let bands_a = eval_bands_core(lon_a, lat_a, uniforms, params);
        let bands_b = eval_bands_core(lon_b, lat_b, uniforms, params);

        if let NoiseType::BandedGas = params.kind { return bands_a * (1.0 - flow_mix) + bands_b * flow_mix; }
        else {
            let lon_m = wrap_pi(lon_a * (1.0 - flow_mix) + lon_b * flow_mix);
            let lat_m = lat_a * (1.0 - flow_mix) + lat_b * flow_mix;
            let cl = lat_m.cos();
            p = Vec3::new(cl * lon_m.cos(), lat_m.sin(), cl * lon_m.sin());
            lon = lon_m; lat = lat_m;
        }
    }
    let t = if params.animate_time { uniforms.time * params.time_speed } else { 0.0 };
    match params.kind {
        NoiseType::Value => noise::fbm_value_3proj(p, params.scale, t, params.octaves, params.lacunarity, params.gain, uniforms.seed).clamp(0.0, 1.0),
        NoiseType::Perlin => noise::fbm_perlin_3proj(p, params.scale, t, params.octaves, params.lacunarity, params.gain, uniforms.seed ^ 0x9E37).clamp(0.0, 1.0),
        NoiseType::Voronoi => noise::voronoi_3proj(p, params.scale, params.cell_size, params.w1, params.w2, params.w3, params.w4, params.dist, uniforms.seed ^ 0xC2B2, t).clamp(0.0, 1.0),
        NoiseType::BandedGas => eval_bands_core(lon, lat, uniforms, params),
        NoiseType::RadialGradient { inner, outer, invert, bias, gamma } => {
            // Gradiente esférico base (no anillos)
            let r = (p.x*p.x + p.z*p.z).sqrt();
            let i = inner.max(1e-6);
            let o = outer.max(i + 1e-6);
            let mut rn = ((r - i) / (o - i)).clamp(0.0, 1.0);
            rn = shape_bias_gamma(rn, bias, gamma);
            if invert { rn } else { 1.0 - rn }
        }
        _ => 0.0, 
    }
}

impl FragmentShader for ProceduralLayerShader {
    fn shade(&self, frag: &FragAttrs, uniforms: &Uniforms) -> (Color, f32) {
        // Mantén ambas: posición de objeto cruda (para anillos/planos)
        // y posición unitaria (para ruidos esféricos en planetas).
        let p_obj = frag.obj_pos;
        let p_unit = {
            let mut q = p_obj;
            let m = q.magnitude();
            if m > 1e-6 { q /= m; }
            q
        };

        // Elegimos cómo evaluar 'n' según el tipo de ruido
        let n = match self.noise.kind {
            // Radial en OBJ: usa radio en el plano XZ/XY *sin normalizar el vector*
            NoiseType::RadialGradient { inner, outer, invert, bias, gamma } => {
                let (px, py, a, b) = if uniforms.ring_plane_xy {
                    (p_obj.x, p_obj.y, uniforms.ring_a.max(1e-6), uniforms.ring_b.max(1e-6))
                } else {
                    (p_obj.x, p_obj.z, uniforms.ring_a.max(1e-6), uniforms.ring_b.max(1e-6))
                };

                // Radio elíptico normalizado: mantiene el grosor del anillo QUIETO
                let r_ell = ((px / a).powi(2) + (py / b).powi(2)).sqrt();

                let i = inner.clamp(0.0, 0.999);
                let o = outer.clamp(i + 1e-6, 1.0);
                let mut rn = ((r_ell - i) / (o - i)).clamp(0.0, 1.0);

                // Curvatura radial; NO aplicar wobble aquí
                rn = shape_bias_gamma(rn, bias, gamma);
                let v = if invert { rn } else { 1.0 - rn };
                v.clamp(0.0, 1.0)
            }

            // Radial en UV nativos del mesh
            NoiseType::UVRadialGradient { center, invert, bias, gamma } => {
                let du = frag.uv.x - center.x;
                let dv = frag.uv.y - center.y;
                let max_r = 0.5_f32; // desde (0.5,0.5) al borde de la textura
                let mut rn = ((du*du + dv*dv).sqrt() / max_r).clamp(0.0, 1.0);
                rn = shape_bias_gamma(rn, bias, gamma);
                let v = if invert { rn } else { 1.0 - rn };
                v.clamp(0.0, 1.0)
            }
            // Radial en UV "sintéticos" generados desde OBJ (centrado en el planeta)
            NoiseType::UVRadialGradientObj { plane, radius_max, invert, bias, gamma } => {
                let (px, py) = match plane {
                    RingPlane::XY => (p_obj.x, p_obj.y),
                    RingPlane::XZ => (p_obj.x, p_obj.z),
                    RingPlane::YZ => (p_obj.y, p_obj.z),
                };
                let r = (px*px + py*py).sqrt();
                let mut rn = (r / radius_max.max(1e-6)).clamp(0.0, 1.0); // 0 centro, 1 borde
                rn = shape_bias_gamma(rn, bias, gamma);
                let v = if invert { rn } else { 1.0 - rn };
                v.clamp(0.0, 1.0)
            }
            // El resto de tipos se evalúan como antes (usando p_unit)
            _ => eval_noise(p_unit, uniforms, &self.noise),
        };

        // Color base por palette
        let mut col = sample_color_stops(&self.color_stops, n, self.color_hardness);

        // Tinte angular que GIRA (sin tocar el radio rn): solo para anillos RadialGradient
        if let NoiseType::RadialGradient { .. } = self.noise.kind {
            // Recalcula ángulo elíptico estable (usa x/a, y/b)
            let (px, py, a, b) = if uniforms.ring_plane_xy {
                (p_obj.x, p_obj.y, uniforms.ring_a.max(1e-6), uniforms.ring_b.max(1e-6))
            } else {
                (p_obj.x, p_obj.z, uniforms.ring_a.max(1e-6), uniforms.ring_b.max(1e-6))
            };
            let ex = (px / a);
            let ey = (py / b);
            let mut theta = ey.atan2(ex); // [-π, π]
            if self.noise.animate_spin {
                theta += uniforms.time * self.noise.spin_speed;
            }
            // u ∈ [0,1] desde el ángulo
            let u = 0.5 + theta / (2.0 * std::f32::consts::PI);
            // “grano” angular que viaja alrededor
            let grain = (u * self.noise.ring_swirl_freq * 2.0 * std::f32::consts::PI).sin() * 0.5 + 0.5; // [0,1]

            // Tinte sutil (no cambia geometría ni rn)
            let amp = self.noise.ring_swirl_amp; // sugerencia: 0.03..0.08
            let shade_lo = 1.0 - amp;
            let shade_hi = 1.0 + amp;
            let tint = shade_lo + (shade_hi - shade_lo) * grain; // ~ [1-amp, 1+amp]
            col = col * tint; // si tu Color no clampa, puedes añadir clamp por componente aquí
        }

        if self.lighting_enabled {
            let l = dot(&frag.normal, &self.light_dir).max(0.0);
            let mul = self.light_min + (self.light_max - self.light_min) * l;
            col = col * mul;
        }
        let a = alpha_from_noise(n, &self.alpha_mode);
        (col, a)
    }
}

fn alpha_from_noise(n: f32, mode: &AlphaMode) -> f32 {
    match mode {
        AlphaMode::Opaque => 1.0,
        AlphaMode::Constant(a) => a.clamp(0.0, 1.0),
        AlphaMode::Threshold { threshold, sharpness, coverage_bias, invert } => {
            let thr = (threshold - coverage_bias).clamp(0.0, 1.0);
            let k = (*sharpness).max(1.0);
            let a = ((n - thr) * k).clamp(0.0, 1.0);
            if *invert { 1.0 - a } else { a }
        }
    }
}

pub struct SkyboxShader<'a> {
    pub tex_pos_x: &'a Texture,
    pub tex_neg_x: &'a Texture,
    pub tex_pos_y: &'a Texture,
    pub tex_negy_y: &'a Texture,
    pub tex_pos_z: &'a Texture,
    pub tex_neg_z: &'a Texture,
}

impl<'a> FragmentShader for SkyboxShader<'a> {
    fn shade(&self, frag: &FragAttrs, _uniforms: &Uniforms) -> (Color, f32) {
        // Para un skybox, la "posición del objeto" interpolada es básicamente
        // el vector dirección desde el centro de la cámara hacia el píxel.
        let dir = frag.obj_pos; 
        
        let abs_x = dir.x.abs();
        let abs_y = dir.y.abs();
        let abs_z = dir.z.abs();

        let mut u = 0.0;
        let mut v = 0.0;
        let texture;

        // Algoritmo estándar de Cubemapping: determinar cara dominante
        if abs_x >= abs_y && abs_x >= abs_z {
            // Eje X dominante
            if dir.x > 0.0 {
                texture = self.tex_pos_x;
                u = -dir.z / abs_x;
                v = dir.y / abs_x;
            } else {
                texture = self.tex_neg_x;
                u = dir.z / abs_x;
                v = dir.y / abs_x;
            }
        } else if abs_y >= abs_x && abs_y >= abs_z {
            // Eje Y dominante
            if dir.y > 0.0 {
                texture = self.tex_pos_y;
                u = dir.x / abs_y;
                v = -dir.z / abs_y; // A veces hay que invertir Z dependiendo del asset
            } else {
                texture = self.tex_negy_y;
                u = dir.x / abs_y;
                v = dir.z / abs_y;
            }
        } else {
            // Eje Z dominante
            if dir.z > 0.0 {
                texture = self.tex_pos_z;
                u = dir.x / abs_z;
                v = dir.y / abs_z;
            } else {
                texture = self.tex_neg_z;
                u = -dir.x / abs_z;
                v = dir.y / abs_z;
            }
        }

        // Convertir rango [-1, 1] a [0, 1]
        u = (u + 1.0) * 0.5;
        v = (v + 1.0) * 0.5;
        
        // Invertir V si la imagen sale de cabeza (común en texturas)
        v = 1.0 - v; 

        let color = texture.get_color(u, v);
        
        // El skybox siempre es opaco (alpha 1.0)
        (color, 1.0)
    }
}

pub struct TextureShader<'a> {
    pub texture: &'a Texture,
    pub light_dir: Vec3,
    pub ambient: f32,   // Luz base para que no sea negro total en la sombra
    pub diffuse: f32,   // Intensidad de la luz directa
}

impl<'a> FragmentShader for TextureShader<'a> {
    fn shade(&self, frag: &FragAttrs, _uniforms: &Uniforms) -> (Color, f32) {
        // 1. Muestrear el color base de la textura usando las UVs interpoladas
        let base_color = self.texture.get_color(frag.uv.x, frag.uv.y);

        // 2. Calcular iluminación básica (Lambert)
        // El producto punto entre la normal y la luz
        let diffuse_intensity = dot(&frag.normal, &self.light_dir).max(0.0);
        
        // 3. Combinar ambiente + difusa
        let total_light = self.ambient + (self.diffuse * diffuse_intensity);
        
        // 4. Aplicar la luz al color
        let final_color = base_color * total_light;

        // Retornamos el color y alpha 1.0 (opaco)
        (final_color, 1.0)
    }
}