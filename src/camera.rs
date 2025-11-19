// camera.rs
use nalgebra_glm::{Mat4, Vec3, look_at};

pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y: f32,
    pub aspect: f32,
    pub z_near: f32,
    pub z_far: f32,
}

impl Camera {
    pub fn new(position: Vec3, target: Vec3, aspect: f32) -> Self {
        Self {
            position,
            target,
            up: Vec3::new(0.0, 1.0, 0.0),
            fov_y: std::f32::consts::FRAC_PI_3, // ~60°
            aspect,
            z_near: 0.1,
            z_far: 2000.0,
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        // Cámara tipo OpenGL: mira de position a target
        look_at(&self.position, &self.target, &self.up)
    }

    pub fn projection_matrix(&self) -> Mat4 {
        perspective(self.fov_y, self.aspect, self.z_near, self.z_far)
    }
}

/// Perspectiva estilo OpenGL RH, NDC z en [-1, 1]
pub fn perspective(fov_y: f32, aspect: f32, z_near: f32, z_far: f32) -> Mat4 {
    let f = 1.0 / (fov_y / 2.0).tan();
    let nf = 1.0 / (z_near - z_far);

    Mat4::new(
        f / aspect, 0.0, 0.0,                         0.0,
        0.0,        f,   0.0,                         0.0,
        0.0,        0.0, (z_far + z_near) * nf,      -1.0,
        0.0,        0.0, (2.0 * z_far * z_near) * nf, 0.0,
    )
}
