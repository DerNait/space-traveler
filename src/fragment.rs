use nalgebra_glm::Vec2;
use crate::color::Color;

pub struct Fragment {
    pub position: Vec2,
    pub color: Color,
    pub depth: f32,
    pub alpha: f32, // Nuevo
}

impl Fragment {
    pub fn new(x: f32, y: f32, color: Color, depth: f32, alpha: f32) -> Self {
        Fragment {
            position: Vec2::new(x, y),
            color,
            depth,
            alpha,
        }
    }
}