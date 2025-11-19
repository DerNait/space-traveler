use image::{DynamicImage, GenericImageView};
use crate::color::Color;

pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub data: Vec<Color>,
}

impl Texture {
    pub fn load(path: &str) -> Self {
        let img = image::open(path).expect(&format!("Failed to load texture: {}", path));
        let (width, height) = img.dimensions();
        let mut data = Vec::with_capacity((width * height) as usize);

        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x, y);
                // image devuelve Rgba<u8>, extraemos r, g, b
                data.push(Color::new(pixel[0], pixel[1], pixel[2]));
            }
        }

        Texture { width, height, data }
    }

    pub fn get_color(&self, u: f32, v: f32) -> Color {
        // Mapeo UV a coordenadas de pixel con clamp para seguridad
        let x = (u * (self.width as f32 - 1.0)).round() as u32;
        let y = (v * (self.height as f32 - 1.0)).round() as u32;

        let idx = (y * self.width + x) as usize;
        if idx < self.data.len() {
            self.data[idx]
        } else {
            Color::new(255, 0, 255) // Color de error (magenta)
        }
    }
}