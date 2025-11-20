pub struct Framebuffer {
    pub width: usize,
    pub height: usize,
    pub buffer: Vec<u32>,
    pub zbuffer: Vec<f32>,
    background_color: u32,
    current_color: u32,
}

impl Framebuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Framebuffer {
            width,
            height,
            buffer: vec![0; width * height],
            zbuffer: vec![f32::INFINITY; width * height],
            background_color: 0x000000,
            current_color: 0xFFFFFF,
        }
    }

    pub fn clear(&mut self) {
        for pixel in self.buffer.iter_mut() {
            *pixel = self.background_color;
        }
        for depth in self.zbuffer.iter_mut() {
            *depth = f32::INFINITY;
        }
    }

    pub fn point(&mut self, x: usize, y: usize, depth: f32, color: u32) {
        if x < self.width && y < self.height {
            let index = y * self.width + x;
            if self.zbuffer[index] > depth {
                self.buffer[index] = color;
                self.zbuffer[index] = depth;
            }
        }
    }

    // ESTE ES EL MÉTODO QUE TE FALTABA
    pub fn draw_rgba(&mut self, x: usize, y: usize, depth: f32, color: u32, alpha: f32) {
        if x >= self.width || y >= self.height {
            return;
        }
        let index = y * self.width + x;

        // Z-Check
        if self.zbuffer[index] > depth {
            // Si es opaco, dibujamos directo
            if alpha >= 1.0 {
                self.buffer[index] = color;
                self.zbuffer[index] = depth;
            } else if alpha > 0.0 {
                // Blending básico
                let bg_color = self.buffer[index];
                
                // Extraer componentes del fondo
                let bg_r = ((bg_color >> 16) & 0xFF) as f32;
                let bg_g = ((bg_color >> 8) & 0xFF) as f32;
                let bg_b = (bg_color & 0xFF) as f32;

                // Extraer componentes del nuevo color
                let fg_r = ((color >> 16) & 0xFF) as f32;
                let fg_g = ((color >> 8) & 0xFF) as f32;
                let fg_b = (color & 0xFF) as f32;

                // Mezcla lineal
                let r = (fg_r * alpha + bg_r * (1.0 - alpha)) as u32;
                let g = (fg_g * alpha + bg_g * (1.0 - alpha)) as u32;
                let b = (fg_b * alpha + bg_b * (1.0 - alpha)) as u32;

                let final_color = (r << 16) | (g << 8) | b;

                self.buffer[index] = final_color;
                // Actualizamos Z incluso si es transparente para que ocluya objetos detrás
                self.zbuffer[index] = depth; 
            }
        }
    }

    pub fn set_background_color(&mut self, color: u32) {
        self.background_color = color;
    }

    pub fn set_current_color(&mut self, color: u32) {
        self.current_color = color;
    }

    #[inline(always)]
    pub fn set_pixel_unchecked(&mut self, x: usize, y: usize, color: u32) {
        // Safety: Solo usar si estamos seguros que x,y están dentro del buffer
        let index = y * self.width + x;
        unsafe {
            *self.buffer.get_unchecked_mut(index) = color;
        }
    }
    
    // Método para leer el color de fondo para blending (sin checks)
    #[inline(always)]
    pub fn get_pixel_unchecked(&self, x: usize, y: usize) -> u32 {
        let index = y * self.width + x;
        unsafe { *self.buffer.get_unchecked(index) }
    }
    
    // Actualizar Z-Buffer sin checks
    #[inline(always)]
    pub fn set_depth_unchecked(&mut self, x: usize, y: usize, depth: f32) {
        let index = y * self.width + x;
        unsafe {
            *self.zbuffer.get_unchecked_mut(index) = depth;
        }
    }
    
    // Leer Z-Buffer sin checks
    #[inline(always)]
    pub fn get_depth_unchecked(&self, x: usize, y: usize) -> f32 {
        let index = y * self.width + x;
        unsafe { *self.zbuffer.get_unchecked(index) }
    }
}