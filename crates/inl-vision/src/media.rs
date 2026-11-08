//! Decoded media shared by the image and video paths.

/// A decoded image: tightly packed RGB8, row-major, `data.len() == w * h * 3`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl RgbImage {
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> inl_core::Result<Self> {
        if data.len() != width as usize * height as usize * 3 {
            return Err(inl_core::InlError::InvalidInput(format!(
                "RGB buffer of {} bytes does not match {width}x{height}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }
}
