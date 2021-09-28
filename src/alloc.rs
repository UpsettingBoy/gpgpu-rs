use crate::pixel_impl;

pub mod gpu_buffer;
pub mod gpu_image;

pub trait PixelInfo {
    fn byte_size() -> usize;
    fn wgpu_format() -> wgpu::TextureFormat;
}

pixel_impl! {
    RgbaU8, 4, wgpu::TextureFormat::Rgba8Uint;
    RgbaI8, 4, wgpu::TextureFormat::Rgba8Sint
}

#[macro_export]
macro_rules! pixel_impl {
    ($($name:ident, $size:expr, $wgpu:expr);+) => {
        $(
            pub struct $name;

            impl PixelInfo for $name {
                fn byte_size() -> usize {
                    $size
                }

                fn wgpu_format() -> wgpu::TextureFormat {
                    $wgpu
                }
            }
        )+
    };
}
