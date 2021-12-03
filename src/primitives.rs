//! This module contains the `gpgpu` primitives.
//!
//! # Buffers
//! ## GpuBuffer
//! Intended for large, read-only or read-write (in the shader)
//! chunks of data on the GPU.
//!
//! ## GpuUniformBuffer
//! Intended for small, read-only (in the shader)
//! chunks of data on the GPU.
//!
//! # Images
//! ## GpuImage
//! Intended for write-only (in the shader) images on the GPU.
//!
//! ## GpuConstImage
//! Intended for read-only (in th shader) images on the GPU.

pub mod buffers;
pub mod generic_buffer;
pub mod generic_image;
pub mod image;

/// Gives some information about the pixel format.
pub trait PixelInfo {
    fn byte_size() -> usize;
    fn wgpu_format() -> wgpu::TextureFormat;
    fn wgpu_texture_sample() -> wgpu::TextureSampleType;
}

macro_rules! pixel_info_impl {
    ($($name:ident, $size:expr, $format:expr, $sample:expr, #[$doc:meta]);+) => {
        use crate::primitives::PixelInfo;

        $(
            #[$doc]
            pub struct $name;

            impl PixelInfo for $name {
                fn byte_size() -> usize {
                    $size
                }

                fn wgpu_format() -> wgpu::TextureFormat {
                    $format
                }

                fn wgpu_texture_sample() -> wgpu::TextureSampleType {
                    $sample
                }
            }
        )+
    };
}

pub mod pixels {
    pixel_info_impl! {
        Rgba8Uint, 4, wgpu::TextureFormat::Rgba8Uint, wgpu::TextureSampleType::Uint, #[doc = "Red, green, blue, and alpha channels. 8 bit integer per channel. Unsigned in shader."];
        Rgba8UintNorm, 4, wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureSampleType::Float { filterable: false }, #[doc = "Red, green, blue, and alpha channels. 8 bit integer per channel. [0, 255] converted to/from float [0, 1] in shader."];
        Rgba8Sint, 4, wgpu::TextureFormat::Rgba8Sint, wgpu::TextureSampleType::Sint, #[doc = "Red, green, blue, and alpha channels. 8 bit integer per channel. Signed in shader."];
        Rgba8SintNorm, 4, wgpu::TextureFormat::Rgba8Snorm, wgpu::TextureSampleType::Float { filterable: false }, #[doc = "Red, green, blue, and alpha channels. 8 bit integer per channel. [-127, 127] converted to/from float [-1, 1] in shader."]
        // Luma8, 1, wgpu::TextureFormat::R8Uint, wgpu::TextureSampleType::Uint, #[doc = "Grayscale 8 bit integer channel. Unsigned in shader."];
        // Luma8Norm, 1, wgpu::TextureFormat::R8Unorm, wgpu::TextureSampleType::Float { filterable: false }, #[doc = "Grayscale 8 bit integer channel. Unsigned in shader. [0, 255] converted to/from float [0, 1] in shader."]
    }
}
