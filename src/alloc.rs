#![macro_use]

use crate::AccessMode;

pub mod buffer;
pub mod image;

impl AccessMode {
    pub(crate) fn to_wgpu_storage_texture_access(&self) -> wgpu::StorageTextureAccess {
        match self {
            AccessMode::ReadOnly => wgpu::StorageTextureAccess::ReadOnly,
            AccessMode::WriteOnly => wgpu::StorageTextureAccess::WriteOnly,
            AccessMode::ReadWrite => wgpu::StorageTextureAccess::ReadWrite,
        }
    }
}

/// Gives some information about the pixel format.
pub trait PixelInfo {
    fn byte_size() -> usize;
    fn wgpu_format() -> wgpu::TextureFormat;
    fn wgpu_texture_sample() -> wgpu::TextureSampleType;
}

macro_rules! pixel_impl {
    ($($name:ident, $size:expr, $format:expr, $sample:expr, #[$doc:meta]);+) => {
        use crate::alloc::PixelInfo;

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
