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

pub trait PixelInfo {
    fn byte_size() -> usize;
    fn wgpu_format() -> wgpu::TextureFormat;
}

#[macro_export]
macro_rules! pixel_impl {
    ($($name:ident, $size:expr, $wgpu:expr, #[$doc:meta]);+) => {
        $(
            #[$doc]
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
