pub mod buffer;
pub mod image;

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
