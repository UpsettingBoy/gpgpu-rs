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

use crate::Framework;

pub mod buffers;
pub mod generic_image;
pub mod image;
pub mod images;

/// Interface to get information, create and decompose GPU allocated buffers.
pub trait BufOps<'fw, T>
where
    T: bytemuck::Pod,
{
    // ----------- Information fns -----------

    /// Returns the number of elements the buffer can hold.
    fn capacity(&self) -> usize {
        self.size() / std::mem::size_of::<T>()
    }

    /// Returns the number of bytes of the buffer.
    fn size(&self) -> usize;

    /// Returns a [`wgpu::BindingResource`] of all the elements in the buffer.
    fn as_binding_resource(&self) -> wgpu::BindingResource {
        self.as_gpu_buffer().as_entire_binding()
    }

    /// Returns the [`wgpu::Buffer`] that handles the GPU data of the buffer.
    fn as_gpu_buffer(&self) -> &wgpu::Buffer;

    // ----------- Creation fns --------------

    /// Constructs a new zeroed buffer with the specified capacity.
    ///
    /// The buffer will be able to hold exactly `capacity` elements.
    fn with_capacity(fw: &'fw Framework, capacity: usize) -> Self;

    /// Constructs a new buffer from a slice.
    ///
    /// The buffer `capacity` will be the `slice` length.
    fn from_slice(fw: &'fw Framework, slice: &[T]) -> Self;

    /// Constructs a new buffer from a [`wgpu::Buffer`] and its byte `size`.
    ///
    /// # Safety
    /// If any of the following conditions are not satisfied, the buffer will
    /// panic at any time during its usage.
    /// - `size` needs to be less than or equal to the `buf` creation size.
    /// - `size` needs to be multiple of the `T` size.
    fn from_gpu_parts(fw: &'fw Framework, buf: wgpu::Buffer, size: usize) -> Self;

    // --------- Decomposition fns -------------

    /// Decomposes a buffer into a [`wgpu::Buffer`] and its byte `size`.
    fn into_gpu_parts(self) -> (wgpu::Buffer, usize);
}

/// Interface to get information, create and decompose GPU allocated images.
pub trait ImgOps<'fw> {
    // --------- Information fns --------------

    /// Returns the [`wgpu::Texture`] that handles the GPU image.
    fn as_gpu_texture(&self) -> &wgpu::Texture;

    /// Returns a [`wgpu::Extent3d`] of the image. Convenience function.
    fn get_wgpu_extent3d(&self) -> wgpu::Extent3d;

    /// Returns the width and height of the image.
    fn dimensions(&self) -> (u32, u32);

    // ----------- Creation fns ---------------

    /// Constructs an empty image with the desired `width` and `height`.
    fn new(fw: &'fw Framework, width: u32, height: u32) -> Self;

    /// Construct a new image from a bytes source `data` and its `width` and `height`.
    fn from_bytes(fw: &'fw Framework, data: &[u8], width: u32, height: u32) -> Self;

    fn from_gpu_parts(
        fw: &'fw Framework,
        texture: wgpu::Texture,
        dimensions: wgpu::Extent3d,
    ) -> Self;

    // -------- Decomposition fns -------------

    /// Decomposes an image into a [`wgpu::Texture`] and its [`wgpu::Extent3d`].
    fn into_gpu_parts(self) -> (wgpu::Texture, wgpu::Extent3d);
}

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
