//! An experimental async GPU compute library based on [`wgpu`](https://github.com/gfx-rs/wgpu).
//! It is meant to be used alongside `wgpu` if desired.
//!
//! # Getting Started
//! First run cargo add --github <https://github.com/myname1111/gpgpu-rs> (for this version)
//!
//! If you'd like to use this crate with ['image'](https://github.com/image-rs/image) or ['ndarray'](https://github.com/rust-ndarray/ndarray)
//! Then turn on the image and ndarray features respectively
//!
//! You can then follow one of the examples and or read the book on GPU Computing in rust (TODO)
//!
//! # Features flags
//! `image`: Recommended when using the `image` crate \
//! `ndarray`: Recommended when using the `ndarray` crate

use std::marker::PhantomData;

pub use bindings::SetBindings;
#[cfg(feature = "integrate-ndarray")]
pub use features::integrate_ndarray::GpuArray;
pub use framework::Framework;
pub use kernel::Kernel;
pub use layout::SetLayout;
pub use primitives::{BufOps, ImgOps};
pub use shader::Shader;

pub mod bindings;
pub mod features;
pub mod framework;
pub mod kernel;
pub mod layout;
pub mod primitives;
pub mod shader;

mod entry_type;

#[derive(PartialEq, Eq)]
pub enum GpuBufferUsage {
    /// Read-only buffer.
    /// ### Example WGSL syntax:
    /// ```ignore
    /// [[group(0), binding(0)]] var<storage, read> input: Vector;
    /// ```
    ReadOnly,
    /// Read-write buffer.
    /// ### Example WGSL syntax:
    /// ```ignore
    /// [[group(0), binding(0)]] var<storage, read_write> input: Vector;
    /// ```
    ReadWrite,
}

/// Vector of contiguous homogeneous elements on GPU memory.
/// Its elements must implement [`bytemuck::Pod`].
///
/// Equivalent to OpenCL's Buffer objects.
///
/// ### Example WGSL syntax:
/// ```ignore
/// struct StorageStruct {
///     data: [[stride(4)]] array<i32>;
/// };
///
/// [[group(0), binding(0)]]
/// var<storage, read_write> myStorageBuffer: StorageStruct;
/// ```
///
/// ### Example GLSL syntax:
/// ```glsl
/// layout (set=0, binding=0) buffer myStorageBuffer {
///     int data[];
/// };
/// ```
pub struct GpuBuffer<'fw, T> {
    fw: &'fw Framework,
    buf: wgpu::Buffer,
    size: u64,
    marker: PhantomData<T>,
}

/// Uniform vector of contiguous homogeneous elements on GPU memory.
/// Recommended for small, read-only buffers.
/// Its elements must implement [`bytemuck::Pod`].
///
/// Equivalent to OpenCL's Uniform Buffer objects.
///
/// ### Example WGSL syntax:
/// ```ignore
/// struct UniformStruct {
///     a: vec3<u32>;
///     b: vec3<u32>;
///     c: vec3<u32>;
/// };
///
/// [[group(0), binding(0)]]
/// var<uniform> myUniformBuffer: UniformStruct;
/// ```
///
/// ### Example GLSL syntax:
/// ```glsl
/// layout(std140, binding = 0)
/// uniform UniformStruct {
///     uvec3 a;
///     uvec3 b;
///     uvec3 c;
/// };
/// ```
pub struct GpuUniformBuffer<'fw, T> {
    fw: &'fw Framework,
    buf: wgpu::Buffer,
    size: u64,
    marker: PhantomData<T>,
}

/// 2D-image of homogeneous pixels.
///
/// Equivalent to write-only OpenCL's Image objects.
///
/// This image is write-only.
/// ### Example WGSL syntax:
/// ```ignore
/// [[group(0), binding(0)]]
/// var myStorageImg: texture_storage_2d<rgba8uint, write>;
/// ```
///
/// ### Example GLSL syntax:
/// ```glsl
/// layout (set=0, binding=0, rgba8uint) uimage2D myStorageImg;
/// ```
pub struct GpuImage<'fw, P> {
    fw: &'fw Framework,
    texture: wgpu::Texture,
    size: wgpu::Extent3d,
    full_view: wgpu::TextureView,
    pixel: PhantomData<P>,
}

/// 2D-image of homogeneous pixels.
///
/// Equivalent to read-only OpenCL's Image objects.
///
/// This image is read-only.
/// ### Example WGSL syntax:
/// ```ignore
/// [[group(0), binding(0)]]
/// var myTexture: texture_2d<u32>;
/// ```
///
/// ### Example GLSL syntax:
/// ```glsl
/// layout (set=0, binding=0) utexture2D myTexture;
/// ```
pub struct GpuConstImage<'fw, P> {
    fw: &'fw Framework,
    texture: wgpu::Texture,
    size: wgpu::Extent3d,
    full_view: wgpu::TextureView,
    pixel: PhantomData<P>,
}
