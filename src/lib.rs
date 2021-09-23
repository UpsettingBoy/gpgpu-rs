//! A simple GPU compute library based on [`wgpu`](https://github.com/gfx-rs/wgpu).
//! It is meant to be used alongside `wgpu` if desired.
//!
//! Right now `gpgpu` uses some of `wgpu`'s type on its public API.
//! It may be removed in the future.
//!
//! To start using `gpgpu`, just create a [`Framework`] instance
//! and follow the examples in the main repository.
//!
//! # Example
//! Small program that multiplies 2 vectors A and B; and stores the
//! result in another vector C.
//! ## Rust program
//! ```no_run
//!  use gpgpu::*;
//!  use gpgpu::wgpu;
//!
//!  fn main() -> GpuResult<()> {
//!     let fw = Framework::default();
//!     
//!     // Original CPU data
//!     let cpu_data = (0..10000).into_iter().collect::<Vec<u32>>();
//!
//!     // GPU buffer creation
//!     let buf_a = fw.create_buffer_from_slice(&cpu_data);     // Input
//!     let buf_b = fw.create_buffer_from_slice(&cpu_data);     // Input
//!     let buf_c = fw.create_buffer::<u32>(cpu_data.len());    // Output
//!
//!     // Shader load from SPIR-V file
//!     let shader_module = utils::shader::from_spirv_file(&fw, "<shader path>")?;
//!
//!     // Descriptor set creation
//!     let desc_set = DescriptorSet::default()
//!         .bind_storage_buffer(&buf_a, true)      // Read-only
//!         .bind_storage_buffer(&buf_b, true)      // Read-only
//!         .bind_storage_buffer(&buf_c, false);    // Read-write
//!     
//!     // Kernel creation and enqueuing
//!     fw.create_kernel_builder(&shader_module, "main")   // Entry point
//!         .add_descriptor_set(desc_set)                      
//!         .build()
//!         .enqueue(cpu_data.len() as u32, 1, 1);         // Enqueuing, not very optimus 😅
//!
//!     let output = buf_c.read()?;                        // Read back C from GPU
//!     for (a, b) in cpu_data.into_iter().zip(output) {
//!         assert_eq!(a.pow(2), b);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Shader program
//! The shader is writen in [WGSL](https://gpuweb.github.io/gpuweb/wgsl/)
//! ```ignore
//! // Vector type definition. Used for both input and output
//! [[block]]
//! struct Vector {
//!     data: [[stride(4)]] array<u32>;
//! };
//!
//! // A, B and C vectors
//! [[group(0), binding(0)]] var<storage, read>  a: Vector;
//! [[group(0), binding(1)]] var<storage, read>  b: Vector;
//! [[group(0), binding(2)]] var<storage, read_write> c: Vector;
//!
//! [[stage(compute), workgroup_size(1)]]
//! fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
//!     c.data[global_id.x] = a.data[global_id.x] * b.data[global_id.x];
//! }
//! ```
//!

use std::marker::PhantomData;

pub use wgpu;

pub mod alloc;
pub mod framework;
pub mod kernel;
pub mod utils;

/// Lazy error handling :)
pub type GpuResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Allows the creation of [`GpuBuffer`], [`GpuImage`] and [`Kernel`].
#[allow(dead_code)]
pub struct Framework {
    instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// Vector of contiguous homogeneous elements on GPU memory.
/// This elements must implement [`bytemuck::Pod`](bytemuck::Pod).
///
/// Equivalent to OpenCL's Buffer objects.
pub struct GpuBuffer<'fw, T: bytemuck::Pod> {
    fw: &'fw Framework,
    storage: wgpu::Buffer,
    size: usize,

    _marker: PhantomData<T>,
}

/// 2D-image of homogeneous pixels.
///
/// Equivalent to OpenCL's Image objects.
pub struct GpuImage<'fw> {
    fw: &'fw Framework,
    texture: wgpu::Texture,
    format: wgpu::TextureFormat,
    size: wgpu::Extent3d,
    full_view: wgpu::TextureView,
}

/// Contains a binding group of resources.
#[derive(Default)]
pub struct DescriptorSet<'res> {
    set_layout: Vec<wgpu::BindGroupLayoutEntry>,
    binds: Vec<wgpu::BindGroupEntry<'res>>,
}

/// Creates a [`Kernel`] instance with the bindings
/// used during the configuration of this structure.
pub struct KernelBuilder<'fw, 'res, 'sha> {
    fw: &'fw Framework,
    layouts: Vec<wgpu::BindGroupLayout>,
    descriptors: Vec<DescriptorSet<'res>>,
    sets: Vec<wgpu::BindGroup>,
    shader: &'sha wgpu::ShaderModule,
    entry_point: String,
}

/// Used to enqueue the execution of a shader with the bidings provided.
///
/// Can only be created from [`KernelBuilder`].
/// Equivalent to OpenCL's Kernel.
// pub struct Kernel<'fw, 'res, 'sha> {
pub struct Kernel<'fw> {
    fw: &'fw Framework,
    // layouts: Vec<wgpu::BindGroupLayout>,
    // pipeline_layout: wgpu::PipelineLayout,
    pipeline: wgpu::ComputePipeline,
    // descriptors: Vec<DescriptorSet<'res>>,
    sets: Vec<wgpu::BindGroup>,
    // shader: &'sha wgpu::ShaderModule,
    entry_point: String,
}
