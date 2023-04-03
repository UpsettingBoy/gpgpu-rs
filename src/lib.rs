//! An experimental async GPU compute library based on [`wgpu`](https://github.com/gfx-rs/wgpu).
//! It is meant to be used alongside `wgpu` if desired.
//!
//! To start using `gpgpu`, just create a [`Framework`](crate::Framework) instance
//! and follow the [examples](https://github.com/UpsettingBoy/gpgpu-rs/tree/dev/examples) in the main repository.
//!
//! # Example
//! Small program that multiplies 2 vectors A and B; and stores the
//! result in another vector C.
//! ## Rust program
//! ```no_run
//!     use gpgpu::*;
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Framework initialization
//!     let fw = Framework::default();
//!
//!     // Original CPU data
//!     let cpu_data = (0..10000).into_iter().collect::<Vec<u32>>();
//!
//!     // GPU buffer creation
//!     let buf_a = GpuBuffer::from_slice(&fw, &cpu_data);       // Input
//!     let buf_b = GpuBuffer::from_slice(&fw, &cpu_data);       // Input
//!     let buf_c = GpuBuffer::<u32>::with_capacity(&fw, cpu_data.len() as u64);  // Output
//!
//!     // Shader load from SPIR-V binary file
//!     let shader = Shader::from_spirv_file(&fw, "<SPIR-V shader path>")?;
//!     //  or from a WGSL source file
//!     let shader = Shader::from_wgsl_file(&fw, "<WGSL shader path>")?;    
//!
//!     // Descriptor set and program creation
//!     let desc = DescriptorSet::default()
//!         .bind_buffer(&buf_a, GpuBufferUsage::ReadOnly)
//!         .bind_buffer(&buf_b, GpuBufferUsage::ReadOnly)
//!         .bind_buffer(&buf_c, GpuBufferUsage::ReadWrite);
//!     let program = Program::new(&shader, "main").add_descriptor_set(desc); // Entry point
//!
//!     // Kernel creation and enqueuing
//!     Kernel::new(&fw, program).enqueue(cpu_data.len() as u32, 1, 1); // Enqueuing, not very optimus 😅
//!
//!     let output = buf_c.read_vec_blocking()?;                        // Read back C from GPU
//!     for (a, b) in cpu_data.into_iter().zip(output) {
//!         assert_eq!(a.pow(2), b);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Shader program
//! The shader is written in [WGSL](https://gpuweb.github.io/gpuweb/wgsl/)
//! ```ignore
//! // Vector type definition. Used for both input and output
//! struct Vector {
//!     data: array<u32>,
//! }
//!
//! // A, B and C vectors
//! @group(0) @binding(0) var<storage, read>  a: Vector;
//! @group(0) @binding(1) var<storage, read>  b: Vector;
//! @group(0) @binding(2) var<storage, read_write> c: Vector;
//!
//! @compute @workgroup_size(1)
//! fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
//!     c.data[global_id.x] = a.data[global_id.x] * b.data[global_id.x];
//! }
//! ```
//!

use std::{marker::PhantomData, sync::Arc};

#[cfg(feature = "integrate-ndarray")]
pub use features::integrate_ndarray::GpuArray;
use primitives::PixelInfo;
pub use primitives::{BufOps, ImgOps};

pub mod features;
pub mod framework;
pub mod kernel;
pub mod primitives;

/// Entry point of `gpgpu`. A [`Framework`] must be created
/// first as all GPU primitives needs it to be created.
pub struct Framework {
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    adapter: wgpu::Adapter,
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum GpuBufferUsage {
    /// Read-only object.
    /// ### Example WGSL syntax:
    /// ```ignore
    /// [[group(0), binding(0)]] var<storage, read> input: Vector;
    /// ```
    ReadOnly,
    /// Read-write object.
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
/// More information about its shader representation is
/// under the [`DescriptorSet::bind_buffer`](crate::DescriptorSet::bind_buffer) documentation.
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
/// More information about its shader representation is
/// under the [`DescriptorSet::bind_uniform_buffer`](crate::DescriptorSet::bind_uniform_buffer) documentation.
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
/// More information about its shader representation is
/// under the [`DescriptorSet::bind_image`](crate::DescriptorSet::bind_image) documentation.
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
/// More information about its shader representation is
/// under the [`DescriptorSet::bind_const_image`](crate::DescriptorSet::bind_const_image) documentation.
pub struct GpuConstImage<'fw, P> {
    fw: &'fw Framework,
    texture: wgpu::Texture,
    size: wgpu::Extent3d,
    full_view: wgpu::TextureView,
    pixel: PhantomData<P>,
}

/// Represents a shader.
///
/// It's just a wrapper around [`wgpu::ShaderModule`].
pub struct Shader(wgpu::ShaderModule);

/// Represents an entry point with its bindings on a [`Shader`].
pub struct Program<'sha, 'res> {
    shader: &'sha Shader,
    entry_point: String,
    descriptors: Vec<DescriptorSet<'res>>,
}

/// Contains a binding group of resources.
#[derive(Default, Clone, Debug)]
pub struct DescriptorSet<'res> {
    set_id: u32,
    binds: Vec<wgpu::BindGroupEntry<'res>>,
}

/// Used to enqueue the execution of a shader with the bidings provided.
///
/// Equivalent to OpenCL's Kernel.
pub struct Kernel<'fw> {
    fw: &'fw Framework,
    pipeline: wgpu::ComputePipeline,
    sets: Vec<(u32, wgpu::BindGroup)>,
    entry_point: String,
}

pub struct BindGroupLayoutBuilder {
    entries: Vec<wgpu::BindGroupLayoutEntry>,
    bind_id: u32,
}

impl BindGroupLayoutBuilder {
    pub fn new() -> Self {
        Self {
            entries: vec![],
            bind_id: 0,
        }
    }

    pub fn add_buffer(mut self, usage: GpuBufferUsage) -> Self {
        self.entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                has_dynamic_offset: false,
                min_binding_size: None,
                ty: wgpu::BufferBindingType::Storage {
                    read_only: usage == GpuBufferUsage::ReadOnly,
                },
            },
            count: None,
        });

        self.bind_id += 1;
        self
    }

    pub fn add_uniform_buffer(mut self) -> Self {
        self.entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                has_dynamic_offset: false,
                min_binding_size: None,
                ty: wgpu::BufferBindingType::Uniform,
            },
            count: None,
        });

        self.bind_id += 1;
        self
    }

    pub fn add_image<P: PixelInfo>(mut self) -> Self {
        self.entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: P::wgpu_format(),
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        });

        self.bind_id += 1;
        self
    }

    pub fn add_const_image<P: PixelInfo>(mut self) -> Self {
        self.entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: P::wgpu_texture_sample(),
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        });

        self.bind_id += 1;
        self
    }

    pub fn add_array(mut self, access: GpuBufferUsage) -> Self {
        self.entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                has_dynamic_offset: false,
                min_binding_size: None,
                ty: wgpu::BufferBindingType::Storage {
                    read_only: access == GpuBufferUsage::ReadOnly,
                },
            },
            count: None,
        });

        self.bind_id += 1;
        self
    }
}
