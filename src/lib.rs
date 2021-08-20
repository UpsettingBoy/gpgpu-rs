use std::{marker::PhantomData, sync::Arc};

pub mod alloc;
pub mod framework;
pub mod kernel;
pub mod utils;

pub type GpuResult<T> = Result<T, Box<dyn std::error::Error>>;

pub struct Framework {
    instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

pub struct GpuBuffer<T: bytemuck::Pod> {
    storage: wgpu::Buffer,
    size: usize,

    _marker: PhantomData<T>,
}

pub struct Kernel<'res> {
    shader: Arc<wgpu::ShaderModule>,
    name: String,
    entry_point: String,
    pipeline: wgpu::ComputePipeline,
    descriptors: Vec<DescriptorSet<'res>>,
}

pub(crate) struct DescriptorSet<'res> {
    pub(crate) set: wgpu::BindGroup,
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) bindings: Vec<wgpu::BindGroupEntry<'res>>,
}
