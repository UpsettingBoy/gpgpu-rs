use std::{marker::PhantomData, rc::Rc};

pub mod alloc;
pub mod framework;
pub mod kernel;
pub mod utils;

pub type GpuResult<T> = Result<T, Box<dyn std::error::Error>>;

#[allow(dead_code)]
pub struct Framework {
    instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

pub struct GpuBuffer<'fw, T: bytemuck::Pod> {
    fw: &'fw Framework,
    storage: wgpu::Buffer,
    size: usize,

    _marker: PhantomData<T>,
}

#[derive(Default)]
pub struct DescriptorSet<'res> {
    set_layout: Vec<wgpu::BindGroupLayoutEntry>,
    binds: Vec<wgpu::BindGroupEntry<'res>>,
}

pub struct KernelBuilder<'fw, 'res, 'sha> {
    fw: &'fw Framework,
    layouts: Vec<wgpu::BindGroupLayout>,
    descriptors: Vec<DescriptorSet<'res>>,
    sets: Vec<wgpu::BindGroup>,
    shader: &'sha wgpu::ShaderModule,
    entry_point: String,
}

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
