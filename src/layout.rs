use crate::{primitives::PixelInfo, GpuBufferUsage};

#[derive(Default, Clone)]
pub struct SetLayout {
    pub(crate) layout_entry: Vec<wgpu::BindGroupLayoutEntry>,
}

impl SetLayout {
    pub fn add_buffer(&mut self, bind_id: u32, usage: GpuBufferUsage) {
        let entry = wgpu::BindGroupLayoutEntry {
            binding: bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                has_dynamic_offset: false,
                min_binding_size: None,
                ty: wgpu::BufferBindingType::Storage {
                    read_only: usage == GpuBufferUsage::ReadOnly,
                },
            },
            count: None,
        };

        self.layout_entry.push(entry)
    }

    pub fn add_uniform_buffer(&mut self, bind_id: u32) {
        let entry = wgpu::BindGroupLayoutEntry {
            binding: bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                has_dynamic_offset: false,
                min_binding_size: None,
                ty: wgpu::BufferBindingType::Uniform,
            },
            count: None,
        };

        self.layout_entry.push(entry)
    }

    pub fn add_image<P: PixelInfo>(&mut self, bind_id: u32) {
        let entry = wgpu::BindGroupLayoutEntry {
            binding: bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: P::wgpu_format(),
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };

        self.layout_entry.push(entry)
    }

    pub fn add_const_image<P: PixelInfo>(&mut self, bind_id: u32) {
        let entry = wgpu::BindGroupLayoutEntry {
            binding: bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: P::wgpu_texture_sample(),
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };

        self.layout_entry.push(entry)
    }
}
