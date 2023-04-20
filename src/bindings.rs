use crate::{primitives::*, *};

#[derive(Clone, Default)]
pub struct SetBindings<'res> {
    pub(crate) bindings: Vec<wgpu::BindGroupEntry<'res>>,
}

impl<'res> SetBindings<'res> {
    pub fn add_buffer<T>(&mut self, bind_id: u32, buffer: &'res GpuBuffer<T>)
    where
        T: bytemuck::Pod,
    {
        let bind = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: buffer.as_binding_resource(),
        };

        self.bindings.push(bind)
    }

    pub fn add_uniform_buffer<T>(&mut self, bind_id: u32, buffer: &'res GpuUniformBuffer<T>)
    where
        T: bytemuck::Pod,
    {
        let bind = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: buffer.as_binding_resource(),
        };

        self.bindings.push(bind)
    }

    pub fn add_image<P: PixelInfo>(&mut self, bind_id: u32, img: &'res GpuImage<P>) {
        let bind = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: img.as_binding_resource(),
        };

        self.bindings.push(bind)
    }

    pub fn add_const_image<P: PixelInfo>(&mut self, bind_id: u32, img: &'res GpuConstImage<P>) {
        let bind = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: img.as_binding_resource(),
        };

        self.bindings.push(bind)
    }
}
