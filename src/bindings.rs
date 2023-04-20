use crate::{entry_type::EntryType, primitives::*, *};

#[derive(Clone, Default)]
pub struct SetBindings<'res> {
    pub(crate) bindings: Vec<wgpu::BindGroupEntry<'res>>,
    pub(crate) entry_type: Vec<EntryType>,
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

        self.bindings.push(bind);
        self.entry_type.push(EntryType::Buffer)
    }

    pub fn add_uniform_buffer<T>(&mut self, bind_id: u32, buffer: &'res GpuUniformBuffer<T>)
    where
        T: bytemuck::Pod,
    {
        let bind = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: buffer.as_binding_resource(),
        };

        self.bindings.push(bind);
        self.entry_type.push(EntryType::Uniform)
    }

    pub fn add_image<P: PixelInfo>(&mut self, bind_id: u32, img: &'res GpuImage<P>) {
        let bind = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: img.as_binding_resource(),
        };

        self.bindings.push(bind);
        self.entry_type.push(EntryType::Image)
    }

    pub fn add_const_image<P: PixelInfo>(&mut self, bind_id: u32, img: &'res GpuConstImage<P>) {
        let bind = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: img.as_binding_resource(),
        };

        self.bindings.push(bind);
        self.entry_type.push(EntryType::ConstImage)
    }
}
