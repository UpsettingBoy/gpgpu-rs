use crate::{primitives::generic_buffer::GenericBuffer, DescriptorSet, GpuBufferUsage, GpuResult};

pub struct GpuArray<'fw, T, D>(GenericBuffer<'fw, T>, D);

impl<'fw, T, D> GpuArray<'fw, T, D>
where
    T: bytemuck::Pod,
    D: ndarray::Dimension,
{
    pub fn from_array(
        fw: &'fw crate::Framework,
        array: ndarray::ArrayView<T, D>,
    ) -> GpuResult<Self> {
        let slice: Result<&[T], Box<dyn std::error::Error>> = array
            .as_slice_memory_order()
            .ok_or("Array is not contiguous".into());

        let buf = GenericBuffer::from_slice(fw, slice?);

        Ok(Self(buf, array.raw_dim()))
    }

    pub fn read_to_array(&self) -> GpuResult<ndarray::Array<T, D>> {
        let v = self.0.read()?;
        let array = ndarray::Array::from_shape_vec(self.1.clone(), v).map_err(|_| "Shape error");
        Ok(array?)
    }

    pub fn write_to_array(&mut self, array: ndarray::ArrayView<T, D>) -> GpuResult<()> {
        let slice: Result<&[T], Box<dyn std::error::Error>> = array
            .as_slice_memory_order()
            .ok_or("Array is not contiguous".into());

        self.0.write(slice?);

        Ok(())
    }
}

impl<'res> DescriptorSet<'res> {
    pub fn bind_array<T, D>(mut self, array: &'res GpuArray<T, D>, access: GpuBufferUsage) -> Self
    where
        T: bytemuck::Pod,
        D: ndarray::Dimension,
    {
        let bind_id = self.set_layout.len() as u32;

        let bind_entry = wgpu::BindGroupLayoutEntry {
            binding: bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                has_dynamic_offset: false,
                min_binding_size: None,
                ty: wgpu::BufferBindingType::Storage {
                    read_only: access == GpuBufferUsage::ReadOnly,
                },
            },
            count: None,
        };

        let bind = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: array.0.get_binding_resource(),
        };

        self.set_layout.push(bind_entry);
        self.binds.push(bind);

        self
    }
}
