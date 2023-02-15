use thiserror::Error;

use crate::{primitives::buffers::BufferError, BufOps, DescriptorSet, GpuBuffer, GpuBufferUsage};

#[derive(Error, Debug)]
pub enum NdarrayError {
    #[error("array is not contiguous.")]
    ArrayNotContiguous,
    #[error(transparent)]
    InvalidShape(#[from] ndarray::ShapeError),
}

#[derive(Error, Debug)]
pub enum ArrayError {
    #[error(transparent)]
    NdarrayError(#[from] NdarrayError),
    #[error(transparent)]
    BufferError(#[from] BufferError),
}

pub type ArrayResult<T> = Result<T, ArrayError>;

pub struct GpuArray<'fw, T, D>(GpuBuffer<'fw, T>, D);

impl<'fw, T, D> GpuArray<'fw, T, D>
where
    T: bytemuck::Pod,
    D: ndarray::Dimension,
{
    pub fn from_array(
        fw: &'fw crate::Framework,
        array: ndarray::ArrayView<T, D>,
    ) -> ArrayResult<Self> {
        let slice: Result<&[T], _> = array
            .as_slice_memory_order()
            .ok_or(NdarrayError::ArrayNotContiguous);

        let buf = GpuBuffer::from_slice(fw, slice?);

        Ok(Self(buf, array.raw_dim()))
    }

    pub async fn read(&self) -> ArrayResult<ndarray::Array<T, D>> {
        let v = self.0.read_vec().await?;
        ndarray::Array::from_shape_vec(self.1.clone(), v)
            .map_err(NdarrayError::InvalidShape)
            .map_err(ArrayError::NdarrayError)
    }

    pub fn read_blocking(&self) -> ArrayResult<ndarray::Array<T, D>> {
        futures::executor::block_on(self.read())
    }

    pub async fn write<'st>(&self, array: ndarray::ArrayView<'st, T, D>) -> ArrayResult<u64> {
        let slice: Result<&[T], _> = array
            .as_slice_memory_order()
            .ok_or(NdarrayError::ArrayNotContiguous);

        Ok(self.0.write(slice?).await?)
    }

    pub fn to_gpu_buffer(self) -> GpuBuffer<'fw, T> {
        self.0
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
            resource: array.0.as_binding_resource(),
        };

        self.set_layout.push(bind_entry);
        self.binds.push(bind);

        self
    }
}
