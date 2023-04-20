use thiserror::Error;

use crate::{primitives::buffers::BufferError, *};

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

    pub fn write(&self, array: ndarray::ArrayView<T, D>) -> ArrayResult<u64> {
        let slice: Result<&[T], _> = array
            .as_slice_memory_order()
            .ok_or(NdarrayError::ArrayNotContiguous);

        Ok(self.0.write(slice?)?)
    }

    pub fn to_gpu_buffer(self) -> GpuBuffer<'fw, T> {
        self.0
    }
}

impl SetLayout {
    pub fn add_array(&mut self, bind_id: u32, usage: GpuBufferUsage) {
        self.add_buffer(bind_id, usage)
    }
}

impl<'res> SetBindings<'res> {
    pub fn add_array<T, D>(self, bind_id: u32, arr: &'res GpuArray<T, D>) -> Self
    where
        T: bytemuck::Pod,
        D: ndarray::Dimension,
    {
        self.add_buffer(bind_id, &arr.0)
    }
}
