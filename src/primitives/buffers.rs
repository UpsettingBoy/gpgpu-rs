use crate::{GpuBuffer, GpuUniformBuffer};

use super::generic_buffer::{BufferResult, GenericBuffer};

impl<'fw, T> GpuBuffer<'fw, T>
where
    T: bytemuck::Pod,
{
    /// Gets the inner [`wgpu::Buffer`] of this [`GpuBuffer`].
    pub fn get_wgpu_buffer(&self) -> &wgpu::Buffer {
        self.0.get_wgpu_buffer()
    }

    /// Consumes this [`GpuBuffer`] into a [`wgpu::Buffer`] and its `size` in bytes.
    pub fn into_wgpu_buffer(self) -> (wgpu::Buffer, usize) {
        self.0.into_wgpu_buffer()
    }

    /// Obtains the number of elements (or capacity if created using
    /// [`Framework::create_buffer`](crate::Framework::create_buffer))
    /// of the [`GpuBuffer`].
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Checks if the [`GpuBuffer`] is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Obtains the size in bytes of this [`GpuBuffer`].
    pub fn size(&self) -> usize {
        self.0.size()
    }

    /// Creates an empty [`GpuBuffer`] of the desired `len`gth.
    pub fn new(fw: &'fw crate::Framework, len: usize) -> Self
    where
        T: bytemuck::Pod,
    {
        Self(GenericBuffer::new(fw, len))
    }

    /// Creates a [`GpuBuffer`] from a `data` slice.
    pub fn from_slice(fw: &'fw crate::Framework, data: &[T]) -> Self
    where
        T: bytemuck::Pod,
    {
        Self(GenericBuffer::from_slice(fw, data))
    }

    /// Creates a [`GpuBuffer`] from a [`wgpu::Buffer`] and its size in bytes.
    ///
    /// `buffer` must have `wgpu::BufferUsages::STORAGE`, `wgpu::BufferUsages::COPY_SRC` and `wgpu::BufferUsages::COPY_DST` usages.
    pub fn from_wgpu_buffer(fw: &'fw crate::Framework, buffer: wgpu::Buffer, size: usize) -> Self {
        Self(GenericBuffer::from_wgpu_buffer(fw, buffer, size))
    }

    /// Asyncronously reads the contents of the [`GpuBuffer`] into a [`Vec`].
    ///
    /// In order for this future to resolve, [`Framework::poll`](crate::Framework::poll) or
    /// [`Framework::blocking_poll`](crate::Framework::poll) must be invoked.
    pub async fn read_async(&self) -> BufferResult<Vec<T>> {
        self.0.read_async().await
    }

    /// Blocking read of the content of the [`GpuBuffer`] into a [`Vec`].
    pub fn read(&self) -> BufferResult<Vec<T>> {
        self.0.read()
    }

    /// Asyncronously writes the contents of `data` into the [`GpuBuffer`].
    ///
    /// In order for this future to resolve, [`Framework::poll`](crate::Framework::poll) or
    /// [`Framework::blocking_poll`](crate::Framework::blocking_poll) must be invoked.
    pub async fn write_async(&mut self, data: &[T]) -> BufferResult<()> {
        self.0.write_async(data).await
    }

    /// Writes immediately the `data` contents into the [`GpuBuffer`].
    pub fn write(&mut self, data: &[T]) {
        self.0.write(data)
    }
}

impl<'fw, T> GpuUniformBuffer<'fw, T>
where
    T: bytemuck::Pod,
{
    /// Gets the inner [`wgpu::Buffer`] of this [`GpuUniformBuffer`].
    pub fn get_wgpu_buffer(&self) -> &wgpu::Buffer {
        self.0.get_wgpu_buffer()
    }

    /// Consumes this [`GpuUniformBuffer`] into a [`wgpu::Buffer`] and its `size` in bytes.
    pub fn into_wgpu_buffer(self) -> (wgpu::Buffer, usize) {
        self.0.into_wgpu_buffer()
    }

    /// Obtains the number of elements (or capacity if created using
    /// [`Framework::create_buffer`](crate::Framework::create_buffer))
    /// of the [`GpuUniformBuffer`].
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Checks if the [`GpuUniformBuffer`] is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Obtains the size in bytes of this [`GpuUniformBuffer`].
    pub fn size(&self) -> usize {
        self.0.size()
    }

    /// Creates an empty [`GpuUniformBuffer`] of the desired `len`gth.
    ///
    /// Fails if `sizeof::<T>() * len` is bigger than GPU's max uniform buffer size.
    pub fn new(fw: &'fw crate::Framework, len: usize) -> BufferResult<Self> {
        Ok(Self(GenericBuffer::new_uniform(fw, len)?))
    }

    /// Creates a [`GpuUniformBuffer`] from a `data` slice.
    ///
    /// Fails if `data` byte size is bigger than GPU's max uniform buffer size.
    pub fn from_slice(fw: &'fw crate::Framework, data: &[T]) -> BufferResult<Self>
    where
        T: bytemuck::Pod,
    {
        Ok(Self(GenericBuffer::uniform_from_slice(fw, data)?))
    }

    /// Asyncronously writes the contents of `data` into the [`GpuUniformBuffer`].
    ///
    /// In order for this future to resolve, [`Framework::poll`](crate::Framework::poll) or
    /// [`Framework::blocking_poll`](crate::Framework::blocking_poll) must be invoked.
    pub async fn write_async(&mut self, data: &[T]) -> BufferResult<()> {
        self.0.write_async(data).await
    }

    /// Writes immediately the `data` contents into the [`GpuUniformBuffer`].
    pub fn write(&mut self, data: &[T]) {
        self.0.write(data)
    }
}
