use std::marker::PhantomData;

use wgpu::util::DeviceExt;

use crate::{Framework, GpuResult};

/// Generic internal implementation for [`wgpu::Buffer`]
/// handling both uniform and storage buffers.
pub(crate) struct GenericBuffer<'fw, T> {
    fw: &'fw Framework,
    storage: wgpu::Buffer,
    size: usize,
    _marker: PhantomData<T>,
}

impl<'fw, T> GenericBuffer<'fw, T>
where
    T: bytemuck::Pod,
{
    /// Creates a complete [`wgpu::BindingResource`] of the [`GenericBuffer`].
    pub fn get_binding_resource(&self) -> wgpu::BindingResource {
        self.storage.as_entire_binding()
    }

    /// Gets the inner [`wgpu::Buffer`] of this [`GenericBuffer`].
    pub fn get_wgpu_buffer(&self) -> &wgpu::Buffer {
        &self.storage
    }

    /// Consumes this [`GenericBuffer`] into a [`wgpu::Buffer`] and its `size` in bytes.
    pub fn into_wgpu_buffer(self) -> (wgpu::Buffer, usize) {
        (self.storage, self.size)
    }

    /// Obtains the number of elements (or capacity if created using [`Framework::create_buffer`](crate::Framework::create_buffer))
    /// of the [`GenericBuffer`].
    pub fn len(&self) -> usize {
        self.size / std::mem::size_of::<T>()
    }

    /// Checks if the [`GenericBuffer`] is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Obtains the size in bytes of this [`GenericBuffer`].
    pub fn size(&self) -> usize {
        self.size
    }

    /// Creates an empty [`GenericBuffer`] of the desired `len`gth.
    pub fn new(fw: &'fw crate::Framework, len: usize) -> Self
    where
        T: bytemuck::Pod,
    {
        let size = len * std::mem::size_of::<T>();

        let storage = fw.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            fw,
            storage,
            size,
            _marker: PhantomData,
        }
    }

    /// Creates an empty uniform [`GenericBuffer`] of the desired `len`gth.
    ///
    /// Fails if `sizeof::<T>() * len` is bigger than GPU's max uniform buffer size.
    pub fn new_uniform(fw: &'fw Framework, len: usize) -> GpuResult<Self> {
        let size = len * std::mem::size_of::<T>();

        if size as u32 > fw.limits.max_uniform_buffer_binding_size {
            let msg = format!("Cannot create GpuUniformBuffer of {} bytes (max. {} bytes). Consider creating a GenericBuffer instead.",
                                        size,
                                        fw.limits.max_uniform_buffer_binding_size);
            return Err(msg.into());
        }

        let storage = fw.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            fw,
            storage,
            size,
            _marker: PhantomData,
        })
    }

    /// Creates a [`GenericBuffer`] from a `data` slice.
    pub fn from_slice(fw: &'fw crate::Framework, data: &[T]) -> Self
    where
        T: bytemuck::Pod,
    {
        let size = data.len() * std::mem::size_of::<T>();

        let storage = fw
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        Self {
            fw,
            storage,
            size,
            _marker: PhantomData,
        }
    }

    /// Creates a [`GenericBuffer`] from a [`wgpu::Buffer`] and its size in bytes.
    pub fn from_wgpu_buffer(fw: &'fw crate::Framework, buffer: wgpu::Buffer, size: usize) -> Self {
        Self {
            fw,
            storage: buffer,
            size,
            _marker: PhantomData,
        }
    }

    /// Creates an uniform [`GenericBuffer`] from a `data` slice.
    ///
    /// Fails if `data` byte size is bigger than GPU's max uniform buffer size.
    pub fn uniform_from_slice(fw: &'fw crate::Framework, data: &[T]) -> GpuResult<Self>
    where
        T: bytemuck::Pod,
    {
        let size = data.len() * std::mem::size_of::<T>();

        if size as u32 > fw.limits.max_uniform_buffer_binding_size {
            let msg = format!(
                "Cannot create an uniform GenericBuffer of {} bytes (max. {} bytes).",
                size, fw.limits.max_uniform_buffer_binding_size
            );
            return Err(msg.into());
        }

        let storage = fw
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        Ok(Self {
            fw,
            storage,
            size,
            _marker: PhantomData,
        })
    }

    /// Asyncronously reads the contents of the [`GenericBuffer`] into a [`Vec`].
    pub async fn read_async(&self) -> GpuResult<Vec<T>> {
        let staging = self.fw.create_download_staging_buffer(self.size);

        let mut encoder = self
            .fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GenericBuffer::read_async"),
            });
        encoder.copy_buffer_to_buffer(&self.storage, 0, &staging, 0, self.size as u64);

        self.fw.queue.submit(Some(encoder.finish()));

        let buff_slice = staging.slice(..);
        let buf_future = buff_slice.map_async(wgpu::MapMode::Read);

        buf_future.await?;

        let data = buff_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Blocking read of the content of the [`GenericBuffer`] into a [`Vec`].
    pub fn read(&self) -> GpuResult<Vec<T>> {
        let staging = self.fw.create_download_staging_buffer(self.size);

        let mut encoder = self
            .fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GenericBuffer::read"),
            });
        encoder.copy_buffer_to_buffer(&self.storage, 0, &staging, 0, self.size as u64);

        self.fw.queue.submit(Some(encoder.finish()));

        let buff_slice = staging.slice(..);
        let buf_future = buff_slice.map_async(wgpu::MapMode::Read);

        self.fw.blocking_poll();

        futures::executor::block_on(buf_future)?;

        let data = buff_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Asyncronously writes the contents of `data` into the [`GenericBuffer`].
    pub async fn write_async(&mut self, data: &[T]) -> GpuResult<()> {
        let staging = self.fw.create_upload_staging_buffer(self.size);

        let mut encoder = self
            .fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GenericBuffer::write_async"),
            });
        encoder.copy_buffer_to_buffer(&staging, 0, &self.storage, 0, self.size as u64);

        self.fw.queue.submit(Some(encoder.finish()));

        let buff_slice = self.storage.slice(..);
        let buf_future = buff_slice.map_async(wgpu::MapMode::Write);

        buf_future.await?;

        let mut write_view = buff_slice.get_mapped_range_mut();
        write_view.copy_from_slice(bytemuck::cast_slice(data));

        drop(write_view);
        self.storage.unmap();

        Ok(())
    }

    /// Writes the `data` information into the [`GenericBuffer`] immediately.
    pub fn write(&mut self, data: &[T]) {
        self.fw
            .queue
            .write_buffer(&self.storage, 0, bytemuck::cast_slice(data));

        let encoder = self
            .fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GenericBuffer::write"),
            });

        self.fw.queue.submit(Some(encoder.finish()));
    }
}
