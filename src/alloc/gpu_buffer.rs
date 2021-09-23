use crate::{GpuBuffer, GpuResult};

impl<'fw, T> GpuBuffer<'fw, T>
where
    T: bytemuck::Pod,
{
    /// Creates a complete [`BindingResource`](wgpu::BindingResource) of the [`GpuBuffer`].
    pub fn as_binding_resource(&self) -> wgpu::BindingResource {
        self.storage.as_entire_binding()
    }

    /// Obtains the number of elements (or capacity if created using [`Framework::create_buffer`](crate::Framework::create_buffer))
    /// of the [`GpuBuffer`].
    pub fn len(&self) -> usize {
        self.size / std::mem::size_of::<T>()
    }

    /// Checks if the [`GpuBuffer`] is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Obtains the size in bytes of this [`GpuBuffer`].
    pub fn size(&self) -> usize {
        self.size
    }

    /// Asyncronously reads the contents of the [`GpuBuffer`] into a [`Vec`].
    ///
    /// In order for this future to resolve, [`Framework::poll`](crate::Framework::poll) or [`Framework::blocking_poll`](crate::Framework::poll)
    /// must be invoked.
    pub async fn read_async(&self) -> GpuResult<Vec<T>> {
        let staging = self.fw.create_staging_buffer(self.size);

        let mut encoder = self
            .fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Buffer copy"),
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

    /// Blocking read of the content of the [`GpuBuffer`] into a [`Vec`].
    pub fn read(&self) -> GpuResult<Vec<T>> {
        let staging = self.fw.create_staging_buffer(self.size);

        let mut encoder = self
            .fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Buffer copy"),
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

    /// Asyncronously writes the contents of `data` into the [`GpuBuffer`].
    ///
    /// In order for this future to resolve, [`Framework::poll`](crate::Framework::poll) or [`Framework::blocking_poll`](crate::Framework::blocking_poll)
    /// must be invoked.
    pub async fn write_async(&mut self, data: &[T]) -> GpuResult<()> {
        let staging = self.fw.create_staging_buffer(data.len());

        let mut encoder = self
            .fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Buffer copy"),
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

    /// Writes the `data` information into the [`GpuBuffer`] immediately.
    pub fn write(&mut self, data: &[T]) {
        self.fw
            .queue
            .write_buffer(&self.storage, 0, bytemuck::cast_slice(data));

        let encoder = self
            .fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Buffer write"),
            });

        self.fw.queue.submit(Some(encoder.finish()));
    }
}
