use crate::{GpuBuffer, GpuResult};

impl<'fw, T> GpuBuffer<'fw, T>
where
    T: bytemuck::Pod,
{
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.size / std::mem::size_of::<T>()
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn as_binding_resource(&self) -> wgpu::BindingResource {
        self.storage.as_entire_binding()
    }

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

        self.fw.poll_wait();

        futures::executor::block_on(buf_future)?;

        let data = buff_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging.unmap();

        Ok(result)
    }
}
