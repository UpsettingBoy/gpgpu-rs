use crate::{GpuImage, GpuResult};

impl<'fw> GpuImage<'fw> {
    /// Creates a complete [`BindingResource`](wgpu::BindingResource) of the [`GpuImage`].
    pub fn as_binding_resource(&self) -> wgpu::BindingResource {
        wgpu::BindingResource::TextureView(&self.full_view)
    }

    /// Asyncronously reads the contents of the [`GpuImage`] into a [`Vec`].
    ///
    /// In order for this future to resolve, [`Framework::poll`] or [`Framework::blocking_poll`]
    /// must be invoked.
    pub async fn read_async(&self) -> GpuResult<Vec<u8>> {
        use std::num::NonZeroU32;

        let staging = self
            .fw
            .create_staging_buffer((4 * self.size.width * self.size.height) as usize);

        let mut encoder = self
            .fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Image copy"),
            });

        let copy_texture = wgpu::ImageCopyTexture {
            aspect: wgpu::TextureAspect::All,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            texture: &self.texture,
        };

        let copy_buffer = wgpu::ImageCopyBuffer {
            buffer: &staging,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(4 * self.size.width),
                // rows_per_image: NonZeroU32::new(self.size.height),
                rows_per_image: None,
            },
        };

        encoder.copy_texture_to_buffer(copy_texture, copy_buffer, self.size);

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

    /// Blocking read of the content of the [`GpuImage`] into a [`Vec`].
    pub fn read(&self) -> GpuResult<Vec<u8>> {
        use std::num::NonZeroU32;

        let staging = self
            .fw
            .create_staging_buffer((4 * self.size.width * self.size.height) as usize);

        let mut encoder = self
            .fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Image copy"),
            });

        let copy_texture = wgpu::ImageCopyTexture {
            aspect: wgpu::TextureAspect::All,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            texture: &self.texture,
        };

        let copy_buffer = wgpu::ImageCopyBuffer {
            buffer: &staging,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(4 * self.size.width),
                // rows_per_image: NonZeroU32::new(self.size.height),
                rows_per_image: None,
            },
        };

        encoder.copy_texture_to_buffer(copy_texture, copy_buffer, self.size);

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

    /// Writes the `img_bytes` bytes into the [`GpuImage`] immediately.
    /// The image is format specified at the [`GpuImage`] creation.
    pub fn write(&mut self, img_bytes: &[u8]) {
        use std::num::NonZeroU32;

        self.fw.queue.write_texture(
            self.texture.as_image_copy(),
            img_bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(NonZeroU32::new(4 * self.size.width).unwrap()), // TODO: change 4 for img format pixel byte size
                rows_per_image: None,
            },
            self.size,
        );

        let encoder = self
            .fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Buffer write"),
            });

        self.fw.queue.submit(Some(encoder.finish()));
    }
}
