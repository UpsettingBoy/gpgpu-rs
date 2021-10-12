use std::marker::PhantomData;

use crate::{GpuImage, GpuResult};

use super::PixelInfo;

impl<'fw, P> GpuImage<'fw, P>
where
    P: PixelInfo + super::GpgpuImageToImageBuffer,
{
    pub fn read_to_image(&self) -> ::image::ImageBuffer<P::ImgPixel, Vec<P::ImgPrimitive>> {
        todo!()
    }
}

impl<'fw, P> GpuImage<'fw, P>
where
    P: PixelInfo,
{
    pub(crate) fn new(fw: &'fw crate::Framework, width: u32, height: u32) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let format = P::wgpu_format();

        let texture = fw.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            dimension: wgpu::TextureDimension::D2,
            mip_level_count: 1,
            sample_count: 1,
            format,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let full_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        GpuImage {
            fw,
            texture,
            format,
            size,
            full_view,
            _pixel: PhantomData,
        }
    }

    /// Creates a complete [`BindingResource`](wgpu::BindingResource) of the [`GpuImage`].
    pub fn as_binding_resource(&self) -> wgpu::BindingResource {
        wgpu::BindingResource::TextureView(&self.full_view)
    }

    /// Asyncronously reads the contents of the [`GpuImage`] into a [`Vec`].
    ///
    /// In order for this future to resolve, [`Framework::poll`](crate::Framework::poll) or [`Framework::blocking_poll`](crate::Framework::blocking_poll)
    /// must be invoked.
    pub async fn read_async(&self) -> GpuResult<Vec<u8>> {
        use std::num::NonZeroU32;

        let bytes_per_pixel = P::byte_size() as u32;
        let unpadded_bytes_per_row = self.size.width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding;

        let staging = self
            .fw
            .create_staging_buffer((padded_bytes_per_row * self.size.height) as usize);

        let mut encoder = self
            .fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GpuImage::read_async"),
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
                bytes_per_row: NonZeroU32::new(padded_bytes_per_row),
                rows_per_image: None,
            },
        };

        encoder.copy_texture_to_buffer(copy_texture, copy_buffer, self.size);

        self.fw.queue.submit(Some(encoder.finish()));

        let buff_slice = staging.slice(..);
        let buf_future = buff_slice.map_async(wgpu::MapMode::Read);

        buf_future.await?;

        let data = buff_slice.get_mapped_range();
        let result = data
            .chunks(padded_bytes_per_row as usize)
            .flat_map(|row| &row[0..unpadded_bytes_per_row as usize])
            .copied()
            .collect::<Vec<_>>();

        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Blocking read of the content of the [`GpuImage`] into a [`Vec`].
    pub fn read(&self) -> GpuResult<Vec<u8>> {
        use std::num::NonZeroU32;

        let bytes_per_pixel = P::byte_size() as u32;
        let unpadded_bytes_per_row = self.size.width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding;

        let staging = self
            .fw
            .create_staging_buffer((padded_bytes_per_row * self.size.height) as usize);

        let mut encoder = self
            .fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GpuImage::read"),
            });

        let copy_texture = self.texture.as_image_copy();

        let copy_buffer = wgpu::ImageCopyBuffer {
            buffer: &staging,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(padded_bytes_per_row),
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
        let result = data
            .chunks(padded_bytes_per_row as usize)
            .flat_map(|row| &row[0..unpadded_bytes_per_row as usize])
            .copied()
            .collect::<Vec<_>>();

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
                bytes_per_row: Some(
                    NonZeroU32::new(P::byte_size() as u32 * self.size.width).unwrap(),
                ),
                rows_per_image: None,
            },
            self.size,
        );

        let encoder = self
            .fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GpuImage::write"),
            });

        self.fw.queue.submit(Some(encoder.finish()));
    }

    /// Asyncronously writes `img_bytes` into the [`GpuImage`].
    ///
    /// In order for this future to resolve, [`Framework::poll`](crate::Framework::poll) or [`Framework::blocking_poll`](crate::Framework::blocking_poll)
    /// must be invoked.
    pub async fn write_async(&mut self, img_bytes: &[u8]) -> GpuResult<()> {
        use std::num::NonZeroU32;

        let bytes_per_pixel = P::byte_size() as u32;
        let unpadded_bytes_per_row = self.size.width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding;

        let staging = self
            .fw
            .create_staging_buffer((padded_bytes_per_row * self.size.height) as usize);

        let mut encoder = self
            .fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GpuImage::write_async"),
            });

        let staging_view = staging.slice(..);
        staging_view.map_async(wgpu::MapMode::Write).await?;

        let mut staging_map = staging_view.get_mapped_range_mut();
        staging_map
            .chunks_mut(padded_bytes_per_row as usize)
            .enumerate()
            .for_each(|(chunk_id, padded_row)| {
                let start = chunk_id * unpadded_bytes_per_row as usize;
                let end = start + unpadded_bytes_per_row as usize;

                padded_row[0..unpadded_bytes_per_row as usize]
                    .copy_from_slice(&img_bytes[start..end]);
            });

        drop(staging_map);
        staging.unmap();

        let copy_texture = self.texture.as_image_copy();

        let copy_buffer = wgpu::ImageCopyBuffer {
            buffer: &staging,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(padded_bytes_per_row),
                rows_per_image: None,
            },
        };

        encoder.copy_buffer_to_texture(copy_buffer, copy_texture, self.size);

        self.fw.queue.submit(Some(encoder.finish()));

        Ok(())
    }
}
