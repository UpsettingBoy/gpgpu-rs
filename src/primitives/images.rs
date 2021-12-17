use std::marker::PhantomData;

use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::{BufOps, GpuBuffer, GpuConstImage, GpuImage};

use super::{ImgOps, PixelInfo};

// TODO https://github.com/bitflags/bitflags/issues/180
const GPU_IMAGE_USAGES: wgpu::TextureUsages = wgpu::TextureUsages::from_bits_truncate(
    wgpu::TextureUsages::STORAGE_BINDING.bits()
        | wgpu::TextureUsages::COPY_SRC.bits()
        | wgpu::TextureUsages::COPY_DST.bits(),
);
const GPU_CONST_IMAGE_USAGES: wgpu::TextureUsages = wgpu::TextureUsages::from_bits_truncate(
    wgpu::TextureUsages::TEXTURE_BINDING.bits() | wgpu::TextureUsages::COPY_DST.bits(),
);

pub type ImageResult<T> = Result<T, ImageError>;

#[derive(Error, Debug)]
pub enum ImageError {
    #[error(transparent)]
    BufferError(#[from] crate::primitives::buffers::BufferError),
    #[error(
        "Buffer is too small (required size {required} bytes, current size {current} bytes). "
    )]
    BufferTooSmall { required: usize, current: usize },
    #[error("Buffer does not contains an integer number of pixels.")]
    NotIntegerPixelNumber,
    #[error("Buffer does not contains an integer number of rows.")]
    NotIntegerRowNumber,
}

impl<'fw, P> ImgOps<'fw> for GpuImage<'fw, P>
where
    P: PixelInfo,
{
    fn as_gpu_texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    fn get_wgpu_extent3d(&self) -> wgpu::Extent3d {
        self.size
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.size.width, self.size.height)
    }

    fn new(fw: &'fw crate::Framework, width: u32, height: u32) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let format = P::wgpu_format();

        let texture = fw.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("GpuImage::new"),
            size,
            dimension: wgpu::TextureDimension::D2,
            mip_level_count: 1,
            sample_count: 1,
            format,
            usage: GPU_IMAGE_USAGES,
        });

        let full_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            fw,
            texture,
            size,
            full_view,
            pixel: PhantomData,
        }
    }

    fn from_bytes(fw: &'fw crate::Framework, data: &[u8], width: u32, height: u32) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let format = P::wgpu_format();

        let texture = fw.device.create_texture_with_data(
            &fw.queue,
            &wgpu::TextureDescriptor {
                label: Some("GpuImage::from_bytes"),
                size,
                dimension: wgpu::TextureDimension::D2,
                mip_level_count: 1,
                sample_count: 1,
                format,
                usage: GPU_IMAGE_USAGES,
            },
            data,
        );

        let full_view = texture.create_view(&Default::default());

        Self {
            fw,
            texture,
            size,
            full_view,
            pixel: PhantomData,
        }
    }

    /// Constructs an image from a [`wgpu::Texture`] and its [`wgpu::Extent3d`].
    ///
    /// # Safety
    /// If any of the following conditions are not satisfied, the image will
    /// panic at any time during its usage.
    /// - `texture` needs to be `wgpu::TextureUsages::STORAGE_BINDING`, `wgpu::TextureUsages::COPY_SRC`,
    ///   and `wgpu::TextureUsages::COPY_DST`` usable.
    /// - `T` needs to be the exact same codification `texture` is.
    /// - `dimensions` needs to have the exact `width` and `height` of `texture` and `depth_or_array_layers = 1`
    fn from_gpu_parts(
        fw: &'fw crate::Framework,
        texture: wgpu::Texture,
        dimensions: wgpu::Extent3d,
    ) -> Self {
        let full_view = texture.create_view(&Default::default());

        Self {
            fw,
            texture,
            size: dimensions,
            full_view,
            pixel: PhantomData,
        }
    }

    fn into_gpu_parts(self) -> (wgpu::Texture, wgpu::Extent3d) {
        (self.texture, self.size)
    }
}

impl<'fw, P> GpuImage<'fw, P>
where
    P: PixelInfo,
{
    /// Pulls some elements from the [`GpuImage`] into `buf`, returning how many pixels were read.
    pub async fn read(&self, buf: &mut [u8]) -> ImageResult<usize> {
        use std::num::NonZeroU32;

        let (width, height) = self.dimensions();

        let buf_pixels = buf.len() / P::byte_size();
        let img_pixels = (width * height) as usize * P::byte_size();

        if buf_pixels < img_pixels {
            return Err(ImageError::BufferTooSmall {
                required: img_pixels,
                current: buf_pixels,
            });
        }

        let bytes_per_pixel = P::byte_size() as u32;
        let unpadded_bytes_per_row = self.size.width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding;

        let staging_size = (padded_bytes_per_row * self.size.height) as usize;

        let staging = self.fw.create_download_staging_buffer(staging_size);

        let mut encoder = self
            .fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GenericImage::read_async"),
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

        let gpu_buf = GpuBuffer::from_gpu_parts(self.fw, staging, staging_size);
        Ok(gpu_buf.read(buf).await? / P::byte_size())
    }

    /// Pulls all the pixels from the [`GpuImage`] into a [`Vec`].
    pub async fn read_vec(&self) -> ImageResult<Vec<u8>> {
        let (width, height) = self.dimensions();
        let img_pixels = (width * height) as usize * P::byte_size();

        let mut buf = vec![0u8; img_pixels];
        self.read(&mut buf).await?;

        Ok(buf)
    }

    /// Blocking version of `GpuImage::read()`.
    pub fn read_blocking(&self, buf: &mut [u8]) -> ImageResult<usize> {
        futures::executor::block_on(self.read(buf))
    }

    /// Blocking version of `GpuImage::read_vec()`.
    pub fn read_vec_blocking(&self) -> ImageResult<Vec<u8>> {
        futures::executor::block_on(self.read_vec())
    }

    /// Writes a buffer into this [`GpuImage`], returning how many pixels were written. The operation is instantly offloaded.
    ///
    /// This function will attempt to write the entire contents of `buf`, unless its capacity
    /// exceeds the one of the image, in which case the first `width * height` pixels are written.
    pub fn write(&mut self, buf: &[u8]) -> ImageResult<usize> {
        use std::num::NonZeroU32;

        if buf.len() % P::byte_size() != 0 {
            return Err(ImageError::NotIntegerPixelNumber);
        }
        if buf.len() % (P::byte_size() * self.size.width as usize) != 0 {
            return Err(ImageError::NotIntegerRowNumber);
        }

        let image_bytes = P::byte_size() * (self.size.width * self.size.height) as usize;

        self.fw.queue.write_texture(
            self.texture.as_image_copy(),
            buf,
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

        Ok(if buf.len() > image_bytes {
            image_bytes
        } else {
            buf.len()
        })
    }
}

impl<'fw, P> ImgOps<'fw> for GpuConstImage<'fw, P>
where
    P: PixelInfo,
{
    fn as_gpu_texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    fn get_wgpu_extent3d(&self) -> wgpu::Extent3d {
        self.size
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.size.width, self.size.height)
    }

    fn new(fw: &'fw crate::Framework, width: u32, height: u32) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let format = P::wgpu_format();

        let texture = fw.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("GpuConstImage::new"),
            size,
            dimension: wgpu::TextureDimension::D2,
            mip_level_count: 1,
            sample_count: 1,
            format,
            usage: GPU_CONST_IMAGE_USAGES,
        });

        let full_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            fw,
            texture,
            size,
            full_view,
            pixel: PhantomData,
        }
    }

    fn from_bytes(fw: &'fw crate::Framework, data: &[u8], width: u32, height: u32) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let format = P::wgpu_format();

        let texture = fw.device.create_texture_with_data(
            &fw.queue,
            &wgpu::TextureDescriptor {
                label: Some("GpuConstImage::from_bytes"),
                size,
                dimension: wgpu::TextureDimension::D2,
                mip_level_count: 1,
                sample_count: 1,
                format,
                usage: GPU_CONST_IMAGE_USAGES,
            },
            data,
        );

        let full_view = texture.create_view(&Default::default());

        Self {
            fw,
            texture,
            size,
            full_view,
            pixel: PhantomData,
        }
    }

    /// Constructs an image from a [`wgpu::Texture`] and its [`wgpu::Extent3d`].
    ///
    /// # Safety
    /// If any of the following conditions are not satisfied, the image will
    /// panic at any time during its usage.
    /// - `texture` needs to be `wgpu::TextureUsages::TEXTURE_BINDING` and `wgpu::TextureUsages::COPY_SRC` usable.
    /// - `T` needs to be the exact same codification `texture` is.
    /// - `dimensions` needs to have the exact `width` and `height` of `texture` and `depth_or_array_layers = 1`
    fn from_gpu_parts(
        fw: &'fw crate::Framework,
        texture: wgpu::Texture,
        dimensions: wgpu::Extent3d,
    ) -> Self {
        let full_view = texture.create_view(&Default::default());

        Self {
            fw,
            texture,
            size: dimensions,
            full_view,
            pixel: PhantomData,
        }
    }

    fn into_gpu_parts(self) -> (wgpu::Texture, wgpu::Extent3d) {
        (self.texture, self.size)
    }
}

impl<'fw, P> GpuConstImage<'fw, P>
where
    P: PixelInfo,
{
    /// Writes a buffer into this [`GpuConstImage`], returning how many pixels were written. The operation is instantly offloaded.
    ///
    /// This function will attempt to write the entire contents of `buf`, unless its capacity
    /// exceeds the one of the image, in which case the first `width * height` pixels are written.
    pub fn write(&mut self, buf: &[u8]) -> ImageResult<usize> {
        use std::num::NonZeroU32;

        if buf.len() % P::byte_size() != 0 {
            return Err(ImageError::NotIntegerPixelNumber);
        }
        if buf.len() % (P::byte_size() * self.size.width as usize) != 0 {
            return Err(ImageError::NotIntegerRowNumber);
        }

        let image_bytes = P::byte_size() * (self.size.width * self.size.height) as usize;

        self.fw.queue.write_texture(
            self.texture.as_image_copy(),
            buf,
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
                label: Some("GpuConstImage::write"),
            });

        self.fw.queue.submit(Some(encoder.finish()));

        Ok(if buf.len() > image_bytes {
            image_bytes
        } else {
            buf.len()
        })
    }
}
