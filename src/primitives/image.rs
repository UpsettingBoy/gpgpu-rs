use super::{generic_image::GenericImage, PixelInfo};
use crate::{GpuConstImage, GpuImage, GpuResult};

impl<'fw, P> GpuImage<'fw, P>
where
    P: PixelInfo,
{
    /// Gets the inner [`wgpu::Texture`] of the [`GpuImage`].
    pub fn get_wgpu_texture(&self) -> &wgpu::Texture {
        self.0.get_wgpu_texture()
    }

    /// Consumes this [`GpuImage`] into a [`wgpu::Texture`] and its `dimensions`.
    pub fn into_wgpu_texture(self) -> (wgpu::Texture, wgpu::Extent3d) {
        self.0.into_wgpu_texture()
    }

    /// Gets the [`wgpu::Extent3d`] of the [`GpuImage`].
    pub fn get_wgpu_extent3d(&self) -> wgpu::Extent3d {
        self.0.get_wgpu_extent3d()
    }

    /// Gets the width and height of the [`GpuImage`].
    pub fn dimensions(&self) -> (u32, u32) {
        self.0.dimensions()
    }

    /// Creates an empty [`GpuImage`] with the desired `width` and `height`.
    pub fn new(fw: &'fw crate::Framework, width: u32, height: u32) -> Self {
        Self(GenericImage::new(fw, width, height))
    }

    /// Creates a new `GpuImage` from an image's raw bytes (`data`) and its dimensions.
    pub fn from_raw_bytes(fw: &'fw crate::Framework, width: u32, height: u32, data: &[u8]) -> Self {
        Self(GenericImage::from_raw_bytes(fw, width, height, data))
    }

    /// Creates a new [`GpuImage`] from a [`wgpu::Texture`] and its dimensions.
    ///
    /// `texture` must have `wgpu::TextureUsages::STORAGE_BINDING`, `wgpu::TextureUsages::COPY_SRC`,
    /// `wgpu::TextureUsages::COPY_DST` and `wgpu::TextureUsages::TEXTURE_BINDING` usages.
    pub fn from_wgpu_texture(
        fw: &'fw crate::Framework,
        texture: wgpu::Texture,
        width: u32,
        height: u32,
    ) -> Self {
        Self(GenericImage::from_wgpu_texture(fw, texture, width, height))
    }

    /// Asyncronously reads the contents of the [`GpuImage`] into a [`Vec`].
    ///
    /// In order for this future to resolve, [`Framework::poll`](crate::Framework::poll) or
    /// [`Framework::blocking_poll`](crate::Framework::blocking_poll)
    /// must be invoked.
    pub async fn read_async(&self) -> GpuResult<Vec<u8>> {
        self.0.read_async().await
    }

    /// Blocking read of the content of the [`GpuImage`] into a [`Vec`].
    pub fn read(&self) -> GpuResult<Vec<u8>> {
        self.0.read()
    }

    /// Writes immediately the `img_bytes` bytes into the [`GpuImage`].
    /// The image is format specified at the [`GpuImage`] creation.
    pub fn write(&mut self, img_bytes: &[u8]) {
        self.0.write(img_bytes)
    }

    /// Asyncronously writes `img_bytes` into the [`GpuImage`].
    ///
    /// In order for this future to resolve, [`Framework::poll`](crate::Framework::poll) or
    /// [`Framework::blocking_poll`](crate::Framework::blocking_poll)
    /// must be invoked.
    pub async fn write_async(&mut self, img_bytes: &[u8]) -> GpuResult<()> {
        self.0.write_async(img_bytes).await
    }
}

impl<'fw, P> GpuConstImage<'fw, P>
where
    P: PixelInfo,
{
    /// Gets the inner [`wgpu::Texture`] of the [`GpuConstImage`].
    pub fn get_wgpu_texture(&self) -> &wgpu::Texture {
        self.0.get_wgpu_texture()
    }

    /// Gets the [`wgpu::Extent3d`] of the [`GpuConstImage`].
    pub fn get_wgpu_extent3d(&self) -> wgpu::Extent3d {
        self.0.get_wgpu_extent3d()
    }

    /// Gets the width and height of the [`GpuConstImage`].
    pub fn dimensions(&self) -> (u32, u32) {
        self.0.dimensions()
    }

    /// Creates an empty [`GpuConstImage`] with the desired `width` and `height`.
    pub fn new(fw: &'fw crate::Framework, width: u32, height: u32) -> Self {
        Self(GenericImage::new(fw, width, height))
    }

    /// Creates a new [`GpuConstImage`] from an image's raw bytes (`data`) and its dimensions.
    pub fn from_raw_bytes(fw: &'fw crate::Framework, width: u32, height: u32, data: &[u8]) -> Self {
        Self(GenericImage::from_raw_bytes(fw, width, height, data))
    }

    /// Creates a new [`GpuConstImage`] from a [`wgpu::Texture`] and its dimensions.
    ///
    /// `texture` must have `wgpu::TextureUsages::STORAGE_BINDING`, `wgpu::TextureUsages::COPY_SRC`,
    /// `wgpu::TextureUsages::COPY_DST` and `wgpu::TextureUsages::TEXTURE_BINDING` usages.
    pub fn from_wgpu_texture(
        fw: &'fw crate::Framework,
        texture: wgpu::Texture,
        width: u32,
        height: u32,
    ) -> Self {
        Self(GenericImage::from_wgpu_texture(fw, texture, width, height))
    }

    /// Writes immediately the `img_bytes` bytes into the [`GpuConstImage`].
    /// The image is format specified at the [`GpuConstImage`] creation.
    pub fn write(&mut self, img_bytes: &[u8]) {
        self.0.write(img_bytes)
    }

    /// Asyncronously writes `img_bytes` into the [`GpuConstImage`].
    ///
    /// In order for this future to resolve, [`Framework::poll`](crate::Framework::poll) or
    /// [`Framework::blocking_poll`](crate::Framework::blocking_poll)
    /// must be invoked.
    pub async fn write_async(&mut self, img_bytes: &[u8]) -> GpuResult<()> {
        self.0.write_async(img_bytes).await
    }
}
