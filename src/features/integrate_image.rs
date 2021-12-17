use crate::{
    primitives::{
        generic_image::{GenericImage, ImageResult},
        pixels, PixelInfo,
    },
    GpuConstImage, GpuImage,
};

use image::ImageBuffer;

/// Contains information about the `image::ImageBuffer` -> `gpgpu::GpuImage` or `gpgpu::GpuConstImage` images conversion.
pub trait ImageToGpgpu {
    type GpgpuPixel: PixelInfo + GpgpuToImage;
    type NormGpgpuPixel: PixelInfo + GpgpuToImage;
}

/// Contains information about the `gpgpu::GpuImage` or `gpgpu::GpuConstImage`-> `image::ImageBuffer` images conversion.
pub trait GpgpuToImage {
    type ImgPixel: ::image::Pixel + 'static;
}

macro_rules! image_to_gpgpu_impl {
    ($($img_pixel:ty, $pixel:ty, $norm:ty);+) => {
        $(
            impl ImageToGpgpu for $img_pixel {
                type GpgpuPixel = $pixel;
                type NormGpgpuPixel = $norm;
            }
        )+
    }
}

macro_rules! gpgpu_to_image_impl {
    ($($pixel:ty, $($gpgpu_pixel:ty),+);+) => {
        $(
            $(
                impl GpgpuToImage for $gpgpu_pixel {
                    type ImgPixel =  $pixel;
                }
            )+
        )+
    }
}

gpgpu_to_image_impl! {
    ::image::Rgba<u8>, pixels::Rgba8Uint, pixels::Rgba8UintNorm;
    ::image::Rgba<i8>, pixels::Rgba8Sint, pixels::Rgba8SintNorm
    // ::image::Luma<u8>, pixels::Luma8, pixels::Luma8Norm
}

image_to_gpgpu_impl! {
    ::image::Rgba<u8>, pixels::Rgba8Uint, pixels::Rgba8UintNorm;
    ::image::Rgba<i8>, pixels::Rgba8Sint, pixels::Rgba8SintNorm
    // ::image::Luma<u8>, pixels::Luma8, pixels::Luma8Norm
}

type PixelContainer<P> = Vec<<<P as GpgpuToImage>::ImgPixel as image::Pixel>::Subpixel>;

impl<'fw, Pixel> GenericImage<'fw, Pixel>
where
    Pixel: image::Pixel + ImageToGpgpu + 'static,
    Pixel::Subpixel: bytemuck::Pod,
{
    /// Creates a new [`GenericImage`] from a [`image::ImageBuffer`].
    pub fn from_image_buffer<Container>(
        fw: &'fw crate::Framework,
        img: &ImageBuffer<Pixel, Container>,
    ) -> ImageResult<GenericImage<'fw, Pixel::GpgpuPixel>>
    where
        Container: std::ops::Deref<Target = [Pixel::Subpixel]>,
    {
        let (width, height) = img.dimensions();
        let mut output_image = GenericImage::new(fw, width, height);

        let bytes = bytemuck::cast_slice(img);
        output_image.write(bytes)?;

        Ok(output_image)
    }

    /// Creates a new normalised [`GenericImage`] from a [`image::ImageBuffer`].
    pub fn from_image_buffer_normalised<Container>(
        fw: &'fw crate::Framework,
        img: &ImageBuffer<Pixel, Container>,
    ) -> ImageResult<GenericImage<'fw, Pixel::NormGpgpuPixel>>
    where
        Container: std::ops::Deref<Target = [Pixel::Subpixel]>,
    {
        let (width, height) = img.dimensions();
        let mut output_image = GenericImage::new(fw, width, height);

        let bytes = bytemuck::cast_slice(img);
        output_image.write(bytes)?;

        Ok(output_image)
    }
}

impl<'fw, Pixel> GpuImage<'fw, Pixel>
where
    Pixel: image::Pixel + ImageToGpgpu + 'static,
    Pixel::Subpixel: bytemuck::Pod,
{
    /// Creates a new [`GpuImage`] from a [`image::ImageBuffer`].
    pub fn from_image_buffer<Container>(
        fw: &'fw crate::Framework,
        img: &ImageBuffer<Pixel, Container>,
    ) -> ImageResult<GpuImage<'fw, Pixel::GpgpuPixel>>
    where
        Container: std::ops::Deref<Target = [Pixel::Subpixel]>,
    {
        let (width, height) = img.dimensions();
        let mut output_image = GenericImage::new(fw, width, height);

        let bytes = bytemuck::cast_slice(img);
        output_image.write(bytes)?;

        Ok(output_image)
    }

    /// Creates a new normalised [`GpuImage`] from a [`image::ImageBuffer`].
    pub fn from_image_buffer_normalised<Container>(
        fw: &'fw crate::Framework,
        img: &ImageBuffer<Pixel, Container>,
    ) -> ImageResult<GpuImage<'fw, Pixel::NormGpgpuPixel>>
    where
        Container: std::ops::Deref<Target = [Pixel::Subpixel]>,
    {
        let (width, height) = img.dimensions();
        let mut output_image = GenericImage::new(fw, width, height);

        let bytes = bytemuck::cast_slice(img);
        output_image.write(bytes)?;

        Ok(output_image)
    }
}

impl<'fw, Pixel> GpuConstImage<'fw, Pixel>
where
    Pixel: image::Pixel + ImageToGpgpu + 'static,
    Pixel::Subpixel: bytemuck::Pod,
{
    /// Creates a new [`GpuConstImage`] from a [`image::ImageBuffer`].
    pub fn from_image_buffer<Container>(
        fw: &'fw crate::Framework,
        img: &ImageBuffer<Pixel, Container>,
    ) -> ImageResult<GpuConstImage<'fw, Pixel::GpgpuPixel>>
    where
        Container: std::ops::Deref<Target = [Pixel::Subpixel]>,
    {
        let (width, height) = img.dimensions();
        let mut output_image = GenericImage::new(fw, width, height);

        let bytes = bytemuck::cast_slice(img);
        output_image.write(bytes)?;

        Ok(output_image)
    }

    /// Creates a new normalised [`GpuConstImage`] from a [`image::ImageBuffer`].
    pub fn from_image_buffer_normalised<Container>(
        fw: &'fw crate::Framework,
        img: &ImageBuffer<Pixel, Container>,
    ) -> ImageResult<GpuConstImage<'fw, Pixel::NormGpgpuPixel>>
    where
        Container: std::ops::Deref<Target = [Pixel::Subpixel]>,
    {
        let (width, height) = img.dimensions();
        let mut output_image = GenericImage::new(fw, width, height);

        let bytes = bytemuck::cast_slice(img);
        output_image.write(bytes)?;

        Ok(output_image)
    }
}

impl<'fw, P> GenericImage<'fw, P>
where
    P: PixelInfo + GpgpuToImage,
    <<P as GpgpuToImage>::ImgPixel as image::Pixel>::Subpixel: bytemuck::Pod,
{
    /// Blocking read of the [`GpuImage`], creating a new [`image::GenericImage`] as output.
    pub fn read_to_image_buffer(
        &self,
    ) -> ImageResult<::image::ImageBuffer<P::ImgPixel, PixelContainer<P>>> {
        let bytes = self.read()?;
        let container = bytes_to_primitive_vec::<P::ImgPixel>(bytes);

        let (width, height) = self.dimensions();

        Ok(image::ImageBuffer::from_vec(width, height, container).expect("Cannot fail here."))
    }

    /// Asyncronously read of the [`GenericImage`], creating a new [`image::ImageBuffer`] as output.
    ///
    /// In order for this future to resolve, [`Framework::poll`](crate::Framework::poll) or
    /// [`Framework::blocking_poll`](crate::Framework::blocking_poll)
    /// must be invoked.
    pub async fn read_to_image_buffer_async(
        &self,
    ) -> ImageResult<
        ::image::ImageBuffer<
            P::ImgPixel,
            Vec<<<P as GpgpuToImage>::ImgPixel as image::Pixel>::Subpixel>,
        >,
    > {
        let bytes = self.read_async().await?;
        let container = bytes_to_primitive_vec::<P::ImgPixel>(bytes);

        let (width, height) = self.dimensions();

        Ok(image::ImageBuffer::from_vec(width, height, container).expect("Cannot fail here."))
    }

    /// Writes the [`image::ImageBuffer`] `img` into the [`GenericImage`].
    pub fn write_from_image(
        &mut self,
        img: &::image::ImageBuffer<
            P::ImgPixel,
            Vec<<<P as GpgpuToImage>::ImgPixel as image::Pixel>::Subpixel>,
        >,
    ) -> ImageResult<()> {
        let bytes = bytemuck::cast_slice(img);
        self.write(bytes)?;

        Ok(())
    }

    /// Asyncronously writes the [`image::ImageBuffer`] `img` into the [`GenericImage`].
    ///     
    /// In order for this future to resolve, [`Framework::poll`](crate::Framework::poll) or [`Framework::blocking_poll`](crate::Framework::blocking_poll)
    /// must be invoked.
    pub async fn write_from_image_buffer_async(
        &mut self,
        img: &::image::ImageBuffer<
            P::ImgPixel,
            Vec<<<P as GpgpuToImage>::ImgPixel as image::Pixel>::Subpixel>,
        >,
    ) -> ImageResult<()> {
        let bytes = bytemuck::cast_slice(img);
        self.write_async(bytes).await
    }
}

impl<'fw, P> GpuImage<'fw, P>
where
    P: PixelInfo + GpgpuToImage,
    <<P as GpgpuToImage>::ImgPixel as image::Pixel>::Subpixel: bytemuck::Pod,
{
    /// Blocking read of the [`GpuImage`], creating a new [`image::ImageBuffer`] as output.
    pub fn read_to_image_buffer(
        &self,
    ) -> ImageResult<::image::ImageBuffer<P::ImgPixel, PixelContainer<P>>> {
        self.0.read_to_image_buffer()
    }

    /// Asyncronously read of the [`GpuImage`], creating a new [`image::ImageBuffer`] as output.
    ///
    /// In order for this future to resolve, [`Framework::poll`](crate::Framework::poll) or [`Framework::blocking_poll`](crate::Framework::blocking_poll)
    /// must be invoked.
    pub async fn read_to_image_buffer_async(
        &self,
    ) -> ImageResult<
        ::image::ImageBuffer<
            P::ImgPixel,
            Vec<<<P as GpgpuToImage>::ImgPixel as image::Pixel>::Subpixel>,
        >,
    > {
        self.0.read_to_image_buffer_async().await
    }

    /// Writes immediately the [`image::ImageBuffer`] `img` into the [`GpuImage`].
    pub fn write_from_image_buffer(
        &mut self,
        img: &::image::ImageBuffer<
            P::ImgPixel,
            Vec<<<P as GpgpuToImage>::ImgPixel as image::Pixel>::Subpixel>,
        >,
    ) -> ImageResult<()> {
        self.0.write_from_image(img)
    }

    /// Asyncronously writes the [`image::ImageBuffer`] `img` into the [`GpuImage`].
    ///     
    /// In order for this future to resolve, [`Framework::poll`](crate::Framework::poll) or
    /// [`Framework::blocking_poll`](crate::Framework::blocking_poll)
    /// must be invoked.
    pub async fn write_from_image_buffer_async(
        &mut self,
        img: &::image::ImageBuffer<
            P::ImgPixel,
            Vec<<<P as GpgpuToImage>::ImgPixel as image::Pixel>::Subpixel>,
        >,
    ) -> ImageResult<()> {
        self.0.write_from_image_buffer_async(img).await
    }
}

impl<'fw, P> GpuConstImage<'fw, P>
where
    P: PixelInfo + GpgpuToImage,
    <<P as GpgpuToImage>::ImgPixel as image::Pixel>::Subpixel: bytemuck::Pod,
{
    /// Writes immediately the [`image::ImageBuffer`] `img` into the [`GpuConstImage`].
    pub fn write_from_image_buffer(
        &mut self,
        img: &::image::ImageBuffer<
            P::ImgPixel,
            Vec<<<P as GpgpuToImage>::ImgPixel as image::Pixel>::Subpixel>,
        >,
    ) -> ImageResult<()> {
        self.0.write_from_image(img)
    }

    /// Asyncronously writes the [`image::ImageBuffer`] `img` into the [`GpuConstImage`].
    ///     
    /// In order for this future to resolve, [`Framework::poll`](crate::Framework::poll) or
    /// [`Framework::blocking_poll`](crate::Framework::blocking_poll)
    /// must be invoked.
    pub async fn write_from_image_buffer_async(
        &mut self,
        img: &::image::ImageBuffer<
            P::ImgPixel,
            Vec<<<P as GpgpuToImage>::ImgPixel as image::Pixel>::Subpixel>,
        >,
    ) -> ImageResult<()> {
        self.0.write_from_image_buffer_async(img).await
    }
}

pub(crate) fn bytes_to_primitive_vec<P>(mut bytes: Vec<u8>) -> Vec<P::Subpixel>
where
    P: image::Pixel,
    P::Subpixel: bytemuck::Pod,
{
    // Fit vector to min possible size
    bytes.shrink_to_fit();
    assert_eq!(bytes.len(), bytes.capacity()); // Since `Vec::shrink_to_fit` cannot assure that the inner vector memory is
                                               // exactly its theorical min, we panic if that happened.

    // Original memory won't be dropped. No copy of `bytes` needed 😁
    let mut man_drop = std::mem::ManuallyDrop::new(bytes);

    // `bytemuck` will do the aligment and cast for us.
    let (_, new_type, _) = bytemuck::pod_align_to_mut(&mut man_drop);
    let ptr = new_type.as_mut_ptr();

    unsafe { Vec::from_raw_parts(ptr, new_type.len(), new_type.len()) }
}
