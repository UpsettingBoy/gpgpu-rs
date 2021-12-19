use crate::{
    primitives::{images::ImageResult, pixels, ImgOps, PixelInfo},
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

impl<'fw, Pixel> GpuImage<'fw, Pixel>
where
    Pixel: image::Pixel + ImageToGpgpu + 'static,
    Pixel::Subpixel: bytemuck::Pod,
{
    /// Constructs a new [`GpuImage`] from a [`image::ImageBuffer`].
    pub fn from_image_buffer<Container>(
        fw: &'fw crate::Framework,
        img: &ImageBuffer<Pixel, Container>,
    ) -> ImageResult<GpuImage<'fw, Pixel::GpgpuPixel>>
    where
        Container: std::ops::Deref<Target = [Pixel::Subpixel]>,
    {
        let (width, height) = img.dimensions();
        let mut output_image = GpuImage::new(fw, width, height);

        let bytes = bytemuck::cast_slice(img);
        output_image.write(bytes)?;

        Ok(output_image)
    }

    /// Constructs a new normalised [`GpuImage`] from a [`image::ImageBuffer`].
    pub fn from_image_buffer_normalised<Container>(
        fw: &'fw crate::Framework,
        img: &ImageBuffer<Pixel, Container>,
    ) -> ImageResult<GpuImage<'fw, Pixel::NormGpgpuPixel>>
    where
        Container: std::ops::Deref<Target = [Pixel::Subpixel]>,
    {
        let (width, height) = img.dimensions();
        let mut output_image = GpuImage::new(fw, width, height);

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
    /// Constructs a new [`GpuConstImage`] from a [`image::ImageBuffer`].
    pub fn from_image_buffer<Container>(
        fw: &'fw crate::Framework,
        img: &ImageBuffer<Pixel, Container>,
    ) -> ImageResult<GpuConstImage<'fw, Pixel::GpgpuPixel>>
    where
        Container: std::ops::Deref<Target = [Pixel::Subpixel]>,
    {
        let (width, height) = img.dimensions();
        let mut output_image = GpuConstImage::new(fw, width, height);

        let bytes = bytemuck::cast_slice(img);
        output_image.write(bytes)?;

        Ok(output_image)
    }

    /// Constructs a new normalised [`GpuConstImage`] from a [`image::ImageBuffer`].
    pub fn from_image_buffer_normalised<Container>(
        fw: &'fw crate::Framework,
        img: &ImageBuffer<Pixel, Container>,
    ) -> ImageResult<GpuConstImage<'fw, Pixel::NormGpgpuPixel>>
    where
        Container: std::ops::Deref<Target = [Pixel::Subpixel]>,
    {
        let (width, height) = img.dimensions();
        let mut output_image = GpuConstImage::new(fw, width, height);

        let bytes = bytemuck::cast_slice(img);
        output_image.write(bytes)?;

        Ok(output_image)
    }
}

impl<'fw, P> GpuImage<'fw, P>
where
    P: PixelInfo + GpgpuToImage,
    <<P as GpgpuToImage>::ImgPixel as image::Pixel>::Subpixel: bytemuck::Pod,
{
    /// Pulls some elements from the [`GpuImage`] into buf, returning how many pixels were read.
    pub async fn read_into_image_buffer(
        &self,
        buf: &mut ::image::ImageBuffer<
            P::ImgPixel,
            Vec<<<P as GpgpuToImage>::ImgPixel as image::Pixel>::Subpixel>,
        >,
    ) -> ImageResult<usize> {
        let output_slice = bytemuck::cast_slice_mut(buf);
        self.read(output_slice).await
    }

    /// Blocking version of `GpuImage::read_into_image_buffer()`.
    pub fn read_into_image_buffer_blocking(
        &self,
        buf: &mut ::image::ImageBuffer<
            P::ImgPixel,
            Vec<<<P as GpgpuToImage>::ImgPixel as image::Pixel>::Subpixel>,
        >,
    ) -> ImageResult<usize> {
        futures::executor::block_on(self.read_into_image_buffer(buf))
    }

    /// Pulls all the pixels from the [`GpuImage`] into a [`image::ImageBuffer`].
    pub async fn read_to_image_buffer(
        &self,
    ) -> ImageResult<
        ::image::ImageBuffer<
            P::ImgPixel,
            Vec<<<P as GpgpuToImage>::ImgPixel as image::Pixel>::Subpixel>,
        >,
    > {
        let bytes = self.read_vec().await?;
        let container = bytes_to_primitive_vec::<P::ImgPixel>(bytes);

        let (width, height) = self.dimensions();

        Ok(image::ImageBuffer::from_vec(width, height, container).expect("Cannot fail here."))
    }

    /// Blocking version of `GpuImage::read_to_image_buffer()`.
    pub fn read_to_image_buffer_blocking(
        &self,
    ) -> ImageResult<::image::ImageBuffer<P::ImgPixel, PixelContainer<P>>> {
        futures::executor::block_on(self.read_to_image_buffer())
    }

    /// Writes a buffer into this [`GpuImage`], returning how many pixels were written. The operation is instantly offloaded.
    ///
    /// This function will attempt to write the entire contents of buf, unless its capacity
    /// exceeds the one of the image, in which case the first width * height pixels are written.
    pub fn write_image_buffer(
        &mut self,
        buf: &::image::ImageBuffer<
            P::ImgPixel,
            Vec<<<P as GpgpuToImage>::ImgPixel as image::Pixel>::Subpixel>,
        >,
    ) -> ImageResult<usize> {
        let bytes = bytemuck::cast_slice(buf);
        self.write(bytes)
    }
}

impl<'fw, P> GpuConstImage<'fw, P>
where
    P: PixelInfo + GpgpuToImage,
    <<P as GpgpuToImage>::ImgPixel as image::Pixel>::Subpixel: bytemuck::Pod,
{
    /// Writes a buffer into this [`GpuConstImage`], returning how many pixels were written. The operation is instantly offloaded.
    ///
    /// This function will attempt to write the entire contents of buf, unless its capacity
    /// exceeds the one of the image, in which case the first width * height pixels are written.
    pub fn write_image_buffer(
        &mut self,
        buf: &::image::ImageBuffer<
            P::ImgPixel,
            Vec<<<P as GpgpuToImage>::ImgPixel as image::Pixel>::Subpixel>,
        >,
    ) -> ImageResult<usize> {
        let bytes = bytemuck::cast_slice(buf);
        self.write(bytes)
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
