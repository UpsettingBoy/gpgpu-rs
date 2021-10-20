use crate::{
    primitives::{pixels, PixelInfo},
    GpuImage, GpuResult,
};

use image::ImageBuffer;

pub trait ImageToGpgpu {
    type GpgpuPixel: PixelInfo + GpgpuToImage;
    type NormGpgpuPixel: PixelInfo + GpgpuToImage;
}

pub trait GpgpuToImage {
    type ImgPixel: ::image::Pixel + 'static;
    type ImgPrimitive: ::image::Primitive;
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
                    type ImgPrimitive = <$pixel as ::image::Pixel>::Subpixel;
                }
            )+
        )+
    }
}

gpgpu_to_image_impl! {
    ::image::Rgba<u8>, pixels::Rgba8Uint, pixels::Rgba8UintNorm, pixels::Rgba8Sint, pixels::Rgba8SintNorm
}

image_to_gpgpu_impl! {
    ::image::Rgba<u8>, pixels::Rgba8Uint, pixels::Rgba8UintNorm;
    ::image::Rgba<i8>, pixels::Rgba8Sint, pixels::Rgba8SintNorm
}

impl crate::Framework {
    pub fn create_image_from_image_crate<Pixel, Container>(
        &self,
        img: &ImageBuffer<Pixel, Container>,
    ) -> GpuImage<Pixel::GpgpuPixel>
    where
        Pixel: image::Pixel + ImageToGpgpu + 'static,
        Container: std::ops::Deref<Target = [Pixel::Subpixel]>,
    {
        let (width, height) = img.dimensions();
        let mut output_image = GpuImage::new(self, width, height);

        let bytes = primitive_slice_to_bytes(img);
        output_image.write(bytes);

        output_image
    }

    pub fn create_image_from_image_crate_normalised<Pixel, Container>(
        &self,
        img: &ImageBuffer<Pixel, Container>,
    ) -> GpuImage<Pixel::NormGpgpuPixel>
    where
        Pixel: image::Pixel + ImageToGpgpu + 'static,
        Container: std::ops::Deref<Target = [Pixel::Subpixel]>,
    {
        let (width, height) = img.dimensions();
        let mut output_image = GpuImage::new(self, width, height);

        let bytes = primitive_slice_to_bytes(img);
        output_image.write(bytes);

        output_image
    }
}

impl<'fw, P> GpuImage<'fw, P>
where
    P: PixelInfo + GpgpuToImage,
{
    pub fn read_to_image(
        &self,
    ) -> GpuResult<
        ::image::ImageBuffer<
            P::ImgPixel,
            Vec<<<P as GpgpuToImage>::ImgPixel as image::Pixel>::Subpixel>,
        >,
    > {
        let bytes = self.read()?;
        let container = bytes_to_primitive_vec::<P::ImgPixel>(bytes);

        let img: Result<ImageBuffer<P::ImgPixel, Vec<_>>, Box<dyn std::error::Error>> =
            image::ImageBuffer::from_vec(self.size.width, self.size.height, container)
                .ok_or("Buffer is too small!".into());

        img
    }

    pub async fn read_to_image_async(
        &self,
    ) -> GpuResult<
        ::image::ImageBuffer<
            P::ImgPixel,
            Vec<<<P as GpgpuToImage>::ImgPixel as image::Pixel>::Subpixel>,
        >,
    > {
        let bytes = self.read_async().await?;
        let container = bytes_to_primitive_vec::<P::ImgPixel>(bytes);

        let img: Result<ImageBuffer<P::ImgPixel, Vec<_>>, Box<dyn std::error::Error>> =
            image::ImageBuffer::from_vec(self.size.width, self.size.height, container)
                .ok_or("Buffer is too small!".into());

        img
    }

    pub fn write_from_image(
        &mut self,
        img: &::image::ImageBuffer<
            P::ImgPixel,
            Vec<<<P as GpgpuToImage>::ImgPixel as image::Pixel>::Subpixel>,
        >,
    ) {
        let bytes = primitive_slice_to_bytes(img);
        self.write(bytes);
    }

    pub async fn write_from_image_async(
        &mut self,
        img: &::image::ImageBuffer<
            P::ImgPixel,
            Vec<<<P as GpgpuToImage>::ImgPixel as image::Pixel>::Subpixel>,
        >,
    ) -> GpuResult<()> {
        let bytes = primitive_slice_to_bytes(img);
        self.write_async(bytes).await
    }
}

pub(crate) fn primitive_slice_to_bytes<P>(primitive: &[P]) -> &[u8]
where
    P: image::Primitive,
{
    let times = std::mem::size_of::<P>() / std::mem::size_of::<u8>();

    unsafe {
        // Pointer transmutation (as I would do in C 🤣)
        let input_ptr = primitive.as_ptr();
        let new_ptr: *const u8 = std::mem::transmute(input_ptr);

        std::slice::from_raw_parts(new_ptr, primitive.len() * times)
    }
}

pub(crate) fn bytes_to_primitive_vec<P>(mut bytes: Vec<u8>) -> Vec<P::Subpixel>
where
    P: image::Pixel,
{
    // Fit vector to min possible size
    // Since Vec::shrink_to_fit cannot assure that the inner vector memory is
    // exactly its theorical min possible size, UB? 😢
    bytes.shrink_to_fit();
    let len = bytes.len() / std::mem::size_of::<P::Subpixel>(); // Get num of primitives

    unsafe {
        // Pointer transmutation (as I would do in C 🤣)
        let input_ptr = bytes.as_mut_ptr();
        let new_ptr: *mut P::Subpixel = std::mem::transmute(input_ptr);

        // `bytes` cannot be dropped or a copy of the vector will be required
        std::mem::forget(bytes);

        Vec::from_raw_parts(new_ptr, len, len)
    }
}
