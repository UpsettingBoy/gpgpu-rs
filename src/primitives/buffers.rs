use std::marker::PhantomData;

use thiserror::Error;
use wgpu::{util::DeviceExt, MapMode};

use crate::{future::GpuMapFuture, primitives::BufOps, GpuBuffer, GpuUniformBuffer};

// TODO https://github.com/bitflags/bitflags/issues/180
// TODO Unsure wether MAP_READ and MAP_WRITE should only be present on certain buffers
const GPU_BUFFER_USAGES: wgpu::BufferUsages = wgpu::BufferUsages::from_bits_truncate(
    wgpu::BufferUsages::STORAGE.bits()
        | wgpu::BufferUsages::COPY_SRC.bits()
        | wgpu::BufferUsages::COPY_DST.bits()
        | wgpu::BufferUsages::MAP_READ.bits()
        | wgpu::BufferUsages::MAP_WRITE.bits(),
);
const GPU_UNIFORM_USAGES: wgpu::BufferUsages = wgpu::BufferUsages::from_bits_truncate(
    wgpu::BufferUsages::UNIFORM.bits() | wgpu::BufferUsages::COPY_DST.bits(),
);

pub type BufferResult<T> = Result<T, BufferError>;

#[derive(Error, Debug, Clone)]
pub enum BufferError {
    #[error(transparent)]
    AsyncMapError(#[from] wgpu::BufferAsyncError),
}

impl<'fw, T> BufOps<'fw, T> for GpuBuffer<'fw, T>
where
    T: bytemuck::Pod,
{
    fn size(&self) -> u64 {
        self.buf.size()
    }

    fn as_wgpu_buffer(&self) -> &wgpu::Buffer {
        &self.buf
    }

    fn with_capacity(fw: &'fw crate::Framework, capacity: u64) -> Self {
        let size = capacity * std::mem::size_of::<T>() as u64;
        let buf = fw.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuBuffer::with_capacity"),
            size,
            usage: GPU_BUFFER_USAGES,
            mapped_at_creation: false,
        });

        Self {
            fw,
            buf,
            marker: PhantomData,
        }
    }

    fn from_slice(fw: &'fw crate::Framework, slice: &[T]) -> Self {
        let buf = fw
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GpuBuffer::from_slice"),
                contents: bytemuck::cast_slice(slice),
                usage: GPU_BUFFER_USAGES,
            });

        Self {
            fw,
            buf,
            marker: PhantomData,
        }
    }

    fn from_wgpu_buffer(fw: &'fw crate::Framework, buf: wgpu::Buffer) -> Self {
        Self {
            fw,
            buf,
            marker: PhantomData,
        }
    }

    fn into_wgpu_buffer(self) -> wgpu::Buffer {
        self.buf
    }
}

impl<'fw, T> GpuBuffer<'fw, T>
where
    T: bytemuck::Pod,
{
    /// Pulls some elements from the [`GpuBuffer`] into `buf`, returning how many elements were read.
    pub async fn read(&self, buf: &mut [T]) -> BufferResult<u64> {
        let output_size = (buf.len() * std::mem::size_of::<T>()) as u64;
        let read_size = if output_size > self.size() {
            self.size()
        } else {
            output_size
        };

        let buf_slice = self.buf.slice(..read_size);

        GpuMapFuture::new(&self.fw.device, buf_slice, MapMode::Read).await?;
        buf.copy_from_slice(bytemuck::cast_slice(&buf_slice.get_mapped_range()));

        Ok(read_size)
    }

    /// Pulls all the elements from the [`GpuBuffer`] into a [`Vec`].
    pub async fn read_vec(&self) -> BufferResult<Vec<T>> {
        // Safety: Since T is Pod: Zeroed + ... it is safe to use zeroed() to init it.
        let mut buf = vec![unsafe { std::mem::zeroed() }; self.capacity() as usize];
        self.read(&mut buf).await?;

        Ok(buf)
    }

    /// Blocking version of `GpuBuffer::read()`.
    pub fn read_blocking(&self, buf: &mut [T]) -> BufferResult<u64> {
        futures::executor::block_on(self.read(buf))
    }

    /// Blocking version of `GpuBuffer::read_vec()`.
    pub fn read_vec_blocking(&self) -> BufferResult<Vec<T>> {
        futures::executor::block_on(self.read_vec())
    }

    /// Attempts to write a slice into this [`GpuBuffer`], returning how many elements were written.
    ///
    /// This function will attempt to write the entire contents of `buf` unless its capacity
    /// exceeds the one of the source buffer, in which case `GpuBuffer::capacity()` elements are written.
    pub async fn write(&self, buf: &[T]) -> BufferResult<u64> {
        let input_size = (buf.len() * std::mem::size_of::<T>()) as u64;
        let write_size = if input_size > self.size() {
            self.size()
        } else {
            input_size
        };

        let write_slice = self.buf.slice(..write_size);
        GpuMapFuture::new(&self.fw.device, write_slice, MapMode::Write).await?;

        Ok(write_size)
    }
}

impl<'fw, T> BufOps<'fw, T> for GpuUniformBuffer<'fw, T>
where
    T: bytemuck::Pod,
{
    fn size(&self) -> u64 {
        self.buf.size()
    }

    fn as_wgpu_buffer(&self) -> &wgpu::Buffer {
        &self.buf
    }

    fn with_capacity(fw: &'fw crate::Framework, capacity: u64) -> Self {
        let size = capacity * std::mem::size_of::<T>() as u64;

        let buf = fw.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuUniformBuffer::with_capacity"),
            size,
            usage: GPU_UNIFORM_USAGES,
            mapped_at_creation: false,
        });

        Self {
            fw,
            buf,
            marker: PhantomData,
        }
    }

    fn from_slice(fw: &'fw crate::Framework, slice: &[T]) -> Self {
        let buf = fw
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GpuUniformBuffer::from_slice"),
                contents: bytemuck::cast_slice(slice),
                usage: GPU_UNIFORM_USAGES,
            });

        Self {
            fw,
            buf,
            marker: PhantomData,
        }
    }

    fn from_wgpu_buffer(fw: &'fw crate::Framework, buf: wgpu::Buffer) -> Self {
        Self {
            fw,
            buf,
            marker: PhantomData,
        }
    }

    fn into_wgpu_buffer(self) -> wgpu::Buffer {
        self.buf
    }
}

impl<'fw, T> GpuUniformBuffer<'fw, T>
where
    T: bytemuck::Pod,
{
    /// Attempts to write a slice into this [`GpuUniformBuffer`], returning how many elements were written.
    ///
    /// This function will attempt to write the entire contents of `buf` unless its capacity
    /// exceeds the one of the source buffer, in which case `GpuUniformBuffer::capacity()` elements are written.
    pub async fn write(&self, buf: &[T]) -> BufferResult<u64> {
        let input_size = (buf.len() * std::mem::size_of::<T>()) as u64;
        let upload_size = if input_size > self.size() {
            self.size()
        } else {
            input_size
        };

        let buf_slice = self.buf.slice(..upload_size);
        GpuMapFuture::new(&self.fw.device, buf_slice, MapMode::Write).await?;

        buf_slice
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(buf));

        Ok(upload_size)
    }
}
