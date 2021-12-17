use std::marker::PhantomData;

use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::{GpuBuffer, GpuUniformBuffer};

use super::BufOps;

// TODO https://github.com/bitflags/bitflags/issues/180
const GPU_BUFFER_USAGES: wgpu::BufferUsages = wgpu::BufferUsages::from_bits_truncate(
    wgpu::BufferUsages::STORAGE.bits()
        | wgpu::BufferUsages::COPY_SRC.bits()
        | wgpu::BufferUsages::COPY_DST.bits(),
);
const GPU_UNIFORM_USAGES: wgpu::BufferUsages = wgpu::BufferUsages::from_bits_truncate(
    wgpu::BufferUsages::UNIFORM.bits() | wgpu::BufferUsages::COPY_DST.bits(),
);

pub type BufferResult<T> = Result<T, BufferError>;

#[derive(Error, Debug)]
pub enum BufferError {
    #[error(transparent)]
    AsyncMapError(#[from] wgpu::BufferAsyncError),
}

impl<'fw, T> BufOps<'fw, T> for GpuBuffer<'fw, T>
where
    T: bytemuck::Pod,
{
    fn size(&self) -> usize {
        self.size
    }

    fn as_gpu_buffer(&self) -> &wgpu::Buffer {
        &self.buf
    }

    fn with_capacity(fw: &'fw crate::Framework, capacity: usize) -> Self {
        let size = capacity * std::mem::size_of::<T>();
        let buf = fw.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuBuffer::with_capacity"),
            size: size as u64,
            usage: GPU_BUFFER_USAGES,
            mapped_at_creation: false,
        });

        Self {
            fw,
            buf,
            size,
            marker: PhantomData,
        }
    }

    fn from_slice(fw: &'fw crate::Framework, slice: &[T]) -> Self {
        let size = slice.len() * std::mem::size_of::<T>();
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
            size,
            marker: PhantomData,
        }
    }

    fn from_gpu_parts(fw: &'fw crate::Framework, buf: wgpu::Buffer, size: usize) -> Self {
        Self {
            fw,
            buf,
            size,
            marker: PhantomData,
        }
    }

    fn into_gpu_parts(self) -> (wgpu::Buffer, usize) {
        (self.buf, self.size)
    }
}

impl<'fw, T> GpuBuffer<'fw, T>
where
    T: bytemuck::Pod,
{
    /// Pull some elements from the [`GpuBuffer`] into `buf`, returning how many elements were read.
    pub async fn read(&self, buf: &mut [T]) -> BufferResult<usize> {
        let output_size = buf.len() * std::mem::size_of::<T>();
        let download_size = if output_size > self.size {
            self.size
        } else {
            output_size
        };

        let download = wgpu::util::DownloadBuffer::read_buffer(
            &self.fw.device,
            &self.fw.queue,
            &self.buf.slice(..download_size as u64),
        )
        .await?;

        buf.copy_from_slice(bytemuck::cast_slice(&download));

        Ok(download_size)
    }

    /// Pulls all the elements from the [`GpuBuffer`] into a [`Vec`].
    pub async fn read_vec(&self) -> BufferResult<Vec<T>> {
        // Safety: Since T is Pod: Zeroed + ... it is safe to use zeroed() to init it.
        let mut buf = vec![unsafe { std::mem::zeroed() }; self.capacity()];
        self.read(&mut buf).await?;

        Ok(buf)
    }

    /// Blocking version of `GpuBuffer::read()`.
    pub fn read_blocking(&self, buf: &mut [T]) -> BufferResult<usize> {
        futures::executor::block_on(self.read(buf))
    }

    /// Blocking version of `GpuBuffer::read_vec()`.
    pub fn read_vec_blocking(&self) -> BufferResult<Vec<T>> {
        futures::executor::block_on(self.read_vec())
    }

    /// Write a buffer into this [`GpuBuffer`], returning how many elements were written. The operation is instantly offloaded.
    ///
    /// This function will attempt to write the entire contents of `buf` unless its capacity
    /// exceeds the one of the source buffer, in which case `GpuBuffer::capacity()` elements are written.
    pub fn write(&mut self, buf: &[T]) -> BufferResult<usize> {
        let input_size = buf.len() * std::mem::size_of::<T>();
        let upload_size = if input_size > self.size {
            self.size
        } else {
            input_size
        };

        self.fw
            .queue
            .write_buffer(&self.buf, 0, bytemuck::cast_slice(buf));

        let encoder = self
            .fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GpuBuffer::write"),
            });
        self.fw.queue.submit(Some(encoder.finish()));

        Ok(upload_size)
    }
}

impl<'fw, T> BufOps<'fw, T> for GpuUniformBuffer<'fw, T>
where
    T: bytemuck::Pod,
{
    fn size(&self) -> usize {
        self.size
    }

    fn as_gpu_buffer(&self) -> &wgpu::Buffer {
        &self.buf
    }

    fn with_capacity(fw: &'fw crate::Framework, capacity: usize) -> Self {
        let size = capacity * std::mem::size_of::<T>();

        let buf = fw.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuUniformBuffer::with_capacity"),
            size: size as u64,
            usage: GPU_UNIFORM_USAGES,
            mapped_at_creation: false,
        });

        Self {
            fw,
            buf,
            size,
            marker: PhantomData,
        }
    }

    fn from_slice(fw: &'fw crate::Framework, slice: &[T]) -> Self {
        let size = slice.len() * std::mem::size_of::<T>();
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
            size,
            marker: PhantomData,
        }
    }

    fn from_gpu_parts(fw: &'fw crate::Framework, buf: wgpu::Buffer, size: usize) -> Self {
        Self {
            fw,
            buf,
            size,
            marker: PhantomData,
        }
    }

    fn into_gpu_parts(self) -> (wgpu::Buffer, usize) {
        (self.buf, self.size)
    }
}

impl<'fw, T> GpuUniformBuffer<'fw, T>
where
    T: bytemuck::Pod,
{
    /// Write a buffer into this [`GpuUniformBuffer`], returning how many elements were written. The operation is instantly offloaded.
    ///
    /// This function will attempt to write the entire contents of `buf` unless its capacity
    /// exceeds the one of the source buffer, in which case `GpuBuffer::capacity()` elements are written.
    pub fn write(&mut self, buf: &[T]) -> BufferResult<usize> {
        let input_size = buf.len() * std::mem::size_of::<T>();
        let upload_size = if input_size > self.size {
            self.size
        } else {
            input_size
        };

        self.fw
            .queue
            .write_buffer(&self.buf, 0, bytemuck::cast_slice(buf));

        let encoder = self
            .fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GpuUniformBuffer::write"),
            });
        self.fw.queue.submit(Some(encoder.finish()));

        Ok(upload_size)
    }
}
