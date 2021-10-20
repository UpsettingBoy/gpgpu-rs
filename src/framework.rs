use std::marker::PhantomData;

use wgpu::util::DeviceExt;

use crate::{primitives::PixelInfo, Framework, GpuBuffer, GpuImage, KernelBuilder};

impl Default for Framework {
    fn default() -> Self {
        let backend = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY);

        let instance = wgpu::Instance::new(backend);
        let (device, queue) = futures::executor::block_on(async {
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                })
                .await
                .unwrap();

            adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: None,
                        features: wgpu::Features::empty(),
                        limits: wgpu::Limits::downlevel_webgl2_defaults(),
                    },
                    None,
                )
                .await
                .unwrap()
        });

        Self {
            instance,
            device,
            queue,
        }
    }
}

impl Framework {
    /// Creates a new [`Framework`] instance from `wgpu` objects.
    ///
    /// This is mainly used when you want to use wgpu alongside of `gpgpu_rs`
    /// or you need special requiremients (device, features, ...).
    ///
    /// If you only want a [`Framework`] using a HighPerformance GPU with
    /// the minimal features to run on the web (WebGPU), use [`Framework::default()`](Framework::default).
    pub fn new(instance: wgpu::Instance, device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            instance,
            device,
            queue,
        }
    }

    /// Creates a [`KernelBuilder`] from a `wgpu` [`ShaderModule`](wgpu::ShaderModule)
    ///
    /// A `ShaderModule` can be created using the `utils::shader` methods,
    /// [`utils::shader::from_spirv_file()`](crate::utils::shader::from_spirv_file) and
    /// [`utils::shader::from_spirv_bytes()`](crate::utils::shader::from_spirv_bytes); or
    /// using `wgpu`.
    pub fn create_kernel_builder<'sha>(
        &self,
        shader: &'sha wgpu::ShaderModule,
        entry_point: impl Into<String>,
    ) -> KernelBuilder<'_, '_, 'sha> {
        KernelBuilder {
            fw: self,
            layouts: Vec::new(),
            descriptors: Vec::new(),
            sets: Vec::new(),
            shader,
            entry_point: entry_point.into(),
        }
    }

    /// Creates an empty [`GpuBuffer`] of the desired `len`gth.
    pub fn create_buffer<T>(&self, len: usize) -> GpuBuffer<T>
    where
        T: bytemuck::Pod,
    {
        let size = len * std::mem::size_of::<T>();

        let storage = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        GpuBuffer {
            fw: self,
            storage,
            size,
            _marker: PhantomData,
        }
    }

    /// Creates a [`GpuBuffer`] from a `data` slice.
    pub fn create_buffer_from_slice<T>(&self, data: &[T]) -> GpuBuffer<T>
    where
        T: bytemuck::Pod,
    {
        let size = data.len() * std::mem::size_of::<T>();

        let storage = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        GpuBuffer {
            fw: self,
            storage,
            size,
            _marker: PhantomData,
        }
    }

    // TODO: Reuse staging buffers from pool instead of creating-destroying for every read.
    // Could be even selected typing it in Framework as Framework<Cache = Recreate> or Framework<Cache = Pool>, etc
    pub(crate) fn create_staging_buffer(&self, size: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ
                | wgpu::BufferUsages::MAP_WRITE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Creates an empty [`GpuImage`] with the desired `width`, `height` and [`TextureFormat`](wgpu::TextureFormat).
    pub fn create_image<P>(&self, width: u32, height: u32) -> GpuImage<P>
    where
        P: PixelInfo,
    {
        GpuImage::new(self, width, height)
    }

    /// Non-blocking GPU poll.
    pub fn poll(&self) {
        self.device.poll(wgpu::Maintain::Poll);
    }

    /// Blocking GPU poll.
    pub fn blocking_poll(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }
}
