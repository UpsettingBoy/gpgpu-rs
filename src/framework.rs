use crate::{Framework, KernelBuilder};

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
            limits: device.limits(),
            device,
            queue,
        }
    }
}

impl Framework {
    /// Creates a new [`Framework`] instance from `wgpu` objects.
    ///
    /// This is mainly used when you want to use `wgpu` and `gpgpu` alongside
    /// or you need special requiremients (device, features, ...).
    ///
    /// If you only want a [`Framework`] using a HighPerformance GPU with
    /// the minimal features to run on the web (WebGPU), use [`Framework::default`](crate::Framework::default).
    pub fn new(instance: wgpu::Instance, device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            instance,
            limits: device.limits(),
            device,
            queue,
        }
    }

    /// Creates a [`KernelBuilder`] from a `wgpu` [`wgpu::ShaderModule`] and
    /// its `entry_point`.
    ///
    /// A `ShaderModule` can be created using the `utils::shader` methods,
    /// [`utils::shader::from_spirv_file`](crate::utils::shader::from_spirv_file) and
    /// [`utils::shader::from_spirv_bytes`](crate::utils::shader::from_spirv_bytes); or
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

    pub(crate) fn create_download_staging_buffer(&self, size: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub(crate) fn create_upload_staging_buffer(&self, size: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
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
