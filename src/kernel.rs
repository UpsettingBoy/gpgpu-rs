use crate::{
    primitives::PixelInfo, DescriptorSet, GpuBuffer, GpuBufferUsage, GpuConstImage, GpuImage,
    GpuUniformBuffer, Kernel, KernelBuilder,
};

impl<'res> DescriptorSet<'res> {
    /// Binds a [`GpuUniformBuffer`] as a uniform buffer in the shader.
    ///
    /// ### Example WGSL syntax:
    /// ```ignore
    /// [[block]]
    /// struct UniformStruct {
    ///     a: vec3<u32>;
    ///     b: vec3<u32>;
    ///     c: vec3<u32>;
    /// };
    ///
    /// [[group(0), binding(0)]]
    /// var<uniform> myUniformBuffer: UniformStruct;
    /// ```
    ///
    /// ### Example GLSL syntax:
    /// ```glsl
    /// layout(std140, binding = 0)
    /// uniform UniformStruct {
    ///     uvec3 a;
    ///     uvec3 b;
    ///     uvec3 c;
    /// };
    /// ```
    pub fn bind_uniform_buffer<T>(mut self, uniform_buf: &'res GpuUniformBuffer<T>) -> Self
    where
        T: bytemuck::Pod,
    {
        let bind_id = self.set_layout.len() as u32;

        let bind_entry = wgpu::BindGroupLayoutEntry {
            binding: bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                has_dynamic_offset: false,
                min_binding_size: None,
                ty: wgpu::BufferBindingType::Uniform,
            },
            count: None,
        };

        let bind = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: uniform_buf.0.get_binding_resource(),
        };

        self.set_layout.push(bind_entry);
        self.binds.push(bind);

        self
    }

    /// Binds a [`GpuBuffer`] as a storage buffer in the shader.
    ///
    /// The `access` mode must be either [`AccessMode::ReadOnly`] or [`AccessMode::ReadWrite`].
    ///
    /// ### Example WGSL syntax:
    /// ```ignore
    /// [[block]]
    /// struct StorageStruct {
    ///     data: [[stride(4)]] array<i32>;
    /// };
    ///
    /// [[group(0), binding(0)]]
    /// var<storage, read_write> myStorageBuffer: StorageStruct;
    /// ```
    ///
    /// ### Example GLSL syntax:
    /// ```glsl
    /// layout (set=0, binding=0) buffer myStorageBuffer {
    ///     int data[];
    /// };
    /// ```
    pub fn bind_buffer<T>(mut self, storage_buf: &'res GpuBuffer<T>, access: GpuBufferUsage) -> Self
    where
        T: bytemuck::Pod,
    {
        let bind_id = self.set_layout.len() as u32;

        let bind_entry = wgpu::BindGroupLayoutEntry {
            binding: bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                has_dynamic_offset: false,
                min_binding_size: None,
                ty: wgpu::BufferBindingType::Storage {
                    read_only: access == GpuBufferUsage::ReadOnly,
                },
            },
            count: None,
        };

        let bind = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: storage_buf.0.get_binding_resource(),
        };

        self.set_layout.push(bind_entry);
        self.binds.push(bind);

        self
    }

    /// Binds a [`GpuImage`] as a storage image in the shader.
    /// This image is write-only.
    /// ### Example WGSL syntax:
    /// ```ignore
    /// [[group(0), binding(0)]]
    /// var myStorageImg: texture_storage_2d<rgba8uint, write>;
    /// ```
    ///
    /// ### Example GLSL syntax:
    /// ```glsl
    /// layout (set=0, binding=0, rgba8uint) uimage2D myStorageImg;
    /// ```
    pub fn bind_image<P: PixelInfo>(mut self, img: &'res GpuImage<P>) -> Self {
        let bind_id = self.set_layout.len() as u32;

        let bind_entry = wgpu::BindGroupLayoutEntry {
            binding: bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: P::wgpu_format(),
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };

        let bind = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: img.0.get_binding_resource(),
        };

        self.set_layout.push(bind_entry);
        self.binds.push(bind);

        self
    }

    /// Binds a [`GpuConstImage`] as a texture in the shader.
    /// This image is read-only.
    /// ### Example WGSL syntax:
    /// ```ignore
    /// [[group(0), binding(0)]]
    /// var myTexture: texture_2d<u32>;
    /// ```
    ///
    /// ### Example GLSL syntax:
    /// ```glsl
    /// layout (set=0, binding=0) utexture2D myTexture;
    /// ```
    pub fn bind_const_image<P>(mut self, img: &'res GpuConstImage<P>) -> Self
    where
        P: PixelInfo,
    {
        let bind_id = self.set_layout.len() as u32;

        let bind_entry = wgpu::BindGroupLayoutEntry {
            binding: bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: P::wgpu_texture_sample(),
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };

        let bind = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: img.0.get_binding_resource(),
        };

        self.set_layout.push(bind_entry);
        self.binds.push(bind);

        self
    }
}

impl<'fw, 'res, 'sha> KernelBuilder<'fw, 'res, 'sha> {
    /// Adds a [`DescriptorSet`] into the [`KernelBuilder`] internal layout.
    pub fn add_descriptor_set(mut self, desc: DescriptorSet<'res>) -> Self {
        let set_layout =
            self.fw
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &desc.set_layout,
                });

        let set = self
            .fw
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &set_layout,
                entries: &desc.binds,
            });

        self.layouts.push(set_layout);
        self.descriptors.push(desc);
        self.sets.push(set);

        self
    }

    /// Builds a [`Kernel`] from the layout stored in [`KernelBuilder`].
    pub fn build(self) -> Kernel<'fw> {
        let sets = self.layouts.iter().collect::<Vec<_>>();

        let pipeline_layout =
            self.fw
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &sets,
                    push_constant_ranges: &[],
                });

        let pipeline = self
            .fw
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                module: self.shader,
                entry_point: &self.entry_point,
                layout: Some(&pipeline_layout),
            });

        Kernel {
            fw: self.fw,
            pipeline,
            sets: self.sets,
            entry_point: self.entry_point,
        }
    }
}

impl<'fw> Kernel<'fw> {
    /// Enqueues the execution of this [`Kernel`] to the GPU.
    ///
    /// [`Kernel`] will dispatch `x`, `y` and `z` workgroups per dimension.
    pub fn enqueue(&self, x: u32, y: u32, z: u32) {
        let mut encoder = self
            .fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Kernel::enqueue"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Kernel::enqueue"),
            });

            cpass.set_pipeline(&self.pipeline);

            for (id_set, set) in self.sets.iter().enumerate() {
                cpass.set_bind_group(id_set as u32, set, &[]);
            }

            cpass.insert_debug_marker(&self.entry_point);
            cpass.dispatch(x, y, z);
        }

        self.fw.queue.submit(Some(encoder.finish()));
    }
}
