use crate::{DescriptorSet, GpuBuffer, GpuImage, Kernel, KernelBuilder};

impl<'res> DescriptorSet<'res> {
    pub fn bind_uniform_buffer<T>(mut self, uniform_buf: &'res GpuBuffer<T>) -> Self
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
            resource: uniform_buf.as_binding_resource(),
        };

        self.set_layout.push(bind_entry);
        self.binds.push(bind);

        self
    }

    pub fn bind_storage_buffer<T>(
        mut self,
        storage_buf: &'res GpuBuffer<T>,
        read_only: bool,
    ) -> Self
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
                ty: wgpu::BufferBindingType::Storage { read_only },
            },
            count: None,
        };

        let bind = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: storage_buf.as_binding_resource(),
        };

        self.set_layout.push(bind_entry);
        self.binds.push(bind);

        self
    }

    pub fn bind_storage_image(
        mut self,
        img: &'res GpuImage,
        access: wgpu::StorageTextureAccess,
    ) -> Self {
        let bind_id = self.set_layout.len() as u32;

        let bind_entry = wgpu::BindGroupLayoutEntry {
            binding: bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access,
                format: img.format,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };

        let bind = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: img.as_binding_resource(),
        };

        self.set_layout.push(bind_entry);
        self.binds.push(bind);

        self
    }

    pub fn bind_image(mut self, img: &'res GpuImage, sample_type: wgpu::TextureSampleType) -> Self {
        let bind_id = self.set_layout.len() as u32;

        let bind_entry = wgpu::BindGroupLayoutEntry {
            binding: bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type,
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };

        let bind = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: img.as_binding_resource(),
        };

        self.set_layout.push(bind_entry);
        self.binds.push(bind);

        self
    }
}

impl<'fw, 'res, 'sha> KernelBuilder<'fw, 'res, 'sha> {
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

    // pub fn build(self) -> Kernel<'fw, 'res, 'sha> {
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
            // layouts: self.layouts,
            // pipeline_layout,
            pipeline,
            // descriptors: self.descriptors,
            sets: self.sets,
            // shader: self.shader,
            entry_point: self.entry_point,
        }
    }
}

// impl<'fw, 'res, 'sha> Kernel<'fw, 'res, 'sha> {
impl<'fw> Kernel<'fw> {
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
