use std::{borrow::Cow, path::Path};

use crate::{
    primitives::{BufOps, PixelInfo},
    DescriptorSet, Framework, GpuBuffer, GpuBufferUsage, GpuConstImage, GpuImage, GpuUniformBuffer,
    Kernel, Program, Shader,
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
            resource: uniform_buf.as_binding_resource(),
        };

        self.set_layout.push(bind_entry);
        self.binds.push(bind);

        self
    }

    /// Binds a [`GpuBuffer`] as a storage buffer in the shader with a specific `usage`.
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
    pub fn bind_buffer<T>(mut self, storage_buf: &'res GpuBuffer<T>, usage: GpuBufferUsage) -> Self
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
                    read_only: usage == GpuBufferUsage::ReadOnly,
                },
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

impl Shader {
    /// Initialises a [`Shader`] from a SPIR-V file.
    pub fn from_spirv_file(fw: &Framework, path: impl AsRef<Path>) -> std::io::Result<Self> {
        let bytes = std::fs::read(&path)?;
        let shader_name = path.as_ref().to_str();

        Ok(Self::from_spirv_bytes(fw, &bytes, shader_name))
    }

    /// Initialises a [`Shader`] from SPIR-V bytes with an optional `name`.
    pub fn from_spirv_bytes(fw: &Framework, bytes: &[u8], name: Option<&str>) -> Self {
        let source = wgpu::util::make_spirv(bytes);

        let shader = fw
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: name,
                source,
            });

        Self(shader)
    }

    /// Initialises a [`Shader`] from a `WGSL` file.
    pub fn from_wgsl_file(fw: &Framework, path: impl AsRef<Path>) -> std::io::Result<Self> {
        let source_string = std::fs::read_to_string(&path)?;
        let shader_name = path.as_ref().to_str();

        Ok(Self(fw.device.create_shader_module(
            &wgpu::ShaderModuleDescriptor {
                label: shader_name,
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(source_string)),
            },
        )))
    }
}

impl<'sha, 'res> Program<'sha, 'res> {
    /// Creates a new [`Program`] using a `shader` and an `entry_point`.
    pub fn new(shader: &'sha Shader, entry_point: impl Into<String>) -> Self {
        Self {
            shader,
            entry_point: entry_point.into(),
            descriptors: Vec::new(),
        }
    }

    /// Adds a [`DescriptorSet`] to this [`Program`] layout.
    pub fn add_descriptor_set(mut self, desc: DescriptorSet<'res>) -> Self {
        self.descriptors.push(desc);
        self
    }
}

impl<'fw> Kernel<'fw> {
    /// Creates a [`Kernel`] from a [`Program`].
    pub fn new<'sha, 'res>(fw: &'fw Framework, program: Program<'sha, 'res>) -> Self {
        let mut layouts = Vec::new();
        let mut sets = Vec::new();

        // Unwraping of descriptors from program
        for desc in &program.descriptors {
            let set_layout = fw
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &desc.set_layout,
                });

            let set = fw.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &set_layout,
                entries: &desc.binds,
            });

            layouts.push(set_layout);
            sets.push(set);
        }

        // Compute pipeline bindings
        let group_layouts = layouts.iter().collect::<Vec<_>>();

        let pipeline_layout = fw
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &group_layouts,
                push_constant_ranges: &[],
            });

        let pipeline = fw
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                module: &program.shader.0,
                entry_point: &program.entry_point,
                layout: Some(&pipeline_layout),
            });

        Self {
            fw,
            pipeline,
            sets,
            entry_point: program.entry_point,
        }
    }

    /// Enqueues the execution of this [`Kernel`] onto the GPU.
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
