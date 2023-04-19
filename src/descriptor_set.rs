use crate::{primitives::*, *};

/// Contains a binding group of resources.
#[derive(Default, Clone)]
pub struct DescriptorSet<'res> {
    pub(crate) set_layout: Vec<wgpu::BindGroupLayoutEntry>,
    pub(crate) binds: Vec<wgpu::BindGroupEntry<'res>>,
}

impl<'res> DescriptorSet<'res> {
    /// Binds a [`GpuUniformBuffer`] as a uniform buffer in the shader.
    ///
    /// ### Example WGSL syntax:
    /// ```ignore
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
    pub fn bind_uniform_buffer<T>(
        mut self,
        uniform_buf: &'res GpuUniformBuffer<T>,
        bind_id: u32,
    ) -> Self
    where
        T: bytemuck::Pod,
    {
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
    pub fn bind_buffer<T>(
        mut self,
        storage_buf: &'res GpuBuffer<T>,
        usage: GpuBufferUsage,
        bind_id: u32,
    ) -> Self
    where
        T: bytemuck::Pod,
    {
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
    pub fn bind_image<P: PixelInfo>(mut self, img: &'res GpuImage<P>, bind_id: u32) -> Self {
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
            resource: img.as_binding_resource(),
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
    pub fn bind_const_image<P>(mut self, img: &'res GpuConstImage<P>, bind_id: u32) -> Self
    where
        P: PixelInfo,
    {
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
            resource: img.as_binding_resource(),
        };

        self.set_layout.push(bind_entry);
        self.binds.push(bind);

        self
    }
}
