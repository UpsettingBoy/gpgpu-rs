//! Provides the [`SetBindings`] struct used to bind the data to the gpu
use crate::{entry_type::EntryType, primitives::*, *};

/// Binds data to bindings that will be sent to the gpu
///
/// # Example
/// ```
/// # use gpgpu::*;
/// let fw = Framework::default();
///
/// let data = (0..10000).into_iter().collect::<Vec<u32>>();
/// let scalar = 10u32;
///
/// // Create the buffer
/// let buffer = GpuBuffer::from_slice(&fw, &data);
///
/// // Create the uniform
/// let uniform = GpuUniformBuffer::from_slice(&fw, &[scalar]);
///
/// let binds = SetBindings::default()
///     .add_buffer(0, &buffer) // Binds data to the gpu with id 0
///     .add_uniform_buffer(1, &uniform); // Bind a uniform with id 1
/// ```
#[derive(Clone, Default)]
pub struct SetBindings<'res> {
    pub(crate) bindings: Vec<wgpu::BindGroupEntry<'res>>,
    pub(crate) entry_type: Vec<EntryType>,
}

impl<'res> SetBindings<'res> {
    /// Binds a new [`GpuBuffer`] with a bind id
    /// # Example
    /// ```
    /// # let fw = Framework::default();
    /// let data = (0..10000).into_iter().collect::<Vec<u32>>();
    /// let buffer = GpuBuffer::from_slice(&fw, &data);
    ///
    /// SetBindings::default().add_buffer(0, &buffer);
    /// ```
    pub fn add_buffer<T>(mut self, bind_id: u32, buffer: &'res GpuBuffer<T>) -> Self
    where
        T: bytemuck::Pod,
    {
        let bind = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: buffer.as_binding_resource(),
        };

        self.bindings.push(bind);
        self.entry_type.push(EntryType::Buffer);

        self
    }

    /// Binds a new [`GpuUniformBuffer`] with a bind id
    /// # Example
    /// ```
    /// # let fw = Framework::default();
    /// let scalar = 10u32;
    /// let uniform = GpuUniformBuffer::from_slice(&fw, &[scalar]);
    ///
    /// SetBindings::default().add_uniform_buffer(0, &uniform);
    /// ```
    pub fn add_uniform_buffer<T>(mut self, bind_id: u32, buffer: &'res GpuUniformBuffer<T>) -> Self
    where
        T: bytemuck::Pod,
    {
        let bind = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: buffer.as_binding_resource(),
        };

        self.bindings.push(bind);
        self.entry_type.push(EntryType::Uniform);

        self
    }

    /// Binds a new [`GpuImage`] with a bind id
    /// # Example
    /// ```
    /// # let fw = Framework::default();
    /// let data = [0, 0, 0];
    /// let image = GpuImage::from_bytes(&fw, &data, 1, 1);
    ///
    /// SetBindings::default().add_image(0, &image);
    /// ```
    pub fn add_image<P: PixelInfo>(mut self, bind_id: u32, img: &'res GpuImage<P>) -> Self {
        let bind = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: img.as_binding_resource(),
        };

        self.bindings.push(bind);
        self.entry_type.push(EntryType::Image);

        self
    }

    /// Binds a new [`GpuConstImage`] with a bind id
    /// # Example
    /// ```
    /// # let fw = Framework::default();
    /// let data = [0, 0, 0];
    /// let image = GpuConstImage::from_bytes(&fw, &data, 1, 1);
    ///
    /// SetBindings::default().add_const_image(0, &image);
    /// ```
    pub fn add_const_image<P: PixelInfo>(
        mut self,
        bind_id: u32,
        img: &'res GpuConstImage<P>,
    ) -> Self {
        let bind = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: img.as_binding_resource(),
        };

        self.bindings.push(bind);
        self.entry_type.push(EntryType::ConstImage);

        self
    }

    pub(crate) fn to_bind_group(
        &self,
        fw: &Framework,
        layout: &wgpu::BindGroupLayout,
        entry_types: &Vec<EntryType>,
    ) -> wgpu::BindGroup {
        // TODO: Make custom error struct/enum
        if self.entry_type.len() != entry_types.len() {
            panic!("SetBindings must have the same layout as SetLayout")
        }

        for entry_type in self.entry_type.iter().zip(entry_types.iter()) {
            if entry_type.0 != entry_type.1 {
                panic!("SetBindings do not have the same entry type as SetLayout")
            }
        }

        fw.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &self.bindings,
        })
    }
}
