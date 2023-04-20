use crate::{bindings::SetBindings, entry_type::EntryType, *};

/// Used to enqueue the execution of a shader with the bidings provided.
///
/// Equivalent to OpenCL's Kernel.
pub struct Kernel<'fw, 'a> {
    fw: &'fw Framework,
    pipeline: wgpu::ComputePipeline,
    entry_types: Vec<Vec<EntryType>>,
    layouts: Vec<wgpu::BindGroupLayout>,
    function_name: &'a str,
}

impl<'fw, 'a> Kernel<'fw, 'a> {
    /// Creates a [`Kernel`] from a [`Program`].
    pub fn new<'sha, 'res>(
        fw: &'fw Framework,
        shader: &'sha Shader,
        function_name: &'a str,
        layouts: Vec<SetLayout>,
    ) -> Self {
        let entry_types = layouts
            .iter()
            .map(|layout| layout.entry_type.clone())
            .collect();

        // Compute pipeline bindings
        let layouts = layouts
            .iter()
            .map(|layout| {
                fw.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &layout.layout_entry,
                    })
            })
            .collect::<Vec<_>>();

        let mut group_layouts = vec![];

        for layout in layouts.iter() {
            group_layouts.push(layout)
        }

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
                module: &shader.0,
                entry_point: function_name,
                layout: Some(&pipeline_layout),
            });

        Self {
            fw,
            pipeline,
            entry_types,
            layouts,
            function_name,
        }
    }

    /// executes this [`Kernel`] with the give bindings.
    ///
    /// [`Kernel`] will dispatch `x`, `y` and `z` workgroups per dimension.
    pub fn run(&self, bindings: Vec<SetBindings>, x: u32, y: u32, z: u32) {
        let mut encoder = self
            .fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Kernel::enqueue"),
            });

        let bind_groups = bindings
            .iter()
            .zip(self.layouts.iter())
            .zip(self.entry_types.iter())
            .map(|((binding, layout), entry_type)| {
                binding.into_bind_group(self.fw, layout, entry_type)
            })
            .collect::<Vec<_>>();

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Kernel::enqueue"),
        });

        cpass.set_pipeline(&self.pipeline);

        for (bind_id, binds) in bind_groups.iter().enumerate() {
            cpass.set_bind_group(bind_id as u32, &binds, &[])
        }

        cpass.insert_debug_marker(self.function_name);
        cpass.dispatch_workgroups(x, y, z)
    }
}
