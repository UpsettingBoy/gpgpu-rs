use crate::*;

/// Used to enqueue the execution of a shader with the bidings provided.
///
/// Equivalent to OpenCL's Kernel.
pub struct Kernel<'fw> {
    fw: &'fw Framework,
    pipeline: wgpu::ComputePipeline,
    // sets: Vec<wgpu::BindGroup>,
    // entry_point: String,
}

impl<'fw> Kernel<'fw> {
    /// Creates a [`Kernel`] from a [`Program`].
    pub fn new<'sha, 'res>(
        fw: &'fw Framework,
        shader: &'sha Shader,
        entry_point: impl Into<String>,
        layouts: Vec<SetLayout>,
    ) -> Self {
        // let set = fw.device.create_bind_group(&wgpu::BindGroupDescriptor {
        //     label: None,
        //     layout: &set_layout,
        //     entries: &desc.binds,
        // });
        //
        // log::debug!("Binding set = {} with {:#?}", set_id, &desc.binds);

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
                entry_point: &entry_point.into(),
                layout: Some(&pipeline_layout),
            });

        Self { fw, pipeline }
    }

    /// Enqueues the execution of this [`Kernel`] onto the GPU.
    ///
    /// [`Kernel`] will dispatch `x`, `y` and `z` workgroups per dimension.
    pub fn enqueue(&self, x: u32, y: u32, z: u32) {
        todo!();
        // let mut encoder = self
        //     .fw
        //     .device
        //     .create_command_encoder(&wgpu::CommandEncoderDescriptor {
        //         label: Some("Kernel::enqueue"),
        //     });
        // {
        //     let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        //         label: Some("Kernel::enqueue"),
        //     });
        //
        //     cpass.set_pipeline(&self.pipeline);
        //
        //     for (id_set, set) in self.sets.iter().enumerate() {
        //         cpass.set_bind_group(id_set as u32, set, &[]);
        //     }
        //
        //     cpass.insert_debug_marker(&self.entry_point);
        //     cpass.dispatch_workgroups(x, y, z);
        // }
        //
        // self.fw.queue.submit(Some(encoder.finish()));
    }
}
