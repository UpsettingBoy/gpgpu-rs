use crate::*;

/// Used to enqueue the execution of a shader with the bidings provided.
///
/// Equivalent to OpenCL's Kernel.
pub struct Kernel<'fw> {
    fw: &'fw Framework,
    pipeline: wgpu::ComputePipeline,
    sets: Vec<wgpu::BindGroup>,
    entry_point: String,
}

impl<'fw> Kernel<'fw> {
    /// Creates a [`Kernel`] from a [`Program`].
    pub fn new<'sha, 'res>(fw: &'fw Framework, program: Program<'sha, 'res>) -> Self {
        let mut layouts = Vec::new();
        let mut sets = Vec::new();

        // Unwraping of descriptors from program
        for (set_id, desc) in program.descriptors.iter().enumerate() {
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

            log::debug!("Binding set = {} with {:#?}", set_id, &desc.binds);

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
            cpass.dispatch_workgroups(x, y, z);
        }

        self.fw.queue.submit(Some(encoder.finish()));
    }
}
