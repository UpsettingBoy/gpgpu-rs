use std::sync::Arc;

use crate::{DescriptorSet, Framework, Kernel};

impl<'res> Kernel<'res> {
    pub fn add_set(
        mut self,
        fw: Arc<Framework>,
        resources: Vec<wgpu::BindingResource<'res>>,
    ) -> Self {
        let id_layout = self.descriptors.len();
        let layout = self.pipeline.get_bind_group_layout(id_layout as u32);

        let bindings = resources
            .into_iter()
            .enumerate()
            .map(|(id_res, res)| wgpu::BindGroupEntry {
                binding: id_res as u32,
                resource: res,
            })
            .collect::<Vec<_>>();

        let set = fw.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&self.name),
            layout: &layout,
            entries: &bindings,
        });

        let descriptor_set = DescriptorSet {
            set,
            layout,
            bindings,
        };

        self.descriptors.push(descriptor_set);
        self
    }

    pub fn enqueue(&self, fw: Arc<Framework>, dims: (u32, u32, u32)) {
        let mut encoder = fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&self.pipeline);

            for (id_set, set) in self.descriptors.iter().enumerate() {
                cpass.set_bind_group(id_set as u32, &set.set, &[]);
            }

            cpass.insert_debug_marker(&self.name);

            let (x, y, z) = dims;
            cpass.dispatch(x, y, z);
        }

        fw.queue.submit(Some(encoder.finish()));
    }

    // pub fn enqueue_wait(&self, fw: Arc<Framework>, dims: (u32, u32, u32)) {
    //     self.enqueue(fw.clone(), dims);
    //     fw.device.poll(wgpu::Maintain::Wait);
    // }
}
