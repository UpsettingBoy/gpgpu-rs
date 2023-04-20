use crate::primitives::*;

#[derive(Clone, Default)]
pub struct SetBindings<'res> {
    pub(crate) bindings: Vec<wgpu::BindGroupEntry<'res>>,
}

impl<'res> SetBindings<'res> {
    pub fn new(bindings: Vec<(u32, &'res dyn AsBindingResource)>) -> Self {
        let mut out = Self::default();

        for (bind_id, entry) in bindings {
            let bind = wgpu::BindGroupEntry {
                binding: bind_id,
                resource: entry.as_binding_resource(),
            };

            out.bindings.push(bind)
        }

        out
    }
}
