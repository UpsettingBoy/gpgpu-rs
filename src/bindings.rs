use crate::primitives::*;

#[derive(Clone, Default)]
pub struct SetBindings<'res> {
    pub(crate) bindings: Vec<wgpu::BindGroupEntry<'res>>,
}

impl<'res> SetBindings<'res> {
    pub fn add_entry<T: AsBindingResource>(&mut self, bind_id: u32, bind: &'res T) {
        let bind = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: bind.as_binding_resource(),
        };

        self.bindings.push(bind)
    }
}
