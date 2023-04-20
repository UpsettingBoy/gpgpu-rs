#[derive(Clone, Default)]
pub struct SetBindings<'res> {
    pub(crate) bindings: Vec<wgpu::BindGroupEntry<'res>>,
}
