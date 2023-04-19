use crate::{DescriptorSet, Shader};

/// Represents an entry point with its bindings on a [`Shader`].
pub struct Program<'sha, 'res> {
    pub(crate) shader: &'sha Shader,
    pub(crate) entry_point: String,
    pub(crate) descriptors: Vec<DescriptorSet<'res>>,
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
