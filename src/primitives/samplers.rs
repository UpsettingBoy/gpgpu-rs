use crate::Sampler;

impl Sampler {
    /// Creates a new [`Sampler`] using the given wrap and filter mode.
    pub fn new(
        fw: &crate::Framework,
        wrap_mode: crate::SamplerWrapMode,
        filter_mode: crate::SamplerFilterMode,
    ) -> Self {
        let address_mode = match wrap_mode {
            crate::SamplerWrapMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
            crate::SamplerWrapMode::Repeat => wgpu::AddressMode::Repeat,
            crate::SamplerWrapMode::MirrorRepeat => wgpu::AddressMode::MirrorRepeat,
            crate::SamplerWrapMode::ClampToBorder => wgpu::AddressMode::ClampToBorder,
        };
        let wgpu_filter_mode = match filter_mode {
            crate::SamplerFilterMode::Nearest => wgpu::FilterMode::Nearest,
            crate::SamplerFilterMode::Linear => wgpu::FilterMode::Linear,
        };
        let sampler = fw.device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: address_mode,
            address_mode_v: address_mode,
            address_mode_w: address_mode,
            mag_filter: wgpu_filter_mode,
            min_filter: wgpu_filter_mode,
            mipmap_filter: wgpu_filter_mode,
            lod_min_clamp: 0.0,
            lod_max_clamp: std::f32::MAX,
            compare: None,
            anisotropy_clamp: None,
            border_color: None,
        });
        Self {
            sampler,
            filter_mode,
        }
    }
}
