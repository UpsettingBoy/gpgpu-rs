/// Contains some helper functions to create [`wgpu::ShaderModule`](wgpu::ShaderModule) from SPIR-V files or bytes.
pub mod shader {
    use std::{borrow::Cow, path::Path};

    use wgpu::ShaderModule;

    use crate::{Framework, GpuResult};

    /// Creates a [`wgpu::ShaderModule`](wgpu::ShaderModule) instance from a SPIR-V file.
    pub fn from_spirv_file(fw: &Framework, path: impl AsRef<Path>) -> GpuResult<ShaderModule> {
        let bytes = std::fs::read(&path)?;
        let shader_name = path.as_ref().to_str();

        Ok(from_spirv_bytes(fw, &bytes, shader_name))
    }

    /// Creates a [`wgpu::ShaderModule`](wgpu::ShaderModule) instance from SPIR-V bytes
    /// and an optional `shader_name`.
    pub fn from_spirv_bytes(
        fw: &Framework,
        bytes: &[u8],
        shader_name: Option<&str>,
    ) -> ShaderModule {
        let source = wgpu::util::make_spirv(bytes);

        fw.device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: shader_name,
                source,
            })
    }

    /// Creates a [`wgpu::ShaderModule`](wgpu::ShaderModule) instance from a `WGSL` file.
    /// Intended use is `examples` crates.
    pub fn from_wgsl_file(fw: &Framework, path: impl AsRef<Path>) -> GpuResult<ShaderModule> {
        let source_string = std::fs::read_to_string(path)?;

        Ok(fw
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(source_string)),
            }))
    }
}
