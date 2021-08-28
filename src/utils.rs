pub mod shader {
    use std::path::Path;

    use wgpu::ShaderModule;

    use crate::{Framework, GpuResult};

    pub fn from_spirv_file(fw: &Framework, path: impl AsRef<Path>) -> GpuResult<ShaderModule> {
        let bytes = std::fs::read(&path)?;
        let shader_name = path.as_ref().to_str();

        Ok(from_spirv_bytes(fw, &bytes, shader_name))
    }

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
}
