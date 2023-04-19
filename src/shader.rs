use crate::Framework;
use std::{borrow::Cow, path::Path};

/// Represents a shader.
///
/// It's just a wrapper around [`wgpu::ShaderModule`].
pub struct Shader(pub(crate) wgpu::ShaderModule);

impl Shader {
    /// Initialises a [`Shader`] from a SPIR-V file.
    pub fn from_spirv_file(fw: &Framework, path: impl AsRef<Path>) -> std::io::Result<Self> {
        let bytes = std::fs::read(&path)?;
        let shader_name = path.as_ref().to_str();

        Ok(Self::from_spirv_bytes(fw, &bytes, shader_name))
    }

    /// Initialises a [`Shader`] from SPIR-V bytes with an optional `name`.
    pub fn from_spirv_bytes(fw: &Framework, bytes: &[u8], name: Option<&str>) -> Self {
        let source = wgpu::util::make_spirv(bytes);

        let shader = fw
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: name,
                source,
            });

        Self(shader)
    }

    /// Initialises a [`Shader`] from a `WGSL` file.
    pub fn from_wgsl_file(fw: &Framework, path: impl AsRef<Path>) -> std::io::Result<Self> {
        let source_string = std::fs::read_to_string(&path)?;
        let shader_name = path.as_ref().to_str();

        Ok(Self(fw.device.create_shader_module(
            wgpu::ShaderModuleDescriptor {
                label: shader_name,
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(source_string)),
            },
        )))
    }

    /// Initialises a [`Shader`] from a `WGSL` string.
    pub fn from_wgsl_string(
        fw: &Framework,
        source: String,
        name: Option<&str>,
    ) -> std::io::Result<Self> {
        Ok(Self(fw.device.create_shader_module(
            wgpu::ShaderModuleDescriptor {
                label: name,
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
            },
        )))
    }
}
