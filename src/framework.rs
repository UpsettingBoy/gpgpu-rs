use std::{sync::Arc, time::Duration};

use crate::Framework;

impl Default for Framework {
    fn default() -> Self {
        let backend = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY);
        let power_preference = wgpu::util::power_preference_from_env()
            .unwrap_or(wgpu::PowerPreference::HighPerformance);
        let instance = wgpu::Instance::new(backend);

        futures::executor::block_on(async {
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference,
                    ..Default::default()
                })
                .await
                .unwrap();

            Self::new(adapter, Duration::from_millis(10)).await
        })
    }
}

impl Framework {
    /// Creates a new [`Framework`] instance from a [`wgpu::Adapter`] and a `polling_time`.
    ///
    /// Use this method when there are multiple GPUs in use or when a [`wgpu::Surface`] is required.
    pub async fn new(adapter: wgpu::Adapter, polling_time: Duration) -> Self {
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: adapter.limits(), // Bye WebGL2 support :(
                },
                None,
            )
            .await
            .unwrap();

        let device = Arc::new(device);
        let polling_device = Arc::clone(&device);

        std::thread::spawn(move || loop {
            polling_device.poll(wgpu::Maintain::Poll);
            std::thread::sleep(polling_time);
        });

        Self { device, queue }
    }
}
