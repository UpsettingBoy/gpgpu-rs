use std::{sync::Arc, time::Duration};

use crate::Framework;

impl Default for Framework {
    fn default() -> Self {
        let backend = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY);
        let power_preference = wgpu::util::power_preference_from_env()
            .unwrap_or(wgpu::PowerPreference::HighPerformance);
        let instance = wgpu::Instance::new(backend);

        log::debug!(
            "Requesting device with {:#?} and {:#?}",
            backend,
            power_preference
        );

        futures::executor::block_on(async {
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference,
                    ..Default::default()
                })
                .await
                .expect("Failed at adapter creation.");

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
                    features: adapter.features(), // Change this to allow proper WebGL2 support (in the future™️).
                    limits: adapter.limits(),     // Bye WebGL2 support :(
                },
                None,
            )
            .await
            .expect("Failed at device creation.");

        let info = adapter.get_info();
        log::info!(
            "Using {} ({}) - {:#?}.",
            info.name,
            info.device,
            info.backend
        );

        let device = Arc::new(device);
        let polling_device = Arc::clone(&device);

        std::thread::spawn(move || loop {
            polling_device.poll(wgpu::Maintain::Poll);
            std::thread::sleep(polling_time);
        });

        Self {
            device,
            queue,
            adapter,
        }
    }

    /// Gets info about the adapter that created this [`Framework`].
    pub fn info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }

    /// Gets the features that may be used with this [`Framework`].
    pub fn features(&self) -> wgpu::Features {
        self.device.features()
    }

    /// Gets the limits of this [`Framework`].
    pub fn limits(&self) -> wgpu::Limits {
        self.device.limits()
    }

    /// Polls the device in a non-blocking manner.
    pub fn poll(&self) {
        self.device.poll(wgpu::Maintain::Poll);
    }

    /// Polls the device in a blocking manner.
    pub fn poll_blocking(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }
}
