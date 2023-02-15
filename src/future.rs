use std::{future::Future, sync::Arc, task::Poll};

use parking_lot::Mutex;

use crate::primitives::buffers::BufferError;

/// Asynchronously maps a [`wgpu::BufferSlice`] for reading or writing.
pub struct GpuMapFuture<'buf> {
    device: &'buf wgpu::Device,
    buf_slice: wgpu::BufferSlice<'buf>,
    map_mode: wgpu::MapMode,
    is_invoked: bool,
    output: Arc<Mutex<Option<Result<(), BufferError>>>>,
}

impl<'buf> GpuMapFuture<'buf> {
    pub fn new(
        device: &'buf wgpu::Device,
        buf_slice: wgpu::BufferSlice<'buf>,
        map_mode: wgpu::MapMode,
    ) -> Self {
        Self {
            device,
            buf_slice,
            map_mode,
            is_invoked: false,
            output: Arc::new(Mutex::new(None)),
        }
    }
}

impl<'buf> Future for GpuMapFuture<'buf> {
    type Output = Result<(), BufferError>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Self::Output> {
        match self.is_invoked {
            false => {
                let waker = cx.waker().clone();
                let callback_output = Arc::clone(&self.output);

                self.buf_slice
                    .map_async(self.map_mode, move |callback_result| {
                        *callback_output.lock() =
                            Some(callback_result.map_err(BufferError::AsyncMapError));
                        waker.wake();
                    });

                self.is_invoked = true;
                self.device.poll(wgpu::MaintainBase::Poll);

                Poll::Pending
            }
            true => match self.output.lock().take() {
                Some(value) => Poll::Ready(value),
                None => Poll::Pending,
            },
        }
    }
}
