use crate::{GpuBuffer, GpuResult};

pub struct GpuArray<'fw, T, D>
where
    T: bytemuck::Pod,
    D: ndarray::Dimension,
{
    buf: GpuBuffer<'fw, T>,
    dim: D,
}

impl<'fw, T, D> GpuArray<'fw, T, D>
where
    T: bytemuck::Pod,
    D: ndarray::Dimension,
{
    pub fn from_array(
        fw: &'fw crate::Framework,
        array: ndarray::ArrayView<T, D>,
    ) -> GpuResult<Self> {
        let slice: Result<&[T], Box<dyn std::error::Error>> = array
            .as_slice_memory_order()
            .ok_or("Array is not contiguous".into());

        let buf = GpuBuffer::from_slice(fw, slice?);

        Ok(Self {
            buf,
            dim: array.raw_dim(),
        })
    }

    pub fn read_to_array(&self) -> GpuResult<ndarray::Array<T, D>> {
        let v = self.buf.read()?;
        let array = ndarray::Array::from_shape_vec(self.dim.clone(), v).map_err(|_| "Shape error");
        Ok(array?)
    }

    pub fn write_to_array(&mut self, array: ndarray::ArrayView<T, D>) -> GpuResult<()> {
        let slice: Result<&[T], Box<dyn std::error::Error>> = array
            .as_slice_memory_order()
            .ok_or("Array is not contiguous".into());

        self.buf.write(slice?);

        Ok(())
    }
}
