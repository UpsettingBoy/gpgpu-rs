use gpgpu::{
    primitives::pixels::Rgba8UintNorm, DescriptorSet, Framework, GpuConstImage, GpuImage,
    GpuUniformBuffer,
};
use image::buffer::ConvertBuffer;
use minifb::{Key, Window, WindowOptions};
use nokhwa::{Camera, CameraFormat, Resolution};

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;

fn main() {
    let fw = Framework::default();

    let mut camera = {
        let camera_format = CameraFormat::new(
            Resolution {
                width_x: WIDTH as u32,
                height_y: HEIGHT as u32,
            },
            nokhwa::FrameFormat::MJPEG,
            30,
        );

        Camera::new(0, Some(camera_format)).unwrap()
    };

    let mut window = Window::new(
        "gpgpu webcam example",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap();

    camera.open_stream().unwrap();
    window.limit_update_rate(Some(std::time::Duration::from_secs_f32(1.0 / 60.0)));

    let mut gpu_input = GpuConstImage::<Rgba8UintNorm>::new(&fw, WIDTH as u32, HEIGHT as u32);
    let mut buf_time = GpuUniformBuffer::<f32>::new(&fw, 1).unwrap();

    let gpu_output = GpuImage::<Rgba8UintNorm>::new(&fw, WIDTH as u32, HEIGHT as u32);

    let desc = DescriptorSet::default()
        .bind_const_image(&gpu_input)
        .bind_image(&gpu_output)
        .bind_uniform_buffer(&buf_time);

    let shader = gpgpu::utils::shader::from_wgsl_file(&fw, "examples/webcam/shader.wgsl").unwrap();

    let kernel = fw
        .create_kernel_builder(&shader, "main")
        .add_descriptor_set(desc)
        .build();

    let time = std::time::Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let cam_buf = camera.frame().unwrap();
        gpu_input.write_from_image_buffer(&cam_buf.convert());
        buf_time.write(&[time.elapsed().as_secs_f32()]);

        kernel.enqueue(WIDTH as u32 / 32, HEIGHT as u32 / 32, 1);

        let output_buf = gpu_output.read().unwrap();
        let output_buf = bytemuck::cast_slice(&output_buf);

        window
            .update_with_buffer(output_buf, WIDTH, HEIGHT)
            .unwrap();
    }
}
