use std::io::Write;

use gpgpu::{
    primitives::pixels::Rgba8UintNorm, BufOps, DescriptorSet, Framework, GpuConstImage, GpuImage,
    GpuUniformBuffer, ImgOps,
};

use minifb::{Key, Window, WindowOptions};
use nokhwa::{
    pixel_format::RgbAFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
    Camera,
};

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;

fn main() {
    let fw = Framework::default();

    // Camera initilization. Config may not work if not same cam as the Thinkpad T480 one.
    // Change parameters accordingly
    let mut camera = {
        let requested =
            RequestedFormat::new::<RgbAFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        Camera::new(CameraIndex::Index(0), requested).unwrap()
    };

    // Window initialization
    let mut window = Window::new(
        "gpgpu webcam example",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap();

    camera.open_stream().unwrap();
    window.limit_update_rate(Some(std::time::Duration::from_secs_f32(1.0 / 60.0)));

    // Since the same GPU resources could be used during the whole execution
    // of the program, they are outside of the event loop
    let gpu_input = GpuConstImage::<Rgba8UintNorm>::new(&fw, WIDTH as u32, HEIGHT as u32); // Cam frame texture
    let buf_time = GpuUniformBuffer::<f32>::with_capacity(&fw, 1); // Elapsed time buffer (single element) for fancy shaders 😁

    let gpu_output = GpuImage::<Rgba8UintNorm>::new(&fw, WIDTH as u32, HEIGHT as u32); // Shader output

    let shader = gpgpu::Shader::from_wgsl_file(&fw, "examples/webcam/shader.wgsl").unwrap();

    let desc = DescriptorSet::default()
        .bind_const_image(&gpu_input)
        .bind_image(&gpu_output)
        .bind_uniform_buffer(&buf_time);
    let program = gpgpu::Program::new(&shader, "main").add_descriptor_set(desc);

    let kernel = gpgpu::Kernel::new(&fw, program);

    let time = std::time::Instant::now();

    let mut frame_buffer = vec![0u32; WIDTH * HEIGHT * 4];

    let mut total = 0.0;
    let mut count = 0;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let fps = std::time::Instant::now();

        // Adapted for new (using image 0.24) nokhwa version
        let cam_raw = camera.frame().unwrap();
        let cam_buf = cam_raw.decode_image::<RgbAFormat>().unwrap(); // Obtain cam current frame

        gpu_input.write_image_buffer(&cam_buf).unwrap(); // Upload cam frame into the cam frame texture
        buf_time.write(&[time.elapsed().as_secs_f32()]).unwrap(); // Upload elapsed time into elapsed time buffer

        kernel.enqueue(WIDTH as u32 / 32, HEIGHT as u32 / 31, 1);

        gpu_output
            .read_blocking(bytemuck::cast_slice_mut(&mut frame_buffer))
            .unwrap();

        // Write processed cam frame into window frame buffer
        window
            .update_with_buffer(&frame_buffer, WIDTH, HEIGHT)
            .unwrap();

        print_fps(fps.elapsed().as_secs_f32(), &mut total, &mut count);
    }
}

fn print_fps(elapsed: f32, total: &mut f32, count: &mut u32) {
    let fps = 1.0 / elapsed;

    *total += fps;
    *count += 1;

    print!(
        "\rFPS: {:00.0}\tAverage: {:00.2}",
        fps,
        *total / *count as f32
    );

    std::io::stdout().flush().unwrap();
}
