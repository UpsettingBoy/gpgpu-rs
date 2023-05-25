@group(0) @binding(0) var input: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var linearSampler: sampler;
@group(0) @binding(3) var nearestNeighborSampler: sampler;

@compute @workgroup_size(32, 32, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coord = vec2<i32>(global_id.xy);
    let dims = textureDimensions(output);
    let uv = vec2<f32>(coord) / vec2<f32>(dims);

    // Must supply a mip level, since compute shaders don't have access to pixel derivatives.
    var pixel = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if uv.x < 0.5 {
        pixel = textureSampleLevel(input, linearSampler, uv, 0.0);
    } else {
        pixel = textureSampleLevel(input, nearestNeighborSampler, uv, 0.0);
    }

    textureStore(output, coord, pixel);
}
