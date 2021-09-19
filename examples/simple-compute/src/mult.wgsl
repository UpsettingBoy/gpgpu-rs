[[block]]
struct Buffer {
    data: [[stride(4)]] array<u32>; // Stride tells the byte size of each element. u32 = unsigned integer 32-bits = 4 bytes per element.
};

[[group(0), binding(0)]] var<storage, read> a: Buffer;           // Vector A - Input
[[group(0), binding(1)]] var<storage, read> b: Buffer;           // Vector B - Input
[[group(0), binding(2)]] var<storage, read_write> c: Buffer;     // Vector C - Output

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let idx = global_id.x;

    c.data[idx] = a.data[idx] * b.data[idx];
}