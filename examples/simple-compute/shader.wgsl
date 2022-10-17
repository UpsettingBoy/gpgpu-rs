struct Vector {
    data: array<u32>,
}

@group(0) @binding(0) var<storage, read> a: Vector;           // Vector A - Input
@group(0) @binding(1) var<storage, read> b: Vector;           // Vector B - Input
@group(0) @binding(2) var<storage, read_write> c: Vector;     // Vector C - Output

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    c.data[idx] = a.data[idx] * b.data[idx];
}