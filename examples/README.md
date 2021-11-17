# gpgpu examples
| Example name        | Description                                            | Required features  | Cargo command                                                       |
|---------------------|--------------------------------------------------------|--------------------|---------------------------------------------------------------------|
| simple-compute      | Simple compute example for starters                    | :heavy_minus_sign: | cargo r --example simple-compute                                    |
| parallel-compute    | More complex compute example, featuring parallel usage | :heavy_minus_sign: | cargo r --example parallel-compute                                  |
| mirror-image        | Simple image compute example that mirror an image      | :heavy_minus_sign: | cargo r --example mirror-image                                      |
| image-compatibility | `mirror-image` example using `image::ImageBuffer`      | integrate-image    | cargo r --example image-compatibility --features="integrate-image"  |
| webcam (*)          | Webcam shader implemented via compute                  | integrate-image    | cargo r --example webcam --features="integrate-image" --release     |
| ndarray             | Simple compute example using `ndarray::Array`          | integrate-ndarry   | cargo r --example ndarray --features="integrate-ndarray"            |

(*) Example makes use of release mode for visible performance issues.