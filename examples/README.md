# gpgpu examples
| Example name        | Description                                                | Required features | Cargo command                                                       |
|---------------------|------------------------------------------------------------|-------------------|---------------------------------------------------------------------|
| simple-compute      | Simple compute example for starters                        | None              | cargo r --example simple-compute                                    |
| parallel-compute    | More complex compute example, featuring parallel usage     | None              | cargo r --example parallel-compute                                  |
| mirror-image        | Simple image compute example that mirror an image          | None              | cargo r --example mirror-image          |
| image-compatibility | mirror-image example with compatibility with `image` crate | integrate-image   | cargo r --example image-compatibility --features="integrate-image"  |
| webcam (*)              | Webcam shader implemented via compute                      | integrate-image   | cargo r --release --example webcam --features="integrate-image" |

(*) Example makes use of release mode for visible performance issues.