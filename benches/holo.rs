#[cfg(feature = "bench-utilities")]
criterion::criterion_group!(
    benches,
    autd3_gain_holo::bench_utilities::sdp::<
        autd3_driver::acoustics::directivity::Sphere,
        autd3_backend_cuda::CUDABackend,
        4,
        4,
    >,
    autd3_gain_holo::bench_utilities::naive::<
        autd3_driver::acoustics::directivity::Sphere,
        autd3_backend_cuda::CUDABackend,
        4,
        4,
    >,
    autd3_gain_holo::bench_utilities::gs::<
        autd3_driver::acoustics::directivity::Sphere,
        autd3_backend_cuda::CUDABackend,
        4,
        4,
    >,
    autd3_gain_holo::bench_utilities::gspat::<
        autd3_driver::acoustics::directivity::Sphere,
        autd3_backend_cuda::CUDABackend,
        4,
        4,
    >,
    autd3_gain_holo::bench_utilities::lm::<
        autd3_driver::acoustics::directivity::Sphere,
        autd3_backend_cuda::CUDABackend,
        4,
        4,
    >
);
#[cfg(feature = "bench-utilities")]
criterion::criterion_main!(benches);

#[cfg(not(feature = "bench-utilities"))]
fn main() {
    panic!("This benchmark requires `bench-utilities` feature");
}
