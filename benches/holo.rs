#[cfg(feature = "bench-utilities")]
criterion::criterion_group!(
    benches,
    autd3_gain_holo::bench_utilities::foci::<autd3_backend_cuda::CUDABackend, 4>,
    autd3_gain_holo::bench_utilities::devices::<autd3_backend_cuda::CUDABackend, 2>
);
#[cfg(feature = "bench-utilities")]
criterion::criterion_main!(benches);

#[cfg(not(feature = "bench-utilities"))]
fn main() {}
