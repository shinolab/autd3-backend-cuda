#[cfg(target_os = "macos")]
fn main() {}

#[cfg(not(target_os = "macos"))]
fn main() {
    use cuda_config::*;

    let mut build = cc::Build::new();

    #[cfg(feature = "use_meter")]
    build.define("AUTD3_USE_METER", "1");

    build
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-arch=sm_75")
        .flag("-gencode=arch=compute_75,code=sm_75")
        .flag("-gencode=arch=compute_80,code=sm_80")
        .flag("-gencode=arch=compute_86,code=sm_86")
        .flag("-gencode=arch=compute_87,code=sm_87")
        .flag("-gencode=arch=compute_86,code=compute_86")
        .file("cuda_src/kernel.cu")
        .compile("autd3_cuda_kernel");

    if cfg!(target_os = "windows") {
        println!(
            "cargo:rustc-link-search=native={}",
            find_cuda_windows().display()
        );
    } else {
        find_cuda()
            .iter()
            .for_each(|path| println!("cargo:rustc-link-search=native={}", path.display()));
    };

    println!("cargo:rerun-if-changed=cuda_src/kernel.cu");
    println!("cargo:rustc-link-lib=dylib=cusolver");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_LIBRARY_PATH");
}
