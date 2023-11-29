/*
 * File: build.rs
 * Project: autd3capi-gain-holo
 * Created Date: 26/05/2023
 * Author: Shun Suzuki
 * -----
 * Last Modified: 29/11/2023
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2023 Shun Suzuki. All rights reserved.
 *
 */

use std::env;

use autd3capi_wrapper_generator::*;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    if let Err(e) = gen_c(
        &crate_dir,
        "../cpp/include/autd3-backend-holo/internal/native_methods",
    ) {
        eprintln!("{}", e);
    }
    if let Err(e) = gen_cs(&crate_dir, "../dotnet/cs/src/NativeMethods") {
        eprintln!("{}", e);
    }
    if let Err(e) = gen_unity(&crate_dir, "../dotnet/unity/Assets/Scripts/NativeMethods") {
        eprintln!("{}", e);
    }
    if let Err(e) = gen_py(crate_dir, "../python/pyautd3-backend-cuda/native_methods") {
        eprintln!("{}", e);
    }
}
