/*
 * File: main.rs
 * Project: examples
 * Created Date: 29/11/2023
 * Author: Shun Suzuki
 * -----
 * Last Modified: 29/11/2023
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2023 Shun Suzuki. All rights reserved.
 *
 */

use anyhow::Result;

use autd3::prelude::*;

use autd3_backend_cuda::CUDABackend;
use autd3_gain_holo::*;

#[tokio::main]
async fn main() -> Result<()> {
    let mut autd = Controller::builder()
        .add_device(AUTD3::new(Vector3::zeros()))
        .open_with(Nop::builder())
        .await?;

    let backend = CUDABackend::new()?;

    let center = autd.geometry.center() + Vector3::new(0., 0., 150.0 * MILLIMETER);
    let p = Vector3::new(30. * MILLIMETER, 0., 0.);
    let g = GSPAT::new(backend)
        .add_focus(center + p, 5e3 * Pascal)
        .add_focus(center - p, 5e3 * Pascal);

    autd.send(g).await?;

    autd.close().await?;

    Ok(())
}
