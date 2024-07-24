use anyhow::Result;

use autd3::prelude::*;

use autd3_backend_cuda::CUDABackend;
use autd3_gain_holo::*;

#[tokio::main]
async fn main() -> Result<()> {
    let mut autd = Controller::builder([AUTD3::new(Vector3::zeros())])
        .open(Nop::builder())
        .await?;

    let backend = CUDABackend::new()?;

    let center = autd.geometry().center() + Vector3::new(0., 0., 150.0 * mm);
    let p = Vector3::new(30. * mm, 0., 0.);
    let g = GSPAT::new(backend, [(center + p, 5e3 * Pa), (center - p, 5e3 * Pa)]);

    autd.send(g).await?;

    autd.close().await?;

    Ok(())
}
