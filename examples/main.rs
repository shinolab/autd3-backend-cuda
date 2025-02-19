use anyhow::Result;

use autd3::prelude::*;

use autd3_backend_cuda::CUDABackend;
use autd3_gain_holo::*;

fn main() -> Result<()> {
    let mut autd = Controller::open([AUTD3::default()], Nop::new())?;

    let backend = std::sync::Arc::new(CUDABackend::new()?);

    let center = autd.geometry().center() + Vector3::new(0., 0., 150.0 * mm);
    let p = Vector3::new(30. * mm, 0., 0.);
    let g = GSPAT {
        foci: vec![(center + p, 5e3 * Pa), (center - p, 5e3 * Pa)],
        option: GSPATOption::default(),
        backend,
    };

    autd.send(g)?;

    autd.close()?;

    Ok(())
}
