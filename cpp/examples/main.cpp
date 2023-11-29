#include <iostream>

#include "autd3-backend-cuda.hpp"
#include "autd3.hpp"
#include "autd3/link/nop.hpp"
#include "autd3/gain/holo.hpp"

int main() try {
  auto autd = autd3::ControllerBuilder()
                  .add_device(autd3::AUTD3(autd3::Vector3::Zero()))
                  .open_with_async(autd3::link::Nop::builder())
                  .get();

  autd3::modulation::Sine m(150);

  const autd3::Vector3 center = autd.geometry().center() + autd3::Vector3(0.0, 0.0, 150.0);

  const auto backend = std::make_shared<autd3::gain::holo::CUDABackend>();
  autd.send_async(m, autd3::gain::holo::GSPAT(backend)
      .add_focus(center + autd3::Vector3(30.0, 0.0, 0.0), 5e3 * autd3::gain::holo::Pascal)
      .add_focus(center - autd3::Vector3(30.0, 0.0, 0.0), 5e3 * autd3::gain::holo::Pascal))
      .get();

  std::cout << "press enter to finish..." << std::endl;
  std::cin.ignore();

  (void)autd.close_async().get();

  return 0;
} catch (std::exception& ex) {
  std::cerr << ex.what() << std::endl;
}
