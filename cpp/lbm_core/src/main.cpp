#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "lbm/config.hpp"
#include "lbm/io.hpp"
#include "lbm/solver.hpp"

int main(int argc, char** argv) {
    try {
        std::filesystem::path config_path = "configs/lbm_cylinder_base.json";

        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--config") {
                if (index + 1 >= argc) {
                    throw std::runtime_error("Missing value after --config");
                }
                config_path = argv[++index];
            } else {
                throw std::runtime_error("Unknown argument: " + argument);
            }
        }

        const lbm::SimulationConfig config = lbm::load_config(config_path);
        for (const auto& warning : config.basic_stability_warnings()) {
            std::cerr << "[LBM][warning] " << warning << '\n';
        }
        const auto run_dir = lbm::prepare_run_directory(config);
        lbm::write_manifest_json(run_dir, config, config_path);

        lbm::D2Q9BgkSolver solver(config);
        lbm::write_snapshot_csvs(
            run_dir, 0, solver.compute_snapshot_fields(), solver.nx(), solver.ny());

        solver.run([&](int step, const lbm::D2Q9BgkSolver& state) {
            const auto metrics = state.stability_metrics();
            std::cout << "[LBM] step=" << step << " rho_min=" << metrics.min_density
                      << " rho_max=" << metrics.max_density
                      << " speed_max=" << metrics.max_speed << '\n';
            std::cout << "[LBM] writing snapshot at step " << step << '\n';
            lbm::write_snapshot_csvs(
                run_dir,
                step,
                state.compute_snapshot_fields(),
                state.nx(),
                state.ny());
        });

        std::cout << "[LBM] completed run in " << run_dir << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
