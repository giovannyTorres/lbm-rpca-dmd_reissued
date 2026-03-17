#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace lbm {

struct SimulationConfig {
    int nx = 220;
    int ny = 80;
    double reynolds = 150.0;
    double u_in = 0.06;
    int iterations = 1200;
    int save_stride = 200;
    int obstacle_cx = 55;
    int obstacle_cy = 40;
    int obstacle_r = 8;
    std::string output_root = "data/raw";
    std::string run_id = "phase1_cylinder_re150";

    double kinematic_viscosity() const;
    double relaxation_time() const;
    void apply_derived_defaults();
    void validate() const;
    std::vector<std::string> basic_stability_warnings() const;
};

SimulationConfig load_config(const std::filesystem::path& path);

}  // namespace lbm
