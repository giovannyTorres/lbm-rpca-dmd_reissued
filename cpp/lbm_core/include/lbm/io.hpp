#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "lbm/config.hpp"

namespace lbm {

struct SnapshotFields {
    std::vector<double> ux;
    std::vector<double> uy;
    std::vector<double> speed;
    std::vector<double> vorticity;
    std::vector<std::uint8_t> obstacle_mask;
};

void validate_snapshot_fields(const SnapshotFields& fields, int nx, int ny);
std::filesystem::path prepare_run_directory(const SimulationConfig& config);
void write_manifest_json(
    const std::filesystem::path& run_dir,
    const SimulationConfig& config,
    const std::filesystem::path& config_path);
void write_snapshot_csvs(
    const std::filesystem::path& run_dir,
    int step,
    const SnapshotFields& fields,
    int nx,
    int ny);

}  // namespace lbm
