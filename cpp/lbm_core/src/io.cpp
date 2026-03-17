#include "lbm/io.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace lbm {
namespace {

int scalar_index(int x, int y, int nx) {
    return y * nx + x;
}

std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char ch : value) {
        switch (ch) {
            case '"':
                escaped += "\\\"";
                break;
            case '\\':
                escaped += "\\\\";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped.push_back(ch);
                break;
        }
    }
    return escaped;
}

void write_scalar_csv(
    const std::filesystem::path& output_path,
    const std::vector<double>& field,
    int nx,
    int ny) {
    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("Could not write file: " + output_path.string());
    }

    output << std::setprecision(10);
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            if (x > 0) {
                output << ',';
            }
            output << field[scalar_index(x, y, nx)];
        }
        output << '\n';
    }
}

void write_mask_csv(
    const std::filesystem::path& output_path,
    const std::vector<std::uint8_t>& field,
    int nx,
    int ny) {
    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("Could not write file: " + output_path.string());
    }

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            if (x > 0) {
                output << ',';
            }
            output << static_cast<int>(field[scalar_index(x, y, nx)]);
        }
        output << '\n';
    }
}

std::string step_suffix(int step) {
    std::ostringstream buffer;
    buffer << "t" << std::setw(6) << std::setfill('0') << step;
    return buffer.str();
}

void validate_scalar_field(
    const std::vector<double>& field,
    const char* field_name,
    int expected_size) {
    if (static_cast<int>(field.size()) != expected_size) {
        throw std::runtime_error(std::string("Invalid field size for ") + field_name);
    }
    for (double value : field) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                std::string("Non-finite value detected in field ") + field_name);
        }
    }
}

}  // namespace

void validate_snapshot_fields(const SnapshotFields& fields, int nx, int ny) {
    const int expected_size = nx * ny;
    validate_scalar_field(fields.ux, "ux", expected_size);
    validate_scalar_field(fields.uy, "uy", expected_size);
    validate_scalar_field(fields.speed, "speed", expected_size);
    validate_scalar_field(fields.vorticity, "vorticity", expected_size);

    if (static_cast<int>(fields.obstacle_mask.size()) != expected_size) {
        throw std::runtime_error("Invalid field size for obstacle mask");
    }

    constexpr double tolerance = 1e-8;
    for (int index = 0; index < expected_size; ++index) {
        const auto mask_value = fields.obstacle_mask[index];
        if (!(mask_value == 0 || mask_value == 1)) {
            throw std::runtime_error("Obstacle mask contains values other than 0 or 1");
        }

        const double expected_speed = std::sqrt(
            fields.ux[index] * fields.ux[index] + fields.uy[index] * fields.uy[index]);
        if (std::abs(fields.speed[index] - expected_speed) > tolerance) {
            throw std::runtime_error("Speed field is inconsistent with ux and uy");
        }
    }
}

std::filesystem::path prepare_run_directory(const SimulationConfig& config) {
    const auto run_dir =
        std::filesystem::path(config.output_root) / config.run_id;
    std::filesystem::create_directories(run_dir);
    return run_dir;
}

void write_manifest_json(
    const std::filesystem::path& run_dir,
    const SimulationConfig& config,
    const std::filesystem::path& config_path) {
    std::ofstream output(run_dir / "manifest.json");
    if (!output) {
        throw std::runtime_error("Could not write manifest.json");
    }

    output << std::setprecision(10);
    output << "{\n";
    output << "  \"config_path\": \"" << json_escape(config_path.string()) << "\",\n";
    output << "  \"nx\": " << config.nx << ",\n";
    output << "  \"ny\": " << config.ny << ",\n";
    output << "  \"reynolds\": " << config.reynolds << ",\n";
    output << "  \"u_in\": " << config.u_in << ",\n";
    output << "  \"iterations\": " << config.iterations << ",\n";
    output << "  \"save_stride\": " << config.save_stride << ",\n";
    output << "  \"obstacle_cx\": " << config.obstacle_cx << ",\n";
    output << "  \"obstacle_cy\": " << config.obstacle_cy << ",\n";
    output << "  \"obstacle_r\": " << config.obstacle_r << ",\n";
    output << "  \"output_root\": \"" << json_escape(config.output_root) << "\",\n";
    output << "  \"run_id\": \"" << json_escape(config.run_id) << "\",\n";
    output << "  \"nu\": " << config.kinematic_viscosity() << ",\n";
    output << "  \"tau\": " << config.relaxation_time() << "\n";
    output << "}\n";
}

void write_snapshot_csvs(
    const std::filesystem::path& run_dir,
    int step,
    const SnapshotFields& fields,
    int nx,
    int ny) {
    validate_snapshot_fields(fields, nx, ny);

    const auto suffix = step_suffix(step);
    write_scalar_csv(run_dir / ("ux_" + suffix + ".csv"), fields.ux, nx, ny);
    write_scalar_csv(run_dir / ("uy_" + suffix + ".csv"), fields.uy, nx, ny);
    write_scalar_csv(run_dir / ("speed_" + suffix + ".csv"), fields.speed, nx, ny);
    write_scalar_csv(
        run_dir / ("vorticity_" + suffix + ".csv"), fields.vorticity, nx, ny);
    write_mask_csv(run_dir / ("mask_" + suffix + ".csv"), fields.obstacle_mask, nx, ny);
}

}  // namespace lbm
