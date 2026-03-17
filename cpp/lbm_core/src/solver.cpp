#include "lbm/solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace lbm {
namespace {

constexpr int kCx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
constexpr int kCy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
constexpr int kOpposite[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
constexpr double kWeights[9] = {
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
};

double equilibrium(int direction, double rho, double ux, double uy) {
    const double cu = 3.0 * (kCx[direction] * ux + kCy[direction] * uy);
    const double uu = ux * ux + uy * uy;
    return kWeights[direction] * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * uu);
}

}  // namespace

D2Q9BgkSolver::D2Q9BgkSolver(SimulationConfig config)
    : config_(std::move(config)),
      tau_(config_.relaxation_time()),
      f_(config_.nx * config_.ny * q_, 0.0),
      f_post_(config_.nx * config_.ny * q_, 0.0),
      f_next_(config_.nx * config_.ny * q_, 0.0),
      rho_(config_.nx * config_.ny, 1.0),
      ux_(config_.nx * config_.ny, 0.0),
      uy_(config_.nx * config_.ny, 0.0),
      obstacle_(config_.nx * config_.ny, 0) {
    build_obstacle_mask();
    initialize_equilibrium();
    compute_macros();
    apply_inlet_macros();
    validate_runtime_state("initialization");
}

void D2Q9BgkSolver::run(
    const std::function<void(int, const D2Q9BgkSolver&)>& snapshot_callback) {
    for (int iteration = 1; iteration <= config_.iterations; ++iteration) {
        advance_one_step();
        validate_runtime_state("time step");

        if (iteration % config_.save_stride == 0 || iteration == config_.iterations) {
            snapshot_callback(iteration, *this);
        }
    }
}

SnapshotFields D2Q9BgkSolver::compute_snapshot_fields() const {
    const int nx = config_.nx;
    const int ny = config_.ny;
    const int cell_count = nx * ny;

    SnapshotFields fields;
    fields.ux = ux_;
    fields.uy = uy_;
    fields.speed.assign(cell_count, 0.0);
    fields.vorticity.assign(cell_count, 0.0);
    fields.obstacle_mask = obstacle_;

    for (int index = 0; index < cell_count; ++index) {
        fields.speed[index] = std::sqrt(ux_[index] * ux_[index] + uy_[index] * uy_[index]);
    }

    for (int y = 0; y < ny; ++y) {
        const int y_minus = (y - 1 + ny) % ny;
        const int y_plus = (y + 1) % ny;
        for (int x = 1; x < nx - 1; ++x) {
            const int x_minus = x - 1;
            const int x_plus = x + 1;
            const double dv_dx =
                (uy_[scalar_index(x_plus, y)] - uy_[scalar_index(x_minus, y)]) * 0.5;
            const double du_dy =
                (ux_[scalar_index(x, y_plus)] - ux_[scalar_index(x, y_minus)]) * 0.5;
            fields.vorticity[scalar_index(x, y)] = dv_dx - du_dy;
        }

        fields.vorticity[scalar_index(0, y)] = fields.vorticity[scalar_index(1, y)];
        fields.vorticity[scalar_index(nx - 1, y)] =
            fields.vorticity[scalar_index(nx - 2, y)];
    }

    return fields;
}

RuntimeStabilityMetrics D2Q9BgkSolver::stability_metrics() const {
    RuntimeStabilityMetrics metrics;
    metrics.min_density = std::numeric_limits<double>::infinity();
    metrics.max_density = -std::numeric_limits<double>::infinity();
    metrics.max_speed = 0.0;

    const int cell_count = config_.nx * config_.ny;
    for (int index = 0; index < cell_count; ++index) {
        metrics.min_density = std::min(metrics.min_density, rho_[index]);
        metrics.max_density = std::max(metrics.max_density, rho_[index]);
        const double speed = std::sqrt(ux_[index] * ux_[index] + uy_[index] * uy_[index]);
        metrics.max_speed = std::max(metrics.max_speed, speed);
    }

    return metrics;
}

void D2Q9BgkSolver::build_obstacle_mask() {
    for (int y = 0; y < config_.ny; ++y) {
        for (int x = 0; x < config_.nx; ++x) {
            const int dx = x - config_.obstacle_cx;
            const int dy = y - config_.obstacle_cy;
            if (dx * dx + dy * dy <= config_.obstacle_r * config_.obstacle_r) {
                obstacle_[scalar_index(x, y)] = 1;
            }
        }
    }
}

void D2Q9BgkSolver::initialize_equilibrium() {
    for (int y = 0; y < config_.ny; ++y) {
        for (int x = 0; x < config_.nx; ++x) {
            const int cell = scalar_index(x, y);
            if (!obstacle_[cell]) {
                ux_[cell] = config_.u_in;
            }
            rho_[cell] = 1.0;

            for (int direction = 0; direction < q_; ++direction) {
                f_[distribution_index(x, y, direction)] =
                    equilibrium(direction, rho_[cell], ux_[cell], uy_[cell]);
            }
        }
    }
}

void D2Q9BgkSolver::advance_one_step() {
    compute_macros();
    apply_inlet_macros();
    collide();
    stream();
    compute_macros();
    apply_inlet_macros();
    current_step_ += 1;
}

void D2Q9BgkSolver::compute_macros() {
    for (int y = 0; y < config_.ny; ++y) {
        for (int x = 0; x < config_.nx; ++x) {
            const int cell = scalar_index(x, y);
            if (obstacle_[cell]) {
                rho_[cell] = 1.0;
                ux_[cell] = 0.0;
                uy_[cell] = 0.0;
                continue;
            }

            double rho_value = 0.0;
            double ux_value = 0.0;
            double uy_value = 0.0;
            for (int direction = 0; direction < q_; ++direction) {
                const double value = f_[distribution_index(x, y, direction)];
                rho_value += value;
                ux_value += value * kCx[direction];
                uy_value += value * kCy[direction];
            }

            rho_[cell] = rho_value;
            ux_[cell] = ux_value / rho_value;
            uy_[cell] = uy_value / rho_value;
        }
    }
}

void D2Q9BgkSolver::apply_inlet_macros() {
    for (int y = 0; y < config_.ny; ++y) {
        const int cell = scalar_index(0, y);
        if (obstacle_[cell]) {
            continue;
        }
        rho_[cell] = 1.0;
        ux_[cell] = config_.u_in;
        uy_[cell] = 0.0;
    }
}

void D2Q9BgkSolver::collide() {
    for (int y = 0; y < config_.ny; ++y) {
        for (int x = 0; x < config_.nx; ++x) {
            const int cell = scalar_index(x, y);
            for (int direction = 0; direction < q_; ++direction) {
                const int index = distribution_index(x, y, direction);
                if (obstacle_[cell]) {
                    f_post_[index] = f_[distribution_index(x, y, kOpposite[direction])];
                    continue;
                }

                const double feq =
                    equilibrium(direction, rho_[cell], ux_[cell], uy_[cell]);
                f_post_[index] = f_[index] - (f_[index] - feq) / tau_;
            }
        }
    }
}

void D2Q9BgkSolver::stream() {
    std::fill(f_next_.begin(), f_next_.end(), 0.0);

    for (int y = 0; y < config_.ny; ++y) {
        for (int x = 0; x < config_.nx; ++x) {
            const int cell = scalar_index(x, y);
            if (obstacle_[cell]) {
                continue;
            }

            for (int direction = 0; direction < q_; ++direction) {
                const int next_x = x + kCx[direction];
                const int next_y = (y + kCy[direction] + config_.ny) % config_.ny;
                const double value = f_post_[distribution_index(x, y, direction)];

                if (next_x < 0 || next_x >= config_.nx) {
                    f_next_[distribution_index(x, y, kOpposite[direction])] += value;
                } else if (obstacle_[scalar_index(next_x, next_y)]) {
                    f_next_[distribution_index(x, y, kOpposite[direction])] += value;
                } else {
                    f_next_[distribution_index(next_x, next_y, direction)] += value;
                }
            }
        }
    }

    for (int y = 0; y < config_.ny; ++y) {
        for (int direction = 0; direction < q_; ++direction) {
            f_next_[distribution_index(config_.nx - 1, y, direction)] =
                f_next_[distribution_index(config_.nx - 2, y, direction)];
        }
    }

    for (int y = 0; y < config_.ny; ++y) {
        const int cell = scalar_index(0, y);
        if (obstacle_[cell]) {
            continue;
        }
        for (int direction = 0; direction < q_; ++direction) {
            f_next_[distribution_index(0, y, direction)] =
                equilibrium(direction, 1.0, config_.u_in, 0.0);
        }
    }

    f_.swap(f_next_);
}

int D2Q9BgkSolver::scalar_index(int x, int y) const {
    return y * config_.nx + x;
}

int D2Q9BgkSolver::distribution_index(int x, int y, int direction) const {
    return (y * config_.nx + x) * q_ + direction;
}

void D2Q9BgkSolver::validate_runtime_state(const char* stage) const {
    const int cell_count = config_.nx * config_.ny;
    for (double value : f_) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(std::string("Non-finite distribution detected during ") +
                                     stage);
        }
    }

    for (int index = 0; index < cell_count; ++index) {
        if (!std::isfinite(rho_[index]) || !std::isfinite(ux_[index]) ||
            !std::isfinite(uy_[index])) {
            throw std::runtime_error(std::string("Non-finite macro state detected during ") +
                                     stage);
        }
        if (rho_[index] <= 0.0) {
            throw std::runtime_error(std::string("Non-positive density detected during ") +
                                     stage);
        }
    }

    const auto metrics = stability_metrics();
    if (metrics.max_speed >= 0.5) {
        throw std::runtime_error(
            std::string("Velocity magnitude exceeded basic stability guard during ") + stage);
    }
}

}  // namespace lbm
