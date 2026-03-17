#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "lbm/config.hpp"
#include "lbm/io.hpp"

namespace lbm {

struct RuntimeStabilityMetrics {
    double min_density = 0.0;
    double max_density = 0.0;
    double max_speed = 0.0;
};

class D2Q9BgkSolver {
   public:
    explicit D2Q9BgkSolver(SimulationConfig config);

    void run(const std::function<void(int, const D2Q9BgkSolver&)>& snapshot_callback);

    const SimulationConfig& config() const { return config_; }
    int current_step() const { return current_step_; }
    int nx() const { return config_.nx; }
    int ny() const { return config_.ny; }

    SnapshotFields compute_snapshot_fields() const;
    RuntimeStabilityMetrics stability_metrics() const;

   private:
    static constexpr int q_ = 9;

    SimulationConfig config_;
    int current_step_ = 0;
    double tau_ = 0.0;

    std::vector<double> f_;
    std::vector<double> f_post_;
    std::vector<double> f_next_;
    std::vector<double> rho_;
    std::vector<double> ux_;
    std::vector<double> uy_;
    std::vector<std::uint8_t> obstacle_;

    void build_obstacle_mask();
    void initialize_equilibrium();
    void advance_one_step();
    void compute_macros();
    void apply_inlet_macros();
    void collide();
    void stream();
    void validate_runtime_state(const char* stage) const;

    int scalar_index(int x, int y) const;
    int distribution_index(int x, int y, int direction) const;
};

}  // namespace lbm
