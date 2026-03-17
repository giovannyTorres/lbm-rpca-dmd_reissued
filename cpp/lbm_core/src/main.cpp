#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Config {
    int nx = 220;
    int ny = 80;
    double reynolds = 150.0;
    double u_in = 0.06;
    int iterations = 3000;
    int save_stride = 100;
    int seed = 42;
    int obstacle_cx = -1;
    int obstacle_cy = -1;
    int obstacle_r = 8;
    std::string output_root = "data/raw";
    std::string run_id = "phase1_baseline";
};

std::string trim(const std::string& s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

Config load_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("No se pudo abrir config: " + path);

    std::map<std::string, std::string> kv;
    std::string line;
    while (std::getline(in, line)) {
        auto cpos = line.find('#');
        if (cpos != std::string::npos) line = line.substr(0, cpos);
        line = trim(line);
        if (line.empty()) continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        auto key = trim(line.substr(0, eq));
        auto value = trim(line.substr(eq + 1));
        kv[key] = value;
    }

    Config cfg;
    auto get_i = [&](const std::string& k, int d) -> int { return kv.count(k) ? std::stoi(kv[k]) : d; };
    auto get_d = [&](const std::string& k, double d) -> double { return kv.count(k) ? std::stod(kv[k]) : d; };
    auto get_s = [&](const std::string& k, const std::string& d) -> std::string { return kv.count(k) ? kv[k] : d; };

    cfg.nx = get_i("nx", cfg.nx);
    cfg.ny = get_i("ny", cfg.ny);
    cfg.reynolds = get_d("reynolds", cfg.reynolds);
    cfg.u_in = get_d("u_in", cfg.u_in);
    cfg.iterations = get_i("iterations", cfg.iterations);
    cfg.save_stride = get_i("save_stride", cfg.save_stride);
    cfg.seed = get_i("seed", cfg.seed);
    cfg.obstacle_cx = get_i("obstacle_cx", cfg.obstacle_cx);
    cfg.obstacle_cy = get_i("obstacle_cy", cfg.obstacle_cy);
    cfg.obstacle_r = get_i("obstacle_r", cfg.obstacle_r);
    cfg.output_root = get_s("output_root", cfg.output_root);
    cfg.run_id = get_s("run_id", cfg.run_id);

    return cfg;
}

inline int idx(int x, int y, int nx) { return y * nx + x; }
inline int fidx(int x, int y, int k, int nx) { return (y * nx + x) * 9 + k; }

void write_scalar_csv(const fs::path& out, const std::vector<double>& field, int nx, int ny) {
    std::ofstream file(out);
    file << std::setprecision(10);
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            if (x) file << ',';
            file << field[idx(x, y, nx)];
        }
        file << '\n';
    }
}

void write_mask_csv(const fs::path& out, const std::vector<uint8_t>& mask, int nx, int ny) {
    std::ofstream file(out);
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            if (x) file << ',';
            file << static_cast<int>(mask[idx(x, y, nx)]);
        }
        file << '\n';
    }
}

int main(int argc, char** argv) {
    try {
        std::string config_path = "configs/lbm_cylinder_base.cfg";
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--config" && i + 1 < argc) {
                config_path = argv[++i];
            }
        }

        Config cfg = load_config(config_path);
        if (cfg.obstacle_cx < 0) cfg.obstacle_cx = cfg.nx / 4;
        if (cfg.obstacle_cy < 0) cfg.obstacle_cy = cfg.ny / 2;

        const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
        const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
        const int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
        const double w[9] = {4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
                             1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

        const double nu = static_cast<double>(cfg.u_in * 2.0 * cfg.obstacle_r / cfg.reynolds);
        const double tau = 3.0 * nu + 0.5;
        if (tau <= 0.5) throw std::runtime_error("tau <= 0.5, configuración inestable");

        const int nx = cfg.nx;
        const int ny = cfg.ny;
        const int n = nx * ny;

        std::vector<double> f(n * 9, 0.0), f_post(n * 9, 0.0), f_new(n * 9, 0.0);
        std::vector<double> rho(n, 1.0), ux(n, 0.0), uy(n, 0.0), mag(n, 0.0), vort(n, 0.0);
        std::vector<uint8_t> obstacle(n, 0);

        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                int dx = x - cfg.obstacle_cx;
                int dy = y - cfg.obstacle_cy;
                if (dx * dx + dy * dy <= cfg.obstacle_r * cfg.obstacle_r) {
                    obstacle[idx(x, y, nx)] = 1;
                }
            }
        }

        std::mt19937 rng(cfg.seed);
        std::uniform_real_distribution<double> jitter(-1e-6, 1e-6);

        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const int id = idx(x, y, nx);
                if (!obstacle[id] && x < nx / 6) ux[id] = cfg.u_in;
                rho[id] = 1.0;
                for (int k = 0; k < 9; ++k) {
                    double cu = 3.0 * (cx[k] * ux[id] + cy[k] * uy[id]);
                    double uu = ux[id] * ux[id] + uy[id] * uy[id];
                    f[fidx(x, y, k, nx)] = w[k] * rho[id] * (1.0 + cu + 0.5 * cu * cu - 1.5 * uu) + jitter(rng);
                }
            }
        }

        fs::path run_dir = fs::path(cfg.output_root) / cfg.run_id;
        fs::create_directories(run_dir);

        std::ofstream manifest(run_dir / "manifest.txt");
        manifest << "config_path=" << config_path << "\n"
                 << "nx=" << cfg.nx << "\n"
                 << "ny=" << cfg.ny << "\n"
                 << "reynolds=" << cfg.reynolds << "\n"
                 << "u_in=" << cfg.u_in << "\n"
                 << "iterations=" << cfg.iterations << "\n"
                 << "save_stride=" << cfg.save_stride << "\n"
                 << "seed=" << cfg.seed << "\n"
                 << "obstacle_cx=" << cfg.obstacle_cx << "\n"
                 << "obstacle_cy=" << cfg.obstacle_cy << "\n"
                 << "obstacle_r=" << cfg.obstacle_r << "\n"
                 << "nu=" << nu << "\n"
                 << "tau=" << tau << "\n";

        auto save_snapshot = [&](int step) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    int id = idx(x, y, nx);
                    mag[id] = std::sqrt(ux[id] * ux[id] + uy[id] * uy[id]);
                }
            }

            for (int y = 0; y < ny; ++y) {
                int ym = (y - 1 + ny) % ny;
                int yp = (y + 1) % ny;
                for (int x = 1; x < nx - 1; ++x) {
                    int xm = x - 1;
                    int xp = x + 1;
                    double dv_dx = (uy[idx(xp, y, nx)] - uy[idx(xm, y, nx)]) * 0.5;
                    double du_dy = (ux[idx(x, yp, nx)] - ux[idx(x, ym, nx)]) * 0.5;
                    vort[idx(x, y, nx)] = dv_dx - du_dy;
                }
                vort[idx(0, y, nx)] = vort[idx(1, y, nx)];
                vort[idx(nx - 1, y, nx)] = vort[idx(nx - 2, y, nx)];
            }

            std::ostringstream suffix;
            suffix << "t" << std::setw(6) << std::setfill('0') << step;
            write_scalar_csv(run_dir / ("ux_" + suffix.str() + ".csv"), ux, nx, ny);
            write_scalar_csv(run_dir / ("uy_" + suffix.str() + ".csv"), uy, nx, ny);
            write_scalar_csv(run_dir / ("umag_" + suffix.str() + ".csv"), mag, nx, ny);
            write_scalar_csv(run_dir / ("vort_" + suffix.str() + ".csv"), vort, nx, ny);
            write_mask_csv(run_dir / ("mask_" + suffix.str() + ".csv"), obstacle, nx, ny);
        };

        save_snapshot(0);

        for (int it = 1; it <= cfg.iterations; ++it) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    int id = idx(x, y, nx);
                    if (obstacle[id]) {
                        rho[id] = 1.0;
                        ux[id] = 0.0;
                        uy[id] = 0.0;
                        continue;
                    }
                    double rho_loc = 0.0, ux_loc = 0.0, uy_loc = 0.0;
                    for (int k = 0; k < 9; ++k) {
                        double val = f[fidx(x, y, k, nx)];
                        rho_loc += val;
                        ux_loc += val * cx[k];
                        uy_loc += val * cy[k];
                    }
                    ux_loc /= rho_loc;
                    uy_loc /= rho_loc;
                    rho[id] = rho_loc;
                    ux[id] = ux_loc;
                    uy[id] = uy_loc;
                }
            }

            for (int y = 0; y < ny; ++y) {
                int id = idx(0, y, nx);
                if (!obstacle[id]) {
                    rho[id] = 1.0;
                    ux[id] = cfg.u_in;
                    uy[id] = 0.0;
                }
            }

            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    int id = idx(x, y, nx);
                    for (int k = 0; k < 9; ++k) {
                        if (obstacle[id]) {
                            f_post[fidx(x, y, k, nx)] = f[fidx(x, y, opp[k], nx)];
                            continue;
                        }
                        double cu = 3.0 * (cx[k] * ux[id] + cy[k] * uy[id]);
                        double uu = ux[id] * ux[id] + uy[id] * uy[id];
                        double feq = w[k] * rho[id] * (1.0 + cu + 0.5 * cu * cu - 1.5 * uu);
                        f_post[fidx(x, y, k, nx)] = f[fidx(x, y, k, nx)] - (f[fidx(x, y, k, nx)] - feq) / tau;
                    }
                }
            }

            std::fill(f_new.begin(), f_new.end(), 0.0);
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    int id = idx(x, y, nx);
                    if (obstacle[id]) continue;
                    for (int k = 0; k < 9; ++k) {
                        int xn = x + cx[k];
                        int yn = (y + cy[k] + ny) % ny;
                        double val = f_post[fidx(x, y, k, nx)];

                        if (xn < 0 || xn >= nx) {
                            f_new[fidx(x, y, opp[k], nx)] += val;
                        } else if (obstacle[idx(xn, yn, nx)]) {
                            f_new[fidx(x, y, opp[k], nx)] += val;
                        } else {
                            f_new[fidx(xn, yn, k, nx)] += val;
                        }
                    }
                }
            }

            for (int y = 0; y < ny; ++y) {
                for (int k = 0; k < 9; ++k) {
                    f_new[fidx(nx - 1, y, k, nx)] = f_new[fidx(nx - 2, y, k, nx)];
                }
            }

            for (int y = 0; y < ny; ++y) {
                int id = idx(0, y, nx);
                if (obstacle[id]) continue;
                for (int k = 0; k < 9; ++k) {
                    double cu = 3.0 * (cx[k] * cfg.u_in + cy[k] * 0.0);
                    double uu = cfg.u_in * cfg.u_in;
                    f_new[fidx(0, y, k, nx)] = w[k] * 1.0 * (1.0 + cu + 0.5 * cu * cu - 1.5 * uu);
                }
            }

            f.swap(f_new);

            if (it % cfg.save_stride == 0 || it == cfg.iterations) {
                std::cout << "[LBM] snapshot step=" << it << std::endl;
                save_snapshot(it);
            }
        }

        std::cout << "[LBM] finalizado en: " << run_dir << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}
