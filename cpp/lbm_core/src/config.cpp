#include "lbm/config.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace lbm {
namespace {

class FlatJsonParser {
   public:
    explicit FlatJsonParser(std::string source) : source_(std::move(source)) {}

    std::unordered_map<std::string, std::string> parse_object() {
        std::unordered_map<std::string, std::string> values;
        skip_whitespace();
        expect('{');
        skip_whitespace();

        if (peek() == '}') {
            ++position_;
            return values;
        }

        while (true) {
            skip_whitespace();
            const std::string key = parse_string();
            skip_whitespace();
            expect(':');
            skip_whitespace();
            values[key] = parse_value();
            skip_whitespace();

            const char next = peek();
            if (next == ',') {
                ++position_;
                continue;
            }
            if (next == '}') {
                ++position_;
                break;
            }
            throw std::runtime_error("Invalid JSON object: expected ',' or '}'");
        }

        skip_whitespace();
        if (position_ != source_.size()) {
            throw std::runtime_error("Invalid JSON object: trailing characters");
        }
        return values;
    }

   private:
    std::string source_;
    std::size_t position_ = 0;

    char peek() const {
        if (position_ >= source_.size()) {
            return '\0';
        }
        return source_[position_];
    }

    void skip_whitespace() {
        while (position_ < source_.size() &&
               std::isspace(static_cast<unsigned char>(source_[position_]))) {
            ++position_;
        }
    }

    void expect(char expected) {
        if (peek() != expected) {
            throw std::runtime_error(std::string("Invalid JSON object: expected '") +
                                     expected + "'");
        }
        ++position_;
    }

    std::string parse_string() {
        expect('"');
        std::string value;
        while (position_ < source_.size()) {
            const char current = source_[position_++];
            if (current == '"') {
                return value;
            }
            if (current == '\\') {
                if (position_ >= source_.size()) {
                    throw std::runtime_error("Invalid JSON string escape");
                }
                const char escaped = source_[position_++];
                switch (escaped) {
                    case '"':
                    case '\\':
                    case '/':
                        value.push_back(escaped);
                        break;
                    case 'b':
                        value.push_back('\b');
                        break;
                    case 'f':
                        value.push_back('\f');
                        break;
                    case 'n':
                        value.push_back('\n');
                        break;
                    case 'r':
                        value.push_back('\r');
                        break;
                    case 't':
                        value.push_back('\t');
                        break;
                    default:
                        throw std::runtime_error("Unsupported JSON escape sequence");
                }
                continue;
            }
            value.push_back(current);
        }
        throw std::runtime_error("Unterminated JSON string");
    }

    std::string parse_value() {
        if (peek() == '"') {
            return parse_string();
        }

        const std::size_t start = position_;
        while (position_ < source_.size()) {
            const char current = source_[position_];
            if (current == ',' || current == '}') {
                break;
            }
            ++position_;
        }

        const std::size_t end = position_;
        std::size_t first = start;
        while (first < end &&
               std::isspace(static_cast<unsigned char>(source_[first]))) {
            ++first;
        }
        std::size_t last = end;
        while (last > first &&
               std::isspace(static_cast<unsigned char>(source_[last - 1]))) {
            --last;
        }
        if (first == last) {
            throw std::runtime_error("Invalid JSON value");
        }
        return source_.substr(first, last - first);
    }
};

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open config file: " + path.string());
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

int read_int(
    const std::unordered_map<std::string, std::string>& values,
    const std::string& key,
    int default_value) {
    const auto it = values.find(key);
    if (it == values.end()) {
        return default_value;
    }
    return std::stoi(it->second);
}

double read_double(
    const std::unordered_map<std::string, std::string>& values,
    const std::string& key,
    double default_value) {
    const auto it = values.find(key);
    if (it == values.end()) {
        return default_value;
    }
    return std::stod(it->second);
}

std::string read_string(
    const std::unordered_map<std::string, std::string>& values,
    const std::string& key,
    std::string default_value) {
    const auto it = values.find(key);
    if (it == values.end()) {
        return default_value;
    }
    return it->second;
}

}  // namespace

double SimulationConfig::kinematic_viscosity() const {
    return u_in * (2.0 * static_cast<double>(obstacle_r)) / reynolds;
}

double SimulationConfig::relaxation_time() const {
    return 3.0 * kinematic_viscosity() + 0.5;
}

void SimulationConfig::apply_derived_defaults() {
    if (obstacle_cx <= 0) {
        obstacle_cx = nx / 4;
    }
    if (obstacle_cy <= 0) {
        obstacle_cy = ny / 2;
    }
}

void SimulationConfig::validate() const {
    if (nx <= 2 || ny <= 2) {
        throw std::runtime_error("nx and ny must be greater than 2");
    }
    if (reynolds <= 0.0 || u_in <= 0.0) {
        throw std::runtime_error("reynolds and u_in must be positive");
    }
    if (iterations <= 0 || save_stride <= 0 || obstacle_r <= 0) {
        throw std::runtime_error(
            "iterations, save_stride and obstacle_r must be positive");
    }
    if (run_id.empty() || output_root.empty()) {
        throw std::runtime_error("run_id and output_root must be non-empty");
    }
    if (obstacle_cx - obstacle_r < 1 || obstacle_cx + obstacle_r >= nx - 1) {
        throw std::runtime_error("Obstacle must stay away from inlet/outlet boundaries");
    }
    if (obstacle_cy - obstacle_r < 0 || obstacle_cy + obstacle_r >= ny) {
        throw std::runtime_error("Obstacle must stay inside the periodic y-domain");
    }
    if (relaxation_time() <= 0.5) {
        throw std::runtime_error("tau <= 0.5 leads to an unstable configuration");
    }
}

std::vector<std::string> SimulationConfig::basic_stability_warnings() const {
    std::vector<std::string> warnings;
    const double tau = relaxation_time();

    if (tau < 0.53) {
        warnings.push_back("tau is close to 0.5; increase viscosity or reduce u_in");
    }
    if (u_in > 0.08) {
        warnings.push_back(
            "u_in is relatively high for a basic BGK setup; consider <= 0.08");
    }
    if (reynolds > 200.0) {
        warnings.push_back(
            "reynolds is high for the current coarse validation regime; start <= 200");
    }
    if (nx < 12 * obstacle_r) {
        warnings.push_back(
            "nx may be too short downstream of the obstacle for comfortable transients");
    }
    if (ny < 6 * obstacle_r) {
        warnings.push_back(
            "ny may be too tight around the obstacle for conservative initial runs");
    }

    return warnings;
}

SimulationConfig load_config(const std::filesystem::path& path) {
    const auto values = FlatJsonParser(read_text_file(path)).parse_object();

    SimulationConfig config;
    config.nx = read_int(values, "nx", config.nx);
    config.ny = read_int(values, "ny", config.ny);
    config.reynolds = read_double(values, "reynolds", config.reynolds);
    config.u_in = read_double(values, "u_in", config.u_in);
    config.iterations = read_int(values, "iterations", config.iterations);
    config.save_stride = read_int(values, "save_stride", config.save_stride);
    config.obstacle_cx = read_int(values, "obstacle_cx", config.obstacle_cx);
    config.obstacle_cy = read_int(values, "obstacle_cy", config.obstacle_cy);
    config.obstacle_r = read_int(values, "obstacle_r", config.obstacle_r);
    config.output_root = read_string(values, "output_root", config.output_root);
    config.run_id = read_string(values, "run_id", config.run_id);

    config.apply_derived_defaults();
    config.validate();
    return config;
}

}  // namespace lbm
