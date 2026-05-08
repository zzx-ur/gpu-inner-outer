#include "gsc/config.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace gsc {

namespace {

std::string trim(const std::string& value) {
    std::size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }
    std::size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(start, end - start);
}

std::vector<double> parse_csv_doubles(const std::string& value) {
    std::vector<double> out;
    std::stringstream ss(value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        out.push_back(std::stod(trim(token)));
    }
    return out;
}

void assign_array4(const std::vector<double>& values, double out[4], const char* key) {
    if (values.size() != 4) {
        throw std::runtime_error(std::string(key) + " expects 4 comma-separated numbers");
    }
    for (int i = 0; i < 4; ++i) {
        out[i] = values[i];
    }
}

void assign_array2(const std::vector<double>& values, double out[2], const char* key) {
    if (values.size() != 2) {
        throw std::runtime_error(std::string(key) + " expects 2 comma-separated numbers");
    }
    for (int i = 0; i < 2; ++i) {
        out[i] = values[i];
    }
}

HyperRect4D make_rect(const double lb[4], const double ub[4]) {
    HyperRect4D rect;
    for (int i = 0; i < 4; ++i) {
        rect.lb[i] = lb[i];
        rect.ub[i] = ub[i];
    }
    return rect;
}

// 双曲线约束检查: x1^2 - x2^2 <= a 且 b*x2^2 - x1^2 <= c
bool check_hyperbolic_constraint(const double x[kStateDim], const HyperbolicConstraintParams& params) {
    const double x1_sq = x[0] * x[0];
    const double x2_sq = x[1] * x[1];
    return (x1_sq - x2_sq <= params.a) && (params.b * x2_sq - x1_sq <= params.c);
}

// 椭圆约束检查: (x1/a)^2 + (x2/b)^2 <= 1
bool check_elliptic_constraint(const double x[kStateDim], const EllipticConstraintParams& params) {
    const double term1 = (x[0] / params.a) * (x[0] / params.a);
    const double term2 = (x[1] / params.b) * (x[1] / params.b);
    return (term1 + term2 <= 1.0);
}

}  // namespace

bool point_in_rect(const HyperRect4D& rect, const double x[kStateDim]) {
    for (int dim = 0; dim < kStateDim; ++dim) {
        if (x[dim] < rect.lb[dim] || x[dim] > rect.ub[dim]) {
            return false;
        }
    }
    return true;
}

StateConstraintFn make_state_constraint(const CaseConfig& cfg) {
    switch (cfg.constraint_type) {
        case ConstraintType::kNone:
            return nullptr;
        case ConstraintType::kHyperbolic: {
            const auto params = cfg.hyperbolic_params;
            return [params](const double x[kStateDim]) {
                return check_hyperbolic_constraint(x, params);
            };
        }
        case ConstraintType::kElliptic: {
            const auto params = cfg.elliptic_params;
            return [params](const double x[kStateDim]) {
                return check_elliptic_constraint(x, params);
            };
        }
        case ConstraintType::kCustom:
            return cfg.custom_constraint;
        default:
            return nullptr;
    }
}

bool satisfies_state_constraint(const CaseConfig& cfg, const double x[kStateDim]) {
    switch (cfg.constraint_type) {
        case ConstraintType::kNone:
            return true;
        case ConstraintType::kHyperbolic:
            return check_hyperbolic_constraint(x, cfg.hyperbolic_params);
        case ConstraintType::kElliptic:
            return check_elliptic_constraint(x, cfg.elliptic_params);
        case ConstraintType::kCustom:
            if (cfg.custom_constraint) {
                return cfg.custom_constraint(x);
            }
            return true;
        default:
            return true;
    }
}

CaseConfig make_default_case() {
    CaseConfig cfg;
    const double candidate_lb[4] = {-0.5, -0.5, -3.141592653589793, 0.0};
    const double candidate_ub[4] = { 0.5,  0.5,  3.141592653589793, 1.5};
    cfg.candidate = make_rect(candidate_lb, candidate_ub);
    cfg.map = make_rect(cfg.state_lb, cfg.state_ub);
    const double obstacle_lb[4] = {0.5, -0.5, -3.141592653589793, 0.0};
    const double obstacle_ub[4] = {1.0,  0.5,  3.141592653589793, 2.0};
    cfg.obstacles.push_back(make_rect(obstacle_lb, obstacle_ub));
    return cfg;
}

CaseConfig make_wraparound_case() {
    CaseConfig cfg = make_default_case();
    cfg.name = "wraparound_case";
    cfg.state_lb[2] = -3.141592653589793;
    cfg.state_ub[2] = 3.141592653589793;
    cfg.state_eta[2] = 0.5235987755982988;
    cfg.disturbance_half_width[2] = 0.35;
    const double candidate_lb[4] = {-0.5, -0.5, 2.6, 0.0};
    const double candidate_ub[4] = { 0.5,  0.5, 3.141592653589793, 1.5};
    cfg.candidate = make_rect(candidate_lb, candidate_ub);
    cfg.map = make_rect(cfg.state_lb, cfg.state_ub);
    cfg.obstacles.clear();
    return cfg;
}

CaseConfig make_smoke_case() {
    CaseConfig cfg = make_default_case();
    cfg.name = "smoke_case";
    cfg.state_lb[0] = -6.0;
    cfg.state_ub[0] = 6.0;
    cfg.state_lb[1] = -6.0;
    cfg.state_ub[1] = 6.0;
    cfg.state_lb[2] = -3.141592653589793;
    cfg.state_ub[2] = 3.141592653589793;
    cfg.state_lb[3] = 0.0;
    cfg.state_ub[3] = 4.0;
    cfg.state_eta[0] = 0.3;
    cfg.state_eta[1] = 0.3;
    cfg.state_eta[2] = 0.39269908169872414;
    cfg.state_eta[3] = 0.5;
    cfg.input_lb[0] = -1.0;
    cfg.input_ub[0] = 1.0;
    cfg.input_eta[0] = 0.25;
    cfg.input_lb[1] = -1.5;
    cfg.input_ub[1] = 1.5;
    cfg.input_eta[1] = 0.3;
    const double candidate_lb[4] = {-1.0, -1.0, -3.141592653589793, 0.0};
    const double candidate_ub[4] = { 1.0,  1.0,  3.141592653589793, 2.0};
    cfg.candidate = make_rect(candidate_lb, candidate_ub);
    cfg.map = make_rect(cfg.state_lb, cfg.state_ub);
    cfg.obstacles.clear();
    cfg.max_iterations = 4;
    cfg.verbose = false;
    return cfg;
}

CaseConfig make_hyperbolic_case() {
    // 与 MATLAB legacy 示例对齐的双曲线约束案例
    CaseConfig cfg;
    cfg.name = "hyperbolic_case";
    
    // 状态空间: x, y, theta (3D，但我们用4D，第4维设为常量范围)
    cfg.state_lb[0] = -3.5;
    cfg.state_ub[0] = 3.5;
    cfg.state_lb[1] = -3.5;
    cfg.state_ub[1] = 3.5;
    cfg.state_lb[2] = -3.141592653589793;
    cfg.state_ub[2] = 3.141592653589793;
    cfg.state_lb[3] = 0.2;  // 速度下界
    cfg.state_ub[3] = 2.0;  // 速度上界
    cfg.state_eta[0] = 0.1;
    cfg.state_eta[1] = 0.1;
    cfg.state_eta[2] = 0.1;
    cfg.state_eta[3] = 0.2;
    cfg.wrap_dim = 2;
    
    // 输入空间: 速度, 角速度
    cfg.input_lb[0] = 0.2;
    cfg.input_ub[0] = 2.0;
    cfg.input_lb[1] = -1.0;
    cfg.input_ub[1] = 1.0;
    cfg.input_eta[0] = 0.3;
    cfg.input_eta[1] = 0.4;
    
    // 候选集
    const double candidate_lb[4] = {-1.0, -1.0, -3.141592653589793, 0.2};
    const double candidate_ub[4] = { 1.0,  1.0,  3.141592653589793, 2.0};
    cfg.candidate = make_rect(candidate_lb, candidate_ub);
    
    // map 使用 state 边界
    cfg.map = make_rect(cfg.state_lb, cfg.state_ub);
    
    // 双曲线约束: x1^2 - x2^2 <= 4 且 4*x2^2 - x1^2 <= 16
    cfg.constraint_type = ConstraintType::kHyperbolic;
    cfg.hyperbolic_params.a = 4.0;   // x1^2 - x2^2 <= 4
    cfg.hyperbolic_params.b = 4.0;   // 4*x2^2 - x1^2 <= 16
    cfg.hyperbolic_params.c = 16.0;
    
    // 扰动
    cfg.disturbance_half_width[0] = 0.12;
    cfg.disturbance_half_width[1] = 0.12;
    cfg.disturbance_half_width[2] = 0.12;
    cfg.disturbance_half_width[3] = 0.0;
    
    cfg.sample_time = 1.0;
    cfg.integration_substeps = 32;
    cfg.max_iterations = 100;
    cfg.verbose = true;
    
    return cfg;
}

CaseConfig load_case_config(const std::string& path) {
    CaseConfig cfg = make_default_case();
    cfg.obstacles.clear();  // 清除默认障碍物，由配置文件显式定义

    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open config: " + path);
    }

    std::string line;
    while (std::getline(input, line)) {
        const auto hash_pos = line.find('#');
        if (hash_pos != std::string::npos) {
            line = line.substr(0, hash_pos);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        const auto eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            throw std::runtime_error("expected key=value line: " + line);
        }

        const std::string key = trim(line.substr(0, eq_pos));
        const std::string value = trim(line.substr(eq_pos + 1));

        if (key == "name") {
            cfg.name = value;
        } else if (key == "state_lb") {
            assign_array4(parse_csv_doubles(value), cfg.state_lb, "state_lb");
        } else if (key == "state_ub") {
            assign_array4(parse_csv_doubles(value), cfg.state_ub, "state_ub");
        } else if (key == "state_eta") {
            assign_array4(parse_csv_doubles(value), cfg.state_eta, "state_eta");
        } else if (key == "wrap_dim") {
            cfg.wrap_dim = std::stoi(value);
        } else if (key == "input_lb") {
            assign_array2(parse_csv_doubles(value), cfg.input_lb, "input_lb");
        } else if (key == "input_ub") {
            assign_array2(parse_csv_doubles(value), cfg.input_ub, "input_ub");
        } else if (key == "input_eta") {
            assign_array2(parse_csv_doubles(value), cfg.input_eta, "input_eta");
        } else if (key == "candidate_lb") {
            assign_array4(parse_csv_doubles(value), cfg.candidate.lb, "candidate_lb");
        } else if (key == "candidate_ub") {
            assign_array4(parse_csv_doubles(value), cfg.candidate.ub, "candidate_ub");
        } else if (key == "candidate_contracted_lb") {
            assign_array4(parse_csv_doubles(value), cfg.candidate_contracted.lb, "candidate_contracted_lb");
        } else if (key == "candidate_contracted_ub") {
            assign_array4(parse_csv_doubles(value), cfg.candidate_contracted.ub, "candidate_contracted_ub");
        } else if (key == "map_lb") {
            assign_array4(parse_csv_doubles(value), cfg.map.lb, "map_lb");
        } else if (key == "map_ub") {
            assign_array4(parse_csv_doubles(value), cfg.map.ub, "map_ub");
        } else if (key == "disturbance_half_width") {
            assign_array4(parse_csv_doubles(value), cfg.disturbance_half_width, "disturbance_half_width");
        } else if (key == "sample_time") {
            cfg.sample_time = std::stod(value);
        } else if (key == "integration_substeps") {
            cfg.integration_substeps = static_cast<std::uint32_t>(std::stoul(value));
        } else if (key == "max_iterations") {
            cfg.max_iterations = static_cast<std::uint32_t>(std::stoul(value));
        } else if (key == "verbose") {
            const auto lowered = trim(value);
            cfg.verbose = (lowered == "1" || lowered == "true" || lowered == "TRUE");
        } else if (key == "obstacle") {
            const auto values = parse_csv_doubles(value);
            if (values.size() != 8) {
                throw std::runtime_error("obstacle expects 8 comma-separated numbers");
            }
            HyperRect4D rect;
            for (int i = 0; i < 4; ++i) {
                rect.lb[i] = values[i];
                rect.ub[i] = values[i + 4];
            }
            cfg.obstacles.push_back(rect);
        } else if (key == "constraint_type") {
            // 解析约束类型
            if (value == "none") {
                cfg.constraint_type = ConstraintType::kNone;
            } else if (value == "hyperbolic") {
                cfg.constraint_type = ConstraintType::kHyperbolic;
            } else if (value == "elliptic") {
                cfg.constraint_type = ConstraintType::kElliptic;
            } else {
                throw std::runtime_error("unknown constraint_type: " + value);
            }
        } else if (key == "hyperbolic_a") {
            cfg.hyperbolic_params.a = std::stod(value);
        } else if (key == "hyperbolic_b") {
            cfg.hyperbolic_params.b = std::stod(value);
        } else if (key == "hyperbolic_c") {
            cfg.hyperbolic_params.c = std::stod(value);
        } else if (key == "hyperbolic_contracted_a") {
            cfg.hyperbolic_contracted_params.a = std::stod(value);
        } else if (key == "hyperbolic_contracted_b") {
            cfg.hyperbolic_contracted_params.b = std::stod(value);
        } else if (key == "hyperbolic_contracted_c") {
            cfg.hyperbolic_contracted_params.c = std::stod(value);
        } else if (key == "contracted_speed_lb") {
            cfg.contracted_speed_lb = std::stod(value);
        } else if (key == "contracted_speed_ub") {
            cfg.contracted_speed_ub = std::stod(value);
        } else if (key == "elliptic_a") {
            cfg.elliptic_params.a = std::stod(value);
        } else if (key == "elliptic_b") {
            cfg.elliptic_params.b = std::stod(value);
        } else {
            throw std::runtime_error("unknown config key: " + key);
        }
    }

    if (cfg.map.ub[0] == 0.0 && cfg.map.lb[0] == 0.0) {
        cfg.map = make_rect(cfg.state_lb, cfg.state_ub);
    }
    return cfg;
}

}  // namespace gsc
