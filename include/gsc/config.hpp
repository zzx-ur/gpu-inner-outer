#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "gsc/types.hpp"

namespace gsc {

struct HyperRect4D {
    double lb[kStateDim]{};
    double ub[kStateDim]{};
};

// 状态约束类型枚举
enum class ConstraintType {
    kNone,           // 无约束（仅使用 map）
    kHyperbolic,     // 双曲线约束: x1^2 - x2^2 <= a, b*x2^2 - x1^2 <= c
    kElliptic,       // 椭圆约束: (x1/a)^2 + (x2/b)^2 <= 1
    kCustom          // 自定义约束函数
};

// 双曲线约束参数: x1^2 - x2^2 <= a 且 b*x2^2 - x1^2 <= c
struct HyperbolicConstraintParams {
    double a = 4.0;   // x1^2 - x2^2 <= a
    double b = 4.0;   // b*x2^2 - x1^2 <= c
    double c = 16.0;
};

// 椭圆约束参数: (x1/a)^2 + (x2/b)^2 <= 1
struct EllipticConstraintParams {
    double a = 2.0;   // x 轴半径
    double b = 2.0;   // y 轴半径
};

// 状态约束函数类型: 返回 true 表示状态满足约束
using StateConstraintFn = std::function<bool(const double x[kStateDim])>;

struct CaseConfig {
    std::string name = "unicycle_default";
    double state_lb[kStateDim]{-2.0, -2.0, -3.141592653589793, 0.0};
    double state_ub[kStateDim]{ 2.0,  2.0,  3.141592653589793, 2.0};
    double state_eta[kStateDim]{0.5, 0.5, 0.7853981633974483, 0.5};
    int wrap_dim = 2;

    double input_lb[kInputDim]{-0.5, -1.0};
    double input_ub[kInputDim]{ 0.5,  1.0};
    double input_eta[kInputDim]{0.5, 1.0};

    HyperRect4D candidate{};
    HyperRect4D map{};
    std::vector<HyperRect4D> obstacles;

    // 状态约束配置
    ConstraintType constraint_type = ConstraintType::kNone;
    HyperbolicConstraintParams hyperbolic_params;
    EllipticConstraintParams elliptic_params;
    StateConstraintFn custom_constraint;  // 自定义约束函数

    double disturbance_half_width[kStateDim]{0.05, 0.05, 0.02, 0.02};
    double sample_time = 0.2;
    std::uint32_t integration_substeps = 32;
    std::uint32_t max_iterations = 32;
    bool verbose = true;
};

// 根据配置创建状态约束函数
StateConstraintFn make_state_constraint(const CaseConfig& cfg);

// 检查点是否满足状态约束
bool satisfies_state_constraint(const CaseConfig& cfg, const double x[kStateDim]);

bool point_in_rect(const HyperRect4D& rect, const double x[kStateDim]);
CaseConfig make_default_case();
CaseConfig make_wraparound_case();
CaseConfig make_smoke_case();
CaseConfig make_hyperbolic_case();  // 新增：带双曲线约束的案例
CaseConfig load_case_config(const std::string& path);

}  // namespace gsc
