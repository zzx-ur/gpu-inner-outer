#pragma once

#include <cmath>

#include "gsc/grid.hpp"

namespace gsc {

struct Unicycle4DModel {
    double sample_time = 0.2;
    std::uint32_t integration_substeps = 32;
    double disturbance_half_width[kStateDim]{0.05, 0.05, 0.02, 0.02};

    GSC_HD static double wrap_angle(double x) {
        constexpr double kPi = 3.14159265358979323846;
        constexpr double kTwoPi = 2.0 * kPi;
        while (x < -kPi) {
            x += kTwoPi;
        }
        while (x >= kPi) {
            x -= kTwoPi;
        }
        return x;
    }

    GSC_HD void nominal_step(const double x[kStateDim],
                             const double u[kInputDim],
                             double out[kStateDim]) const {
        out[0] = x[0];
        out[1] = x[1];
        out[2] = x[2];
        out[3] = x[3];

        const double dt = sample_time / static_cast<double>(integration_substeps);
        for (std::uint32_t step = 0; step < integration_substeps; ++step) {
            const double dx = out[3] * std::cos(out[2]);
            const double dy = out[3] * std::sin(out[2]);
            const double dtheta = u[1];
            const double dv = u[0];

            out[0] += dt * dx;
            out[1] += dt * dy;
            out[2] = wrap_angle(out[2] + dt * dtheta);
            out[3] += dt * dv;
        }
    }

    // Jacobian of f with respect to state x: Jf_x(u)
    // For unicycle 4D model: x_next = [x + T*v*cos(theta), y + T*v*sin(theta), theta + T*omega, v + T*a]
    // Jf_x = [1, 0, T*|v|, T]
    //        [0, 1, T*|v|, T]
    //        [0, 0, 1,     0]
    //        [0, 0, 0,     1]
    // We compute |Jf_x| * (cell_eta / 2) element-wise for axis-aligned over-approximation
    GSC_HD void jacobian_x_times_half_eta(const double input_center[kInputDim],
                                          const double state_center[kStateDim],
                                          const double cell_eta[kStateDim],
                                          double result[kStateDim]) const {
        const double half_eta[kStateDim] = {
            0.5 * cell_eta[0],
            0.5 * cell_eta[1],
            0.5 * cell_eta[2],
            0.5 * cell_eta[3]
        };
        
        // |Jf_x| * half_eta (row-wise absolute value multiplication)
        // Row 0: |1|*half_eta[0] + |0|*half_eta[1] + |T*v|*half_eta[2] + |T|*half_eta[3]
        // Row 1: |0|*half_eta[0] + |1|*half_eta[1] + |T*v|*half_eta[2] + |T|*half_eta[3]
        // Row 2: |0|*half_eta[0] + |0|*half_eta[1] + |1|*half_eta[2] + |0|*half_eta[3]
        // Row 3: |0|*half_eta[0] + |0|*half_eta[1] + |0|*half_eta[2] + |1|*half_eta[3]
        const double abs_v = std::fabs(state_center[3]);
        result[0] = half_eta[0] + sample_time * abs_v * half_eta[2] + sample_time * half_eta[3];
        result[1] = half_eta[1] + sample_time * abs_v * half_eta[2] + sample_time * half_eta[3];
        result[2] = half_eta[2];
        result[3] = half_eta[3];
    }

    // Jacobian of f with respect to disturbance w: Jf_w(u)
    // Jf_w = [T, 0, 0, 0]
    //        [0, T, 0, 0]
    //        [0, 0, T, 0]
    //        [0, 0, 0, T]
    // We compute |Jf_w| * (disturbance_range / 2) = T * disturbance_half_width
    GSC_HD void jacobian_w_times_half_disturbance(double result[kStateDim]) const {
        result[0] = sample_time * disturbance_half_width[0];
        result[1] = sample_time * disturbance_half_width[1];
        result[2] = sample_time * disturbance_half_width[2];
        result[3] = sample_time * disturbance_half_width[3];
    }

    GSC_HD void successor_box(const double state_center[kStateDim],
                              const double input_center[kInputDim],
                              const double cell_eta[kStateDim],
                              double min_coord[kStateDim],
                              double max_coord[kStateDim]) const {
        // Compute nominal successor (with disturbance center = 0)
        double nominal[kStateDim];
        nominal_step(state_center, input_center, nominal);

        // Compute over-approximation radius using Jacobian:
        // d_x_succ = |Jf_x(u)| * (cell_eta / 2) + |Jf_w(u)| * (disturbance_range / 2)
        double jfx_term[kStateDim];
        double jfw_term[kStateDim];
        jacobian_x_times_half_eta(input_center, state_center, cell_eta, jfx_term);
        jacobian_w_times_half_disturbance(jfw_term);

        for (int d = 0; d < kStateDim; ++d) {
            const double radius = jfx_term[d] + jfw_term[d];
            min_coord[d] = nominal[d] - radius;
            max_coord[d] = nominal[d] + radius;
        }
    }
};

}  // namespace gsc
