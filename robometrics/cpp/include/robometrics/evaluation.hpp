#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace robometrics {
using Vec3 = std::array<float, 3>;
using Matrix = std::array<float, 16>;
using Pose = std::array<float, 7>;
constexpr float infinity = std::numeric_limits<float>::infinity();

struct Model {
    Matrix base{}, ee_offset{};
    std::vector<Matrix> origins;
    std::vector<std::int32_t> parents, q_indices, sphere_links;
    std::vector<Vec3> axes, sphere_centers;
    std::vector<float> radii, padding, lower, upper, velocity;
    std::vector<std::array<std::int32_t, 2>> pairs;
    std::int32_t ee_link = 0;
    std::string source_fingerprint;
    void validate() const;
};

// Borrowed map storage; the caller owns and validates its finite grid snapshot.
struct EsdfView {
    const float* grid = nullptr;
    std::array<std::size_t, 3> shape{};
    Vec3 origin{}, lower{}, upper{};
    float voxel = 0;
    void validate() const;
    float query(const Vec3& point) const;
    bool valid(const Vec3& point) const;
};
struct StateMetrics {
    Pose ee{};
    float min_clearance = infinity, min_self_clearance = infinity;
    float max_joint_violation = 0, normalized_joint_violation = 0;
    float position_error = 0, orientation_error = 0;
    std::int64_t oob_count = 0, valid_sphere_count = 0;
};

struct DynamicsLimits { float acceleration = 15, jerk = 500; };
struct TrajectoryMetrics {
    std::vector<Pose> ee;
    float position_error = 0, orientation_error = 0;
    float min_clearance = infinity, min_self_clearance = infinity;
    std::int64_t oob_count = 0;
    float max_joint_violation = 0, max_velocity_violation = 0;
    float max_acceleration_violation = 0, max_abs_jerk = 0;
};

// Single-thread instance workspace; use separate instances for concurrent calls.
class Evaluator {
public:
    explicit Evaluator(Model model);
    const Model& model() const { return model_; }
    std::vector<StateMetrics> states(const float* q, std::size_t count,
                                    const Pose* goal, const EsdfView* esdf);
    TrajectoryMetrics trajectory(const float* q, const float* dq, const float* ddq,
                                 const float* jerk, std::size_t count, const Pose& goal,
                                 const EsdfView& esdf, DynamicsLimits limits);
private:
    Model model_;
    std::vector<Matrix> poses_;
    std::vector<Vec3> spheres_;
    StateMetrics state(const float* q, const Pose* goal, const EsdfView* esdf);
};
} // namespace robometrics
