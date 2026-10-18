// Single-thread analytic ESDF. Python precomputes reference rotations; this
// translation unit packs constants and fuses world-to-local, SDF and reduction.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using Vec3 = std::array<double, 3>;
using Axis = py::array_t<float, py::array::c_style | py::array::forcecast>;
using Clock = std::chrono::steady_clock;
enum class Kind { Box, Sphere, Capsule, Cylinder };

struct Obstacle {
    Kind kind;
    Vec3 translation{}, half_size{}, base{}, segment{};
    std::array<double, 9> rotation{};
    double radius = 0, half_height = 0, segment_squared = 0;
};

double norm(double x, double y, double z) {
    return std::sqrt(x*x + y*y + z*z);
}

template <std::size_t N>
std::array<double, N> finite_array(py::handle value, const char* name) {
    auto result = py::cast<std::array<double, N>>(value);
    for (double v : result)
        if (!std::isfinite(v)) throw py::value_error(std::string(name) + " must be finite");
    return result;
}

double positive(py::handle value, const char* name) {
    double result = py::cast<double>(value);
    if (!std::isfinite(result) || result <= 0)
        throw py::value_error(std::string(name) + " must be finite and positive");
    return result;
}

std::vector<Obstacle> pack(const py::list& records) {
    std::vector<Obstacle> obstacles;
    obstacles.reserve(records.size());
    for (auto item : records) {
        auto record = py::cast<py::dict>(item);
        Obstacle obstacle;
        auto pose = finite_array<7>(record["pose"], "pose");
        std::copy_n(pose.begin(), 3, obstacle.translation.begin());
        obstacle.rotation = finite_array<9>(record["rotation"], "rotation");
        auto kind = py::cast<std::string>(record["kind"]);
        if (kind == "cuboid") {
            obstacle.kind = Kind::Box;
            auto dims = finite_array<3>(record["dims"], "dims");
            for (int i = 0; i < 3; ++i) {
                if (dims[i] <= 0) throw py::value_error("dims must be positive");
                obstacle.half_size[i] = dims[i] / 2.0;
            }
        } else {
            obstacle.radius = positive(record["radius"], "radius");
            if (kind == "sphere") obstacle.kind = Kind::Sphere;
            else if (kind == "cylinder") {
                obstacle.kind = Kind::Cylinder;
                obstacle.half_height = positive(record["height"], "height") / 2.0;
            } else if (kind == "capsule") {
                obstacle.kind = Kind::Capsule;
                obstacle.base = finite_array<3>(record["base"], "base");
                auto tip = finite_array<3>(record["tip"], "tip");
                for (int i = 0; i < 3; ++i) obstacle.segment[i] = tip[i] - obstacle.base[i];
                const auto& s = obstacle.segment;
                obstacle.segment_squared = s[0]*s[0] + s[1]*s[1] + s[2]*s[2];
                if (!std::isfinite(obstacle.segment_squared) || obstacle.segment_squared <= 0)
                    throw py::value_error("capsule segment must have finite positive squared length");
            } else throw py::value_error("unknown primitive: " + kind);
        }
        obstacles.push_back(obstacle);
    }
    return obstacles;
}

double distance(const Obstacle& o, double wx, double wy, double wz) {
    double dx = wx - o.translation[0], dy = wy - o.translation[1], dz = wz - o.translation[2];
    const auto& r = o.rotation;  // Row-major R; local = (world - translation) @ R.
    double x = dx*r[0] + dy*r[3] + dz*r[6];
    double y = dx*r[1] + dy*r[4] + dz*r[7];
    double z = dx*r[2] + dy*r[5] + dz*r[8];
    if (o.kind == Kind::Box) {
        double qx = std::abs(x) - o.half_size[0];
        double qy = std::abs(y) - o.half_size[1];
        double qz = std::abs(z) - o.half_size[2];
        return norm(std::max(qx, 0.0), std::max(qy, 0.0), std::max(qz, 0.0))
               + std::min(std::max({qx, qy, qz}), 0.0);
    }
    if (o.kind == Kind::Sphere) return norm(x, y, z) - o.radius;
    if (o.kind == Kind::Capsule) {
        const auto& a = o.base;
        const auto& s = o.segment;
        double h = std::clamp(((x-a[0])*s[0] + (y-a[1])*s[1] + (z-a[2])*s[2])
                              / o.segment_squared, 0.0, 1.0);
        return norm(x-(a[0]+h*s[0]), y-(a[1]+h*s[1]), z-(a[2]+h*s[2])) - o.radius;
    }
    double radial = std::sqrt(x*x + y*y) - o.radius;
    double axial = std::abs(z) - o.half_height;
    double outside_r = std::max(radial, 0.0), outside_z = std::max(axial, 0.0);
    return std::sqrt(outside_r*outside_r + outside_z*outside_z)
           + std::min(std::max(radial, axial), 0.0);
}

py::array_t<float> build(const Axis& xs, const Axis& ys, const Axis& zs,
                         const py::list& records, py::object timings) {
    for (const Axis* axis : {&xs, &ys, &zs}) {
        if (axis->ndim() != 1 || axis->size() < 2)
            throw py::value_error("axes must be 1D with at least two centres");
        for (py::ssize_t i = 0; i < axis->size(); ++i)
            if (!std::isfinite(axis->data()[i])) throw py::value_error("axes must be finite");
    }
    const auto nx = xs.size(), ny = ys.size(), nz = zs.size();
    const auto limit = std::numeric_limits<py::ssize_t>::max() / sizeof(float);
    if (nx > limit / ny || nx*ny > limit / nz) throw std::overflow_error("grid size overflow");
    auto start = Clock::now();
    auto obstacles = pack(records);
    auto packed = Clock::now();
    py::array_t<float> output({nx, ny, nz});
    float* grid = output.mutable_data();
    const float* x_values = xs.data();
    const float* y_values = ys.data();
    const float* z_values = zs.data();
    auto compute_start = Clock::now();
    {
        py::gil_scoped_release release;
        py::ssize_t index = 0;
        for (py::ssize_t i = 0; i < nx; ++i)
            for (py::ssize_t j = 0; j < ny; ++j)
                for (py::ssize_t k = 0; k < nz; ++k) {
                    double minimum = 10.0;
                    for (const auto& obstacle : obstacles) {
                        double value = distance(obstacle, x_values[i], y_values[j], z_values[k]);
                        if (std::isnan(value)) throw std::invalid_argument("primitive distance is NaN");
                        minimum = std::min(minimum, value);
                    }
                    grid[index++] = static_cast<float>(minimum);
                }
    }
    auto computed = Clock::now();
    if (!timings.is_none()) {
        auto values = timings.cast<py::dict>();
        values["packing_s"] = std::chrono::duration<double>(packed-start).count();
        values["compute_s"] = std::chrono::duration<double>(computed-compute_start).count();
    }
    return output;
}

PYBIND11_MODULE(_esdf_cpp, m) {
    m.doc() = "Single-thread CPU analytic ESDF; float32 coordinates/output, double intermediates.";
    m.def("build", &build, py::arg("xs"), py::arg("ys"), py::arg("zs"),
          py::arg("obstacles"), py::arg("timings") = py::none());
    m.attr("threads") = 1;
    m.attr("implementation_version") = 1;
    m.attr("build_type") = ESDF_BUILD_TYPE;
    m.attr("compiler") = ESDF_COMPILER;
    m.attr("pybind11_version") = std::to_string(PYBIND11_VERSION_MAJOR) + "."
        + std::to_string(PYBIND11_VERSION_MINOR) + "." + std::to_string(PYBIND11_VERSION_PATCH);
}
