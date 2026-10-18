#include "robometrics/evaluation.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace py = pybind11;

namespace {
using robometrics::DynamicsLimits;
using robometrics::EsdfView;
using robometrics::Evaluator;
using robometrics::Matrix;
using robometrics::Model;
using robometrics::Pose;
using robometrics::StateMetrics;
using robometrics::TrajectoryMetrics;
using robometrics::Vec3;
using Floats = py::array_t<float, py::array::c_style | py::array::forcecast>;
using Ints = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;

void require_shape(const py::array& array,
                   std::initializer_list<py::ssize_t> dimensions,
                   const char* name) {
    bool matches = array.ndim() == static_cast<py::ssize_t>(dimensions.size());
    if (matches) {
        std::size_t axis = 0;
        for (auto size : dimensions) {
            if (array.shape(axis++) != size) {
                matches = false;
                break;
            }
        }
    }
    if (!matches) {
        throw py::value_error(std::string(name) + " shape mismatch");
    }
}

template<class Record, class Array>
std::vector<Record> read_records(const py::dict& fields, const char* name,
                                std::initializer_list<py::ssize_t> tail) {
    auto array = py::cast<Array>(fields[name]);
    if (array.ndim() != static_cast<py::ssize_t>(tail.size() + 1)) {
        throw py::value_error(std::string(name) + " rank mismatch");
    }
    std::size_t axis = 1;
    for (auto size : tail) {
        if (array.shape(axis++) != size) {
            throw py::value_error(std::string(name) + " shape mismatch");
        }
    }
    std::vector<Record> output(array.shape(0));
    if (array.nbytes()) {
        std::memcpy(output.data(), array.data(), array.nbytes());
    }
    return output;
}

template<std::size_t N>
std::array<float, N> read_fixed(const py::dict& fields, const char* name,
                               std::initializer_list<py::ssize_t> dimensions) {
    auto array = py::cast<Floats>(fields[name]);
    require_shape(array, dimensions, name);
    if (array.size() != N) {
        throw py::value_error(std::string(name) + " size mismatch");
    }
    std::array<float, N> output;
    std::copy_n(array.data(), N, output.begin());
    return output;
}

Model read_model(const py::dict& fields) {
    Model model;
    model.base = read_fixed<16>(fields, "base", {4, 4});
    model.ee_offset = read_fixed<16>(fields, "ee_offset", {4, 4});
    model.origins = read_records<Matrix, Floats>(fields, "origins", {4, 4});
    model.parents = read_records<std::int32_t, Ints>(fields, "parents", {});
    model.q_indices = read_records<std::int32_t, Ints>(fields, "q_indices", {});
    model.axes = read_records<Vec3, Floats>(fields, "axes", {3});
    model.sphere_centers = read_records<Vec3, Floats>(fields, "sphere_centers", {3});
    model.sphere_links = read_records<std::int32_t, Ints>(fields, "sphere_links", {});
    model.radii = read_records<float, Floats>(fields, "radii", {});
    model.padding = read_records<float, Floats>(fields, "padding", {});
    model.pairs = read_records<std::array<std::int32_t, 2>, Ints>(fields, "pairs", {2});
    model.lower = read_records<float, Floats>(fields, "lower", {});
    model.upper = read_records<float, Floats>(fields, "upper", {});
    model.velocity = read_records<float, Floats>(fields, "velocity", {});
    model.ee_link = py::cast<std::int32_t>(fields["ee_link"]);
    model.source_fingerprint = py::cast<std::string>(fields["source_fingerprint"]);
    return model;
}

Pose read_goal(const Floats& array) {
    require_shape(array, {7}, "goal");
    Pose goal;
    std::copy_n(array.data(), goal.size(), goal.begin());
    return goal;
}

// Keep the Python grid owner alive while the core borrows its data pointer.
struct MapInput {
    Floats grid;
    EsdfView view;

    explicit MapInput(const py::dict& fields)
        : grid(py::cast<Floats>(fields["grid"])) {
        if (grid.ndim() != 3) {
            throw py::value_error("grid must be rank three");
        }
        view.grid = grid.data();
        for (int axis = 0; axis < 3; ++axis) {
            view.shape[axis] = grid.shape(axis);
        }
        view.origin = read_fixed<3>(fields, "origin", {3});
        view.lower = read_fixed<3>(fields, "lower_bound", {3});
        view.upper = read_fixed<3>(fields, "upper_bound", {3});
        view.voxel = py::cast<float>(fields["voxel_size"]);
        view.validate();
    }
};

py::array_t<float> pose_array(const std::vector<Pose>& poses) {
    py::array_t<float> output({static_cast<py::ssize_t>(poses.size()), py::ssize_t(7)});
    for (std::size_t index = 0; index < poses.size(); ++index) {
        std::copy_n(poses[index].data(), 7, output.mutable_data() + index * 7);
    }
    return output;
}

template<class Value>
void add_state_field(py::dict& output, const char* name,
                     const std::vector<StateMetrics>& states,
                     Value StateMetrics::*member) {
    py::array_t<Value> array(states.size());
    for (std::size_t index = 0; index < states.size(); ++index) {
        array.mutable_data()[index] = states[index].*member;
    }
    output[name] = std::move(array);
}

py::dict state_output(const std::vector<StateMetrics>& states) {
    py::dict output;
    py::array_t<float> ee({static_cast<py::ssize_t>(states.size()), py::ssize_t(7)});
    for (std::size_t index = 0; index < states.size(); ++index) {
        std::copy_n(states[index].ee.data(), 7, ee.mutable_data() + index * 7);
    }
    output["ee"] = std::move(ee);
    add_state_field(output, "min_clearance", states, &StateMetrics::min_clearance);
    add_state_field(output, "min_self_clearance", states, &StateMetrics::min_self_clearance);
    add_state_field(output, "max_joint_violation", states, &StateMetrics::max_joint_violation);
    add_state_field(output, "normalized_joint_violation", states, &StateMetrics::normalized_joint_violation);
    add_state_field(output, "position_error", states, &StateMetrics::position_error);
    add_state_field(output, "orientation_error", states, &StateMetrics::orientation_error);
    add_state_field(output, "oob_count", states, &StateMetrics::oob_count);
    add_state_field(output, "valid_sphere_count", states, &StateMetrics::valid_sphere_count);
    return output;
}

py::dict trajectory_output(const TrajectoryMetrics& metrics) {
    py::dict output;
    output["ee"] = pose_array(metrics.ee);
    output["position_error"] = metrics.position_error;
    output["orientation_error"] = metrics.orientation_error;
    output["min_clearance"] = metrics.min_clearance;
    output["min_self_clearance"] = metrics.min_self_clearance;
    output["oob_count"] = metrics.oob_count;
    output["max_joint_violation"] = metrics.max_joint_violation;
    output["max_velocity_violation"] = metrics.max_velocity_violation;
    output["max_acceleration_violation"] = metrics.max_acceleration_violation;
    output["max_abs_jerk"] = metrics.max_abs_jerk;
    return output;
}

class NativeEvaluator {
    Evaluator core_;
    std::mutex mutex_;

    template<class Function>
    auto with_workspace(Function&& function) {
        py::gil_scoped_release release;
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock()) {
            throw std::runtime_error("evaluator instance does not support concurrent calls");
        }
        return std::forward<Function>(function)();
    }
public:
    explicit NativeEvaluator(const py::dict& fields) : core_(read_model(fields)) {}

    py::dict states(Floats q, py::object goal_object, py::object map_object) {
        const auto dof = static_cast<py::ssize_t>(core_.model().lower.size());
        if (q.ndim() != 2 || q.shape(1) != dof) {
            throw py::value_error("q must have shape (N, D)");
        }
        std::optional<Pose> goal;
        if (!goal_object.is_none()) {
            goal = read_goal(py::cast<Floats>(goal_object));
        }
        std::optional<MapInput> map;
        if (!map_object.is_none()) {
            map.emplace(py::cast<py::dict>(map_object));
        }
        const auto* values = q.data();
        const auto count = static_cast<std::size_t>(q.shape(0));
        auto result = with_workspace([&] {
            return core_.states(values, count, goal ? &*goal : nullptr,
                                map ? &map->view : nullptr);
        });
        return state_output(result);
    }

    py::dict trajectory(Floats q, Floats dq, Floats ddq, Floats jerk,
                        Floats goal_array, py::dict map_fields,
                        float acceleration, float jerk_limit) {
        const auto dof = static_cast<py::ssize_t>(core_.model().lower.size());
        if (q.ndim() != 2 || q.shape(1) != dof || q.shape(0) == 0) {
            throw py::value_error("q must have nonempty shape (T, D)");
        }
        for (const auto* array : {&dq, &ddq, &jerk}) {
            require_shape(*array, {q.shape(0), dof}, "derivatives");
        }
        const Pose goal = read_goal(goal_array);
        const MapInput map(map_fields);
        const DynamicsLimits limits{acceleration, jerk_limit};
        const auto count = static_cast<std::size_t>(q.shape(0));
        const float *positions = q.data(), *velocities = dq.data();
        const float *accelerations = ddq.data(), *jerks = jerk.data();
        auto result = with_workspace([&] {
            return core_.trajectory(positions, velocities, accelerations, jerks,
                                    count, goal, map.view, limits);
        });
        return trajectory_output(result);
    }


};
} // namespace

PYBIND11_MODULE(_evaluation_cpp, module) {
    module.attr("schema_version") = 1;
    module.attr("compiler") = EVALUATION_COMPILER;
    module.attr("build_type") = EVALUATION_BUILD_TYPE;
    module.def("validate_model", [](const py::dict& fields) {
        read_model(fields).validate();
    });
    py::class_<NativeEvaluator>(module, "Evaluator")
        .def(py::init<const py::dict&>())
        .def("states", &NativeEvaluator::states, py::arg("q"),
             py::arg("goal") = py::none(), py::arg("esdf") = py::none())
        .def("trajectory", &NativeEvaluator::trajectory);
}
