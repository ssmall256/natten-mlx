#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <string>

#include "nanobind/metal_runtime.h"
#include "nanobind/na_composed.h"
#include "nanobind/na_split_backward.h"
#include "nanobind/na_split_forward.h"

namespace nb = nanobind;

namespace {

nb::object arg_at(const nb::args& args, size_t index) {
    return nb::borrow<nb::object>(args[index]);
}

template <typename Fn>
void bind_arity(nb::module_& m, const char* name, size_t arity, Fn&& fn) {
    m.def(name, [name, arity, fn = std::forward<Fn>(fn)](const nb::args& args) {
        if (args.size() != arity) {
            std::string message = std::string(name) + " expected " + std::to_string(arity) +
                                  " arguments, got " + std::to_string(args.size());
            throw nb::type_error(message.c_str());
        }
        return fn(args);
    });
}

}  // namespace

NB_MODULE(_nanobind_ext, m) {
    m.doc() = "Native nanobind extension entrypoint for natten-mlx tier-2 backend.";
    nb::module_::import_("mlx.core");

    bind_arity(m, "na1d_forward", 8, [](const nb::args& args) {
        return natten_mlx::nanobind_composed::na1d_forward(
            arg_at(args, 0),
            arg_at(args, 1),
            arg_at(args, 2),
            arg_at(args, 3),
            arg_at(args, 4),
            arg_at(args, 5),
            arg_at(args, 6),
            arg_at(args, 7));
    });
    bind_arity(m, "na2d_forward", 8, [](const nb::args& args) {
        return natten_mlx::nanobind_composed::na2d_forward(
            arg_at(args, 0),
            arg_at(args, 1),
            arg_at(args, 2),
            arg_at(args, 3),
            arg_at(args, 4),
            arg_at(args, 5),
            arg_at(args, 6),
            arg_at(args, 7));
    });
    bind_arity(m, "na3d_forward", 8, [](const nb::args& args) {
        return natten_mlx::nanobind_composed::na3d_forward(
            arg_at(args, 0),
            arg_at(args, 1),
            arg_at(args, 2),
            arg_at(args, 3),
            arg_at(args, 4),
            arg_at(args, 5),
            arg_at(args, 6),
            arg_at(args, 7));
    });

    bind_arity(m, "na1d_qk_forward", 7, [](const nb::args& args) {
        return natten_mlx::nanobind_split_forward::na1d_qk_forward(
            arg_at(args, 0),
            arg_at(args, 1),
            arg_at(args, 2),
            arg_at(args, 3),
            arg_at(args, 4),
            arg_at(args, 5),
            arg_at(args, 6));
    });
    bind_arity(m, "na1d_av_forward", 6, [](const nb::args& args) {
        return natten_mlx::nanobind_split_forward::na1d_av_forward(
            arg_at(args, 0), arg_at(args, 1), arg_at(args, 2), arg_at(args, 3), arg_at(args, 4), arg_at(args, 5));
    });
    bind_arity(m, "na2d_qk_forward", 7, [](const nb::args& args) {
        return natten_mlx::nanobind_split_forward::na2d_qk_forward(
            arg_at(args, 0),
            arg_at(args, 1),
            arg_at(args, 2),
            arg_at(args, 3),
            arg_at(args, 4),
            arg_at(args, 5),
            arg_at(args, 6));
    });
    bind_arity(m, "na2d_av_forward", 6, [](const nb::args& args) {
        return natten_mlx::nanobind_split_forward::na2d_av_forward(
            arg_at(args, 0), arg_at(args, 1), arg_at(args, 2), arg_at(args, 3), arg_at(args, 4), arg_at(args, 5));
    });
    bind_arity(m, "na3d_qk_forward", 7, [](const nb::args& args) {
        return natten_mlx::nanobind_split_forward::na3d_qk_forward(
            arg_at(args, 0),
            arg_at(args, 1),
            arg_at(args, 2),
            arg_at(args, 3),
            arg_at(args, 4),
            arg_at(args, 5),
            arg_at(args, 6));
    });
    bind_arity(m, "na3d_av_forward", 6, [](const nb::args& args) {
        return natten_mlx::nanobind_split_forward::na3d_av_forward(
            arg_at(args, 0), arg_at(args, 1), arg_at(args, 2), arg_at(args, 3), arg_at(args, 4), arg_at(args, 5));
    });

    bind_arity(m, "na1d_backward", 9, [](const nb::args& args) {
        return natten_mlx::nanobind_composed::na1d_backward(
            arg_at(args, 0),
            arg_at(args, 1),
            arg_at(args, 2),
            arg_at(args, 3),
            arg_at(args, 4),
            arg_at(args, 5),
            arg_at(args, 6),
            arg_at(args, 7),
            arg_at(args, 8));
    });
    bind_arity(m, "na2d_backward", 9, [](const nb::args& args) {
        return natten_mlx::nanobind_composed::na2d_backward(
            arg_at(args, 0),
            arg_at(args, 1),
            arg_at(args, 2),
            arg_at(args, 3),
            arg_at(args, 4),
            arg_at(args, 5),
            arg_at(args, 6),
            arg_at(args, 7),
            arg_at(args, 8));
    });
    bind_arity(m, "na3d_backward", 9, [](const nb::args& args) {
        return natten_mlx::nanobind_composed::na3d_backward(
            arg_at(args, 0),
            arg_at(args, 1),
            arg_at(args, 2),
            arg_at(args, 3),
            arg_at(args, 4),
            arg_at(args, 5),
            arg_at(args, 6),
            arg_at(args, 7),
            arg_at(args, 8));
    });

    bind_arity(m, "na1d_qk_backward", 8, [](const nb::args& args) {
        return natten_mlx::nanobind_split_backward::na1d_qk_backward(
            arg_at(args, 0),
            arg_at(args, 1),
            arg_at(args, 2),
            arg_at(args, 3),
            arg_at(args, 4),
            arg_at(args, 5),
            arg_at(args, 6),
            arg_at(args, 7));
    });
    bind_arity(m, "na1d_av_backward", 7, [](const nb::args& args) {
        return natten_mlx::nanobind_split_backward::na1d_av_backward(
            arg_at(args, 0),
            arg_at(args, 1),
            arg_at(args, 2),
            arg_at(args, 3),
            arg_at(args, 4),
            arg_at(args, 5),
            arg_at(args, 6));
    });
    bind_arity(m, "na2d_qk_backward", 8, [](const nb::args& args) {
        return natten_mlx::nanobind_split_backward::na2d_qk_backward(
            arg_at(args, 0),
            arg_at(args, 1),
            arg_at(args, 2),
            arg_at(args, 3),
            arg_at(args, 4),
            arg_at(args, 5),
            arg_at(args, 6),
            arg_at(args, 7));
    });
    bind_arity(m, "na2d_av_backward", 7, [](const nb::args& args) {
        return natten_mlx::nanobind_split_backward::na2d_av_backward(
            arg_at(args, 0),
            arg_at(args, 1),
            arg_at(args, 2),
            arg_at(args, 3),
            arg_at(args, 4),
            arg_at(args, 5),
            arg_at(args, 6));
    });
    bind_arity(m, "na3d_qk_backward", 8, [](const nb::args& args) {
        return natten_mlx::nanobind_split_backward::na3d_qk_backward(
            arg_at(args, 0),
            arg_at(args, 1),
            arg_at(args, 2),
            arg_at(args, 3),
            arg_at(args, 4),
            arg_at(args, 5),
            arg_at(args, 6),
            arg_at(args, 7));
    });
    bind_arity(m, "na3d_av_backward", 7, [](const nb::args& args) {
        return natten_mlx::nanobind_split_backward::na3d_av_backward(
            arg_at(args, 0),
            arg_at(args, 1),
            arg_at(args, 2),
            arg_at(args, 3),
            arg_at(args, 4),
            arg_at(args, 5),
            arg_at(args, 6));
    });

    m.def("_debug_get_last_route", [](const std::string& op) {
        return natten_mlx::nanobind_metal_runtime::debug_get_last_route(op);
    });
    m.def("_debug_clear_last_routes", []() {
        natten_mlx::nanobind_metal_runtime::debug_clear_last_routes();
    });
    m.def("_debug_get_last_kernel", [](const std::string& op) {
        return natten_mlx::nanobind_metal_runtime::debug_get_last_kernel(op);
    });
    m.def("_debug_clear_last_kernels", []() {
        natten_mlx::nanobind_metal_runtime::debug_clear_last_kernels();
    });
    m.def("_debug_force_fused_failure", [](bool enabled) {
        natten_mlx::nanobind_metal_runtime::debug_force_fused_failure(enabled);
    });
    m.def("_debug_force_split_failure", [](bool enabled) {
        natten_mlx::nanobind_metal_runtime::debug_force_split_failure(enabled);
    });
    m.def("_debug_get_python_bridge_calls", []() {
        return natten_mlx::nanobind_metal_runtime::debug_get_python_bridge_calls();
    });
    m.def("_debug_clear_python_bridge_calls", []() {
        natten_mlx::nanobind_metal_runtime::debug_clear_python_bridge_calls();
    });
    m.def("_debug_get_launch_metrics", []() {
        return natten_mlx::nanobind_metal_runtime::debug_get_launch_metrics();
    });
    m.def("_debug_clear_launch_metrics", []() {
        natten_mlx::nanobind_metal_runtime::debug_clear_launch_metrics();
    });
}
