#include <nanobind/nanobind.h>
#include <cmath>
#include <string>

#include <mlx/array.h>

#include "nanobind/na_composed.h"
#include "nanobind/na_split_backward.h"
#include "nanobind/na_split_forward.h"
#include "nanobind/na1d_primitive.h"
#include "nanobind/na2d_primitive.h"
#include "nanobind/na3d_primitive.h"

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

    {
        namespace mx = mlx::core;
        using namespace nb::literals;
        m.def("_na2d_v2_forward",
            [](const mx::array& q,
               const mx::array& k,
               const mx::array& v,
               int kernel_size,
               int stride_h, int stride_w,
               int dilation_h, int dilation_w,
               bool causal_h, bool causal_w,
               float scale) {
                return natten_mlx::na2d_fused_forward_v2(
                    q, k, v, kernel_size,
                    stride_h, stride_w, dilation_h, dilation_w,
                    causal_h, causal_w, scale);
            },
            "q"_a, "k"_a, "v"_a,
            "kernel_size"_a,
            "stride_h"_a, "stride_w"_a,
            "dilation_h"_a, "dilation_w"_a,
            "causal_h"_a, "causal_w"_a,
            "scale"_a);

        m.def("_na1d_v2_forward",
            [](const mx::array& q,
               const mx::array& k,
               const mx::array& v,
               int kernel_size,
               int stride, int dilation,
               bool causal, float scale) {
                return natten_mlx::na1d_fused_forward_v2(
                    q, k, v, kernel_size,
                    stride, dilation, causal, scale);
            },
            "q"_a, "k"_a, "v"_a,
            "kernel_size"_a,
            "stride"_a, "dilation"_a,
            "causal"_a, "scale"_a);

        m.def("_na3d_v2_forward",
            [](const mx::array& q,
               const mx::array& k,
               const mx::array& v,
               int kernel_size,
               int stride_d, int stride_h, int stride_w,
               int dilation_d, int dilation_h, int dilation_w,
               bool causal_d, bool causal_h, bool causal_w,
               float scale) {
                return natten_mlx::na3d_fused_forward_v2(
                    q, k, v, kernel_size,
                    stride_d, stride_h, stride_w,
                    dilation_d, dilation_h, dilation_w,
                    causal_d, causal_h, causal_w, scale);
            },
            "q"_a, "k"_a, "v"_a,
            "kernel_size"_a,
            "stride_d"_a, "stride_h"_a, "stride_w"_a,
            "dilation_d"_a, "dilation_h"_a, "dilation_w"_a,
            "causal_d"_a, "causal_h"_a, "causal_w"_a,
            "scale"_a);
    }
}
