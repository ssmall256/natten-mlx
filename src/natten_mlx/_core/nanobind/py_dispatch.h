#pragma once

#include <cstdlib>
#include <utility>

#include <nanobind/nanobind.h>

#include "metal_runtime.h"

namespace nb = nanobind;

namespace natten_mlx::nanobind_backend {

inline nb::module_ backend_module() {
    return nb::module_::import_("natten_mlx._core._nanobind_metal");
}

inline nb::module_ pure_module() {
    return nb::module_::import_("natten_mlx._core.pure");
}

inline nb::module_ mx_module() {
    return nb::module_::import_("mlx.core");
}

inline nb::module_ metal_sources_module() {
    return nb::module_::import_("natten_mlx._core._metal_sources");
}

inline bool use_native_runtime() {
    static int mode = []() {
        const char* v = std::getenv("NATTEN_NANOBIND_NATIVE_RUNTIME");
        if (v == nullptr) {
            return 1;
        }
        return (v[0] == '0') ? 0 : 1;
    }();
    return mode == 1;
}

template <typename... Args>
inline nb::object call_backend(const char* name, Args&&... args) {
    natten_mlx::nanobind_metal_runtime::debug_inc_python_bridge_calls();
    return backend_module().attr(name)(std::forward<Args>(args)...);
}

template <typename... Args>
inline nb::object call_pure(const char* name, Args&&... args) {
    return pure_module().attr(name)(std::forward<Args>(args)...);
}

}  // namespace natten_mlx::nanobind_backend
