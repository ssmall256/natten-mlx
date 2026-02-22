#pragma once

#include <utility>

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace natten_mlx::nanobind_backend {

inline nb::module_ pure_module() {
    return nb::module_::import_("natten_mlx._core.pure");
}

inline nb::module_ mx_module() {
    return nb::module_::import_("mlx.core");
}

template <typename... Args>
inline nb::object call_pure(const char* name, Args&&... args) {
    return pure_module().attr(name)(std::forward<Args>(args)...);
}

}  // namespace natten_mlx::nanobind_backend
