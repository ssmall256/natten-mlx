#include <nanobind/nanobind.h>

#include <string>

namespace nb = nanobind;

namespace {

nb::object choose_backend() {
    try {
        nb::object fast = nb::module_::import_("natten_mlx._core.fast_metal");
        bool fast_available = nb::cast<bool>(fast.attr("is_available")());
        if (fast_available) {
            return fast;
        }
    } catch (const nb::python_error &) {
    }
    return nb::module_::import_("natten_mlx._core.pure");
}

nb::object dispatch_from_args(const char *name, const nb::args &args) {
    nb::object backend = choose_backend();
    nb::object fn = backend.attr(name);
    PyObject *result = PyObject_CallObject(fn.ptr(), args.ptr());
    if (result == nullptr) {
        throw nb::python_error();
    }
    return nb::steal<nb::object>(result);
}

void bind_passthrough(nb::module_ &m, const char *name, size_t arity) {
    m.def(name, [name, arity](const nb::args &args) {
        if (args.size() != arity) {
            std::string message = std::string(name) + " expected " + std::to_string(arity) +
                                  " arguments, got " + std::to_string(args.size());
            throw nb::type_error(message.c_str());
        }
        return dispatch_from_args(name, args);
    });
}

}  // namespace

NB_MODULE(_nanobind_ext, m) {
    m.doc() = "Native nanobind extension shim for natten-mlx backend dispatch.";

    bind_passthrough(m, "na1d_forward", 8);
    bind_passthrough(m, "na2d_forward", 8);
    bind_passthrough(m, "na1d_qk_forward", 7);
    bind_passthrough(m, "na1d_av_forward", 6);
    bind_passthrough(m, "na2d_qk_forward", 7);
    bind_passthrough(m, "na2d_av_forward", 6);

    bind_passthrough(m, "na1d_backward", 9);
    bind_passthrough(m, "na2d_backward", 9);
    bind_passthrough(m, "na1d_qk_backward", 8);
    bind_passthrough(m, "na1d_av_backward", 7);
    bind_passthrough(m, "na2d_qk_backward", 8);
    bind_passthrough(m, "na2d_av_backward", 7);
}
