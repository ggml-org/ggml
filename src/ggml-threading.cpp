#include "ggml-threading.h"
#include <exception>
#include <mutex>

std::mutex ggml_critical_section_mutex;

void ggml_critical_section_start() {
    ggml_critical_section_mutex.lock();
}

void ggml_critical_section_end(void) {
    ggml_critical_section_mutex.unlock();
}

GGML_NORETURN static void ggml_uncaught_exception() {
    if (const std::exception_ptr e{std::current_exception()}) {
        try {
            std::rethrow_exception(e);
        } catch (const std::exception & ex) {
            ggml_abort("set_terminate", 0, "uncaught exception %s", ex.what());
        } catch (...) {
            ggml_abort("set_terminate", 0, "unknown exception");
        }
    }
    ggml_abort("set_terminate", 0, "std::terminate called");
}

void ggml_uncaught_exception_init() {
    std::set_terminate(ggml_uncaught_exception);
}
