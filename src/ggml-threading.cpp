#include "ggml-threading.h"

#include <cstdlib>
#include <exception>
#include <mutex>

std::mutex ggml_critical_section_mutex;

void ggml_critical_section_start() {
    ggml_critical_section_mutex.lock();
}

void ggml_critical_section_end(void) {
    ggml_critical_section_mutex.unlock();
}

static std::terminate_handler previous_terminate_handler;

GGML_NORETURN static void ggml_uncaught_exception() {
    ggml_print_backtrace();
    if (previous_terminate_handler) {
        previous_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

void ggml_uncaught_exception_init() {
    const auto prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    previous_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
}
