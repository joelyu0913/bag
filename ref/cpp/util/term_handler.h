#pragma once

namespace yang {

// Handle SIGABRT, SIGSEGV and unhandled exceptions to print backtrace
void SetTerminateHandler();

}  // namespace yang
