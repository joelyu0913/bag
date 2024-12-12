#define __STDC_FORMAT_MACROS

#include "yang/util/term_handler.h"

#include <signal.h>
#include <string.h>
#include <unistd.h>

#include <cstdio>
#include <exception>
#include <iostream>

#include "yang/util/backtrace.h"
#include "yang/util/logging.h"

namespace yang {

static void DumpException() {
  static bool thrown = false;

  try {
    // try once to re-throw currently active exception
    if (!thrown) {
      thrown = true;
      throw;
    }
  } catch (const std::exception &e) {
    LOG_ERROR("Caught unhandled exception. what(): {}", e.what());
  } catch (...) {
    LOG_ERROR("Caught unknown/unhandled exception.");
  }

  // backtrace will be dumped in signal handler
  std::abort();
}

static void DumpOnSignal(int p) {
  LOG_ERROR("Caught signal {}", strsignal(p));
  LOG_ERROR("Backtrace: \n{}", fmt::join(Backtrace(128), "\n"));

  // call the default handler
  raise(p);
}

void SetTerminateHandler() {
  struct sigaction act;
  memset(&act, 0, sizeof(act));
  act.sa_handler = DumpOnSignal;
  // Restore the signal action to the default upon entry to the signal handler
  act.sa_flags = SA_RESETHAND;
  sigaction(SIGABRT, &act, nullptr);
  sigaction(SIGSEGV, &act, nullptr);

  std::set_terminate(DumpException);
}

}  // namespace yang
