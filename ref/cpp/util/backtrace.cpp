#define __STDC_FORMAT_MACROS

#include "yang/util/backtrace.h"

#include <cxxabi.h>
#include <inttypes.h>

#include <cstdio>

#define UNW_LOCAL_ONLY
#include <libunwind.h>

namespace yang {

std::vector<std::string> Backtrace(int size) {
  std::vector<std::string> ret;

  unw_cursor_t cursor;
  unw_context_t context;

  unw_getcontext(&context);
  unw_init_local(&cursor, &context);

  int n = 0;
  while (n < size && unw_step(&cursor)) {
    unw_word_t ip, sp, off;

    unw_get_reg(&cursor, UNW_REG_IP, &ip);
    unw_get_reg(&cursor, UNW_REG_SP, &sp);

    char symbol[128] = {"<unknown>"};
    char *name = symbol;

    if (!unw_get_proc_name(&cursor, symbol, sizeof(symbol), &off)) {
      int status;
      if ((name = abi::__cxa_demangle(symbol, NULL, NULL, &status)) == 0) name = symbol;
    }

    char buf[256];
    snprintf(buf, sizeof(buf), "#%-2d 0x%016" PRIxPTR " sp=0x%016" PRIxPTR " %s +0x%" PRIxPTR, ++n,
             static_cast<uintptr_t>(ip), static_cast<uintptr_t>(sp), name,
             static_cast<uintptr_t>(off));

    if (name != symbol) free(name);
    ret.push_back(buf);
  }
  return ret;
}

}  // namespace yang
