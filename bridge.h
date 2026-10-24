#ifndef GCC_LLM_BRIDGE_H
#define GCC_LLM_BRIDGE_H

#include <stddef.h>

#ifdef _WIN32
  #define BRIDGE_API __declspec(dllexport)
#else 
  #define BRIDGE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

BRIDGE_API int bridge_llm_init(const char *model_path, int n_threads);

BRIDGE_API int bridge_llm_generate(const char *user_prompt, char *out_buf, size_t max_buf_len);

BRIDGE_API void bridge_llm_free(void);

#ifdef __cplusplus
}
#endif
#endif