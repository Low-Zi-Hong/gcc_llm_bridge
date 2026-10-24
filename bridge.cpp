#include "bridge.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

static llama_model  *g_model    =nullptr;
static llama_context  *g_ctx    =nullptr;
static llama_sampler  *g_sampler    =nullptr;

static std::string sanitize_output(const std::string &raw) {
  std::string s = raw;

  // clear out the empty line infront
  size_t start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) return "";
  s = s.substr(start);

  if (s.rfind("```", 0) == 0) {
        size_t first_nl = s.find('\n');
        if (first_nl != std::string::npos) {
            s = s.substr(first_nl + 1);
        }
    }

  size_t last_fence = s.rfind("```");
  if (last_fence != std::string::npos) {
      s = s.substr(0, last_fence);
  }

  start = s.find_first_not_of(" \t\r\n");
  size_t end = s.find_last_not_of(" \t\r\n");
  if (start == std::string::npos || end == std::string::npos) return "";
  
  return s.substr(start, end - start + 1);
}

static void dummy_log_callback(ggml_log_level level, const char *text, void *user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

int bridge_llm_init(const char *model_path, int n_threads) {
  if(g_model && g_ctx) return 0;

  llama_log_set(dummy_log_callback, nullptr);
  llama_backend_init();

  llama_model_params mparams = llama_model_default_params();
  g_model = llama_load_model_from_file(model_path, mparams);
  if(!g_model) {
    fprintf(stderr, "[GCC_LLM_BRIDGE] Fail to load GGUP from: %s\n", model_path);
    return -1;
  }

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 2048;
  cparams.n_threads = n_threads > 0 ? n_threads : 4;
  cparams.n_threads_batch = cparams.n_threads;

  g_ctx = llama_init_from_model(g_model, cparams);
  if (!g_ctx) {
    fprintf(stderr, "[GCC_LLM_BRIDGE] Failed to init llama_context\n");
    llama_free_model(g_model);
    g_model = nullptr;
    return -2;
  }

  g_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
  llama_sampler_chain_add(g_sampler, llama_sampler_init_greedy());

  return 0;
}


int bridge_llm_generate(const char *user_prompt, char *out_buf, size_t max_buf_len) {
  if (!g_model || !g_ctx || !g_sampler || !out_buf || max_buf_len == 0) {
    return -1;
  }

  const struct llama_vocab *vocab = llama_model_get_vocab(g_model);
    if (!vocab) return -1;

std::string system_prompt = 
    "You are an expert C++ code generator for an in-place preprocessor directive.\n"
    "You will be given [Code Before], [Code After], and a [Task].\n"
    "Generate ONLY executable, non-empty C++ statements to replace the directive.\n"
    "Rules:\n"
    "1. Output ONLY the raw C++ statements needed to complete the task.\n"
    "2. NEVER output comments, placeholders like '// Insert code here', '// TODO', or explanations.\n"
    "3. Do NOT repeat code from [Code Before] or [Code After].\n"
    "4. Do NOT wrap code in markdown (no ```cpp, no ```).\n"
    "5. Accurately use the variables, types, and containers defined in the context.";

  std::string prompt;
  // 1. System Turn (必须由 <|im_end|> 显式闭合)
  prompt += "<|im_start|>system\n";
  prompt += system_prompt;
  prompt += "\n<|im_end|>\n";

  // 2. User Turn (必须由 <|im_end|> 显式闭合)
  prompt += "<|im_start|>user\n";
  prompt += user_prompt;
  prompt += "\n<|im_end|>\n";

  // 3. Assistant Turn (由模型接续自回归生成)
  prompt += "<|im_start|>assistant\n";
 
  llama_memory_t mem = llama_get_memory(g_ctx);
  llama_memory_clear(mem, true);
  llama_sampler_reset(g_sampler);

  int n_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.length(), NULL, 0, true, true);
    if (n_tokens <= 0) return -1;

  std::vector<llama_token> tokens(n_tokens);
  llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true);

  llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
  for(size_t i = 0; i < tokens.size(); i++) {
    batch.token[i]    =tokens[i];
    batch.pos[i]      =i;
    batch.n_seq_id[i]    =1;
    batch.seq_id[i][0] = 0;
    batch.logits[i]    = (i==tokens.size()-1);
  }
  batch.n_tokens = tokens.size();
  if(llama_decode(g_ctx, batch) != 0) {
    llama_batch_free(batch);
  }
  llama_batch_free(batch);

  std::string generated_text;
    const int max_predict = 256;
    int n_cur = tokens.size();

    llama_batch step_batch = llama_batch_init(1, 0, 1);

    for (int step = 0; step < max_predict; ++step) {

        llama_token new_token = llama_sampler_sample(g_sampler, g_ctx, -1);

        if (llama_token_is_eog(vocab, new_token)) {
            break;
        }

        char piece[64];
#if LLAMA_API_VERSION >= 3000 || defined(LLAMA_VOCAB_H)
        int n_chars = llama_token_to_piece(vocab, new_token, piece, sizeof(piece), 0, false);
#else
        int n_chars = llama_token_to_piece(vocab, new_token, piece, sizeof(piece), 0, false);
#endif
        if (n_chars > 0) {
            generated_text.append(piece, n_chars);
        }


        step_batch.token[0]     = new_token;
        step_batch.pos[0]       = n_cur;
        step_batch.n_seq_id[0]  = 1;
        step_batch.seq_id[0][0] = 0;
        step_batch.logits[0]    = true;
        step_batch.n_tokens     = 1;

        if (llama_decode(g_ctx, step_batch) != 0) {
            break;
        }
        n_cur++;
    }
    llama_batch_free(step_batch);

    std::string final_code = sanitize_output(generated_text);
    if (final_code.empty()) {
        return -1;
    }

    size_t copy_len = std::min(final_code.length(), max_buf_len - 1);
    memcpy(out_buf, final_code.c_str(), copy_len);
    out_buf[copy_len] = '\0';

    return static_cast<int>(copy_len);
}

void bridge_llm_free(void) {
    if (g_sampler) {
        llama_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_free_model(g_model);
        g_model = nullptr;
    }
    llama_backend_free();
}



