# GCC LLM Bridge (#llm)

An experimental C/C++ preprocessor extension for GCC that integrates a local Large Language Model directly into the compilation pipeline via a custom `#llm` directive.

Instead of relying on external scripts or IDE plugins, the compiler dynamically queries a local model (via `llama.cpp`) during preprocessing and injects the generated C++ code directly into the compiler's lexical token stream.

---

## How It Works

1. **Directive Interception**: The preprocessor (`libcpp`) intercepts the custom `#llm "prompt"` directive.
2. **Context Slicing**: Slices source code context (`[Code Before]` and `[Code After]`) directly from GCC's in-memory buffer without disk I/O.
3. **Bridge Inference**: Passes the prompt along with the structural context to `gcc_llm_bridge.dll`.
4. **Local Execution**: A quantized GGUF model (such as `Qwen2.5-Coder-1.5B/3B-Instruct`) generates the required raw C++ statements.
5. **Token Injection**: `libcpp` pushes the generated source directly onto the input buffer stack (`cpp_push_buffer`) for seamless AST parsing.

---

## Example

### Source Code (`test.cpp`)

```cpp
#include <iostream>

int main() {
    int ls[10] = {5, 2, 9, 1, 5, 6, 3, 8, 7, 4};

    // In-place statement generation
    #llm "write in-place loops to insertion sort ls, no functions, statements only"

    #llm "cout all the ls variable above, using a for loop"
    std::cout << std::endl;

    return 0;
}

```

### Compilation Output

```text
[GCC #llm Generated Code for "write in-place loops to insertion sort ls, no functions, statements only"]:
----------------------------------------
for (int i = 1; i < 10; ++i) {
    int key = ls[i];
    int j = i - 1;
    while (j >= 0 && ls[j] > key) {
        ls[j + 1] = ls[j];
        --j;
    }
    ls[j + 1] = key;
}
----------------------------------------

[GCC #llm Generated Code for "cout all the ls variable above, using a for loop"]:
----------------------------------------
for (int i = 0; i < 10; i++) {
    std::cout << ls[i] << " ";
}
----------------------------------------

Compilation results: 0 Errors, 0 Warnings -> test.exe

```

---

## Architecture

* **`custom-gcc/libcpp`**: Custom `do_llm` handler in GCC preprocessor (`directives.cc`). Slices buffer boundaries and manages lexical stack injection.
* **`gcc_llm_bridge.dll`**: Native C/C++ interface bridging GCC to `llama.cpp`. Linked statically against runtimes (`libgcc`, `libstdc++`, `winpthread`) to eliminate ABI conflicts and DLL search issues.
* **Inference Engine**: `llama.cpp` executing local GGUF models on CPU/GPU.

---

## Environment Variables

Runtime configuration can be customized via environment variables:

| Variable | Description | Default |
| --- | --- | --- |
| `GCC_LLM_DLL` | Absolute path to `gcc_llm_bridge.dll` | Same directory as `cc1plus.exe` |
| `GCC_LLM_MODEL` | Path to the local `.gguf` weights | Same directory / Preconfigured fallback |

---

## Building the Bridge

To build `gcc_llm_bridge.dll` on Windows using MinGW-w64 (PowerShell):

```powershell
g++ -shared -O3 -std=c++11 bridge.cpp `
    -I./include `
    -L./lib -lllama -lggml `
    -static-libgcc -static-libstdc++ `
    "-Wl,-Bstatic" -lwinpthread "-Wl,-Bdynamic" `
    -o gcc_llm_bridge.dll

```

Verify dependencies to ensure no runtime DLL conflicts:

```powershell
objdump -p gcc_llm_bridge.dll | findstr /i "DLL Name"

```

---

## License

MIT

```