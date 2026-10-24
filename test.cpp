#include <stdio.h>
#include <windows.h>

typedef int (*fn_init)(const char*, int);
typedef int (*fn_gen)(const char*, char*, size_t);
typedef void (*fn_free)(void);

int main() {
    printf("[*] 正在载入 gcc_llm_bridge.dll...\n");
    HMODULE h = LoadLibraryA("gcc_llm_bridge.dll");
    if (!h) {
        printf("[-] 加载失败，错误码: %lu\n", GetLastError());
        printf("    提示: 检查 llama.dll 和 ggml.dll 是否放在同级目录或 PATH 中\n");
        return 1;
    }

    fn_init b_init = (fn_init)GetProcAddress(h, "bridge_llm_init");
    fn_gen  b_gen  = (fn_gen)GetProcAddress(h, "bridge_llm_generate");
    fn_free b_free = (fn_free)GetProcAddress(h, "bridge_llm_free");

    if (!b_init || !b_gen || !b_free) {
        printf("[-] 获取导出符号失败!\n");
        FreeLibrary(h);
        return 1;
    }

    printf("[*] 初始化模型中...\n");
    // 替换为你的 GGUF 模型文件绝对路径（Windows 路径用正斜杠 / 避免转义）
    const char *model_path = "C:/Users/lzh/Desktop/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    if (b_init(model_path, 4) != 0) {
        printf("[-] 模型初始化失败，请检查路径是否正确\n");
        FreeLibrary(h);
        return 1;
    }

    char buf[4096];
    printf("[*] 正在执行本地推理...\n");
    int len = b_gen("bubble sort array int* a with size 10", buf, sizeof(buf));

    if (len > 0) {
        printf("\n========== 生成的代码 (%d 字节) ==========\n", len);
        printf("%s\n", buf);
        printf("==========================================\n");
    } else {
        printf("[-] 推理返回失败\n");
    }

    b_free();
    FreeLibrary(h);
    printf("[+] 测试完毕，资源已释放。\n");
    return 0;
}