#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include "ggml.h" // 假设你的项目里有这个头

// 返回一个已初始化的后端对象，供 ggml 注册
struct ggml_backend * ggml_backend_tpu_init(void);

// 如果需要暴露设备初始化 / 释放函数，也可以在这里添加
bool ggml_tpu_device_init(void);
void ggml_tpu_device_free(void);

#ifdef __cplusplus
}
#endif
