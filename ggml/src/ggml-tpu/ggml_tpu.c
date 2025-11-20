// Minimal ggml "tpu" backend example
// 说明：请根据你本地 ggml 的具体类型和 API 适配（名字、成员等）。

#include "ggml_tpu.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>

// ---- 用户提供或替换的 TPU 仿真 / 硬件接口 ----
// 你已经有的函数原型（示例）：
// uint8_t **tpu_systolic(uint8_t **a, uint8_t **b, uint8_t **c, uint8_t **d, ...);
// 为了简单起见我们在这里声明一个更易用的 wrapper。

extern uint8_t ** tpu_systolic(uint8_t **a,uint8_t **b,uint8_t **c,uint8_t **d,
                     int input_addr, int intput_size,int output_addr,int output_size,
                     int matrix_a_row , int matrix_a_col,int matrix_b_row , int matrix_b_col,
                    int mode);

// ---- 简单 host-backed device buffer 结构 ----
struct tpu_buffer {
    void * ptr;       // 指向实际内存（host 或 device-mapped）
    size_t size;
    int device_id;    // 如果多设备可用
};

// ---- 后端 buffer type API (示例签名) ----
// 注意：不同 ggml 版本的函数签名可能不同，按你本地实现调整
static void * tpu_alloc_buffer(size_t size) {
    struct tpu_buffer * b = malloc(sizeof(*b));
    if (!b) return NULL;
    b->ptr = malloc(size);
    if (!b->ptr) { free(b); return NULL; }
    b->size = size;
    b->device_id = 0;
    return b;
}

static void tpu_free_buffer(void * buf) {
    if (!buf) return;
    struct tpu_buffer * b = (struct tpu_buffer*)buf;
    free(b->ptr);
    free(b);
}

static void tpu_memcpy_to(void * dst_buf, const void * src, size_t n) {
    struct tpu_buffer * b = (struct tpu_buffer*)dst_buf;
    if (n > b->size) n = b->size;
    memcpy(b->ptr, src, n);
}

static void tpu_memcpy_from(void * src_buf, void * dst, size_t n) {
    struct tpu_buffer * b = (struct tpu_buffer*)src_buf;
    if (n > b->size) n = b->size;
    memcpy(dst, b->ptr, n);
}

// ---- 后端支持的 op 判定 ----
static bool tpu_supports_op(const struct ggml_tensor * t) {
    // 这里只把矩阵乘（及可融合 add）的算子放进硬件处理范围
    // ggml 中 op 枚举名请按你本地仓库替换
    switch (t->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID: // 若存在融合 ID 的变体
            return true;
        default:
            return false;
    }
}

// ---- GEMM 调用封装（使用你的 tpu_systolic 或真实驱动） ----
// 这里以 FP16 (uint16_t) 每元素为例。你需要确保 ggml tensor 的 layout 与此兼容。
// 这个实现把 A,B,C 的指针解释为二维指针数组，然后调用 tpu_systolic。
// 注意：实际 ggml tensor 的内存通常是连续 flatten 的，下面仅为 demo。
static void tpu_execute_gemm_fp16(
        uint16_t ** A, uint16_t ** B, uint16_t ** C, uint16_t ** D,
        int a_r, int a_c, int b_r, int b_c)
{
    // 调用你已有的仿真/驱动
    tpu_systolic((uint8_t**)A, (uint8_t**)B, (uint8_t**)C, (uint8_t**)D,
                 0,0,0,0, a_r, a_c, b_r, b_c, 0);
}

// ---- 将 ggml_tensor 转换为 二级指针数组（示例） ----
// 注意：ggml tensor 内存布局视实现而定 —— 常见是连续内存 + strides。
// 这里提供一个简化版本：假设每行连续、每元素为 uint16_t，返回行指针数组。
// 真实工程应处理 strides、对齐、量化等情况。
static uint16_t ** tensor_to_row_ptrs_uint16(struct ggml_tensor * t, int rows, int cols) {
    uint16_t ** rows_ptrs = malloc(sizeof(uint16_t*) * rows);
    uint16_t * base = (uint16_t*) t->data; // 请确保 t->data 与 uint16_t 对齐
    for (int i = 0; i < rows; ++i) {
        rows_ptrs[i] = base + i * cols;
    }
    return rows_ptrs;
}

// ---- submit: 遍历计算图并对可支持的节点调用硬件 ----
// 这是一个极简示例：遍历 graph nodes（假设 graph 给出 node 列表），
// 若 node->op 是矩阵乘则准备数据并调用 tpu_execute_gemm_fp16。
static void tpu_submit(struct ggml_backend_context * ctx,
                       const struct ggml_cgraph * graph) {
    // 注意：ggml_cgraph 在不同版本中字段不同。以下为示意伪代码。
    for (int i = 0; i < graph->n_nodes; ++i) {
        struct ggml_tensor * node = graph->nodes[i];
        if (!tpu_supports_op(node)) continue;

        // 假设 node->src[0], src[1] 分别为 A,B；dst 为 node
        struct ggml_tensor * dst = node;
        struct ggml_tensor * A_t = dst->src[0];
        struct ggml_tensor * B_t = dst->src[1];
        struct ggml_tensor * C_t = NULL; // 如果有 bias/add 融合

        // 获取几何信息（请根据 ggml 的字段替换）
        int a_r = A_t->ne[0];
        int a_c = A_t->ne[1];
        int b_r = B_t->ne[0];
        int b_c = B_t->ne[1];

        // 把 ggml tensor 的 data 转换为行指针
        uint16_t ** A_rows = tensor_to_row_ptrs_uint16(A_t, a_r, a_c);
        uint16_t ** B_rows = tensor_to_row_ptrs_uint16(B_t, b_r, b_c);
        uint16_t ** C_rows = NULL;
        uint16_t ** D_rows = tensor_to_row_ptrs_uint16(dst, a_r, b_c);

        // 如果 ggml 支持把 C 作为第三个 src，设置 C_rows
        if (dst->n_src == 3) {
            C_t = dst->src[2];
            C_rows = tensor_to_row_ptrs_uint16(C_t, a_r, b_c);
        } else {
            // 如果没有 C，构造一个零矩阵或直接传 NULL（视你的 tpu_systolic 实现）
        }

        // 调用硬件仿真 / 驱动
        tpu_execute_gemm_fp16(A_rows, B_rows, C_rows, D_rows, a_r, a_c, b_r, b_c);

        // 释放临时 row 指针数组（不释放底层数据）
        free(A_rows);
        free(B_rows);
        if (C_rows) free(C_rows);
        free(D_rows);
    }
}

// ---- synchronize: 若硬件是异步，这里等待完成 ----
static void tpu_synchronize(struct ggml_backend_context * ctx) {
    // 如果使用中断或异步队列，在此做等待或 flush
}

// ---- 注册 / 构造 ggml_backend 结构体 ----
struct ggml_backend * ggml_backend_tpu_init(void) {
    struct ggml_backend * backend = calloc(1, sizeof(*backend));
    if (!backend) return NULL;

    backend->name = "tpu";
    backend->type = GGML_BACKEND_DEVICE_TYPE_CUSTOM; // 视 ggml 定义而定
    backend->buffer_type = NULL; // 如果需要，创建并返回一个 buffer_type 对象
    backend->supports_op = tpu_supports_op;
    backend->submit = tpu_submit;
    backend->synchronize = tpu_synchronize;

    // 如果你的 ggml 有 buffer type 注册需要把相关函数指针绑上（见上方 tpu_alloc_buffer 等）
    // backend->buffer_type = ggml_backend_buffer_type_create("tpu", tpu_alloc_buffer, tpu_free_buffer, ...);

    return backend;
}

// 可选：设备 init / free
bool ggml_tpu_device_init(void) {
    // 打开 / mmap / 初始化驱动
    return true;
}
void ggml_tpu_device_free(void) {
    // 清理资源
}
