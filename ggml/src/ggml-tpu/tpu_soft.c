#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>  
#include <string.h>  

//目前的TPU模拟器只需要支持计算FP16精度类型，矩阵大小为32 * 32 的矩阵
//由于C语言没有FP16精度的数据类型，所以需要使用UINT6_t类型模拟
//需要模拟的是FP16的乘法和加法
uint16_t float_to_fp16(float f) {
    uint32_t x = *(uint32_t*)&f;
    uint32_t sign = (x >> 31) & 1;
    int exp = (x >> 23) & 0xFF;
    uint32_t mantissa = x & 0x7FFFFF;

    int new_exp = exp - 127 + 15;

    if (new_exp >= 31) {
        return (uint16_t)(sign << 15) | 0x7C00;
    }
    if (new_exp <= 0) {
        return (uint16_t)(sign << 15);
    }

    // 改进舍入：四舍五入到最近的 FP16 表示
    uint32_t new_mantissa = mantissa >> 13;
    uint32_t remainder = mantissa & 0x1FFF;  // 截取的低 13 位
    if (remainder > 0x1000) {  // 大于一半，进 1
        new_mantissa++;
        if (new_mantissa > 0x3FF) {  // 尾数溢出，调整指数
            new_mantissa = 0;
            new_exp++;
            if (new_exp >= 31) {
                return (uint16_t)(sign << 15) | 0x7C00;
            }
        }
    }

    return (uint16_t)((sign << 15) | (new_exp << 10) | new_mantissa);
}

// 工具函数：将 FP16 (uint16_t) 转回 float
float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    int exp = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    if (exp == 0 && mantissa == 0) {
        return sign ? -0.0f : 0.0f;
    }

    // 组合成 FP32
    uint32_t f_sign = sign << 31;
    uint32_t f_exp = (exp + 127 - 15) << 23;
    uint32_t f_mantissa = mantissa << 13;

    uint32_t f_bits = f_sign | f_exp | f_mantissa;
    return *(float*)&f_bits;
}
//FP16乘法逻辑
uint16_t float16_mul(uint16_t a, uint16_t b) {
    double fa = fp16_to_float(a);
    double fb = fp16_to_float(b);
    double res = fa * fb;
    return float_to_fp16((float)res);
}

uint16_t float16_add(uint16_t a, uint16_t c) {
    double fa = fp16_to_float(a);
    double fc = fp16_to_float(c);
    double res = fa + fc;
    return float_to_fp16((float)res);
}
//FP16乘加运算逻辑
double float16_mac(uint16_t a, uint16_t b, double c) {
    double fa = fp16_to_float(a);
    double fb = fp16_to_float(b);
    double fc = c;
    double res = fa * fb + fc;
    return res;
}

//用于模拟TPU 的计算效果，返回一个二级指针
//uint8_t **tpu_systolic(uint8_t **a,uint8_t **b,uint8_t **c,uint8_t **d,
//                     int input_addr, int intput_size,int output_addr,int output_size,
//                     int matrix_a_row , int matrix_a_col,int matrix_b_row , int matrix_b_col,
//                    int mode)
//{
//    uint16_t **a16 = (uint16_t**)a;
//    uint16_t **b16 = (uint16_t**)b;
//    //uint16_t **c16 = (uint16_t**)c;
//    uint16_t **d16 = (uint16_t**)d;
//     // 检查矩阵维度是否匹配
//    if (matrix_a_col != matrix_b_row) {
//        return NULL; // 矩阵无法相乘
//    }
//    // 逐元素计算 d[i][j] = sum(a[i][k] * b[k][j]) + c[i][j]
//    for (int i = 0; i < matrix_a_row; i++) {
//        for (int j = 0; j < matrix_b_col; j++) {
//            // 初始值为 c[i][j]
//            //double sum = fp16_to_float(c16[i][j]);
//            // 累加 a[i][k] * b[k][j]
//            for (int k = 0; k < matrix_a_col; k++) {
//                sum = float16_mac(a16[i][k], b16[k][j], sum);
//            }
//            // 保存结果到 d
//            d16[i][j] =  float_to_fp16((float)sum);
//        }
//    }
//    return d;          
//}
uint8_t *tpu_systolic(uint8_t *a,uint8_t *b,uint8_t *d,
                     int input_addr, int intput_size,int output_addr,int output_size,
                     int matrix_a_row , int matrix_a_col,int matrix_b_row , int matrix_b_col,
                    int mode)
{
    // 检查矩阵维度是否匹配
    if (matrix_a_col != matrix_b_row) {
      return NULL; // 矩阵无法相乘
    }
		uint16_t **a16 = malloc(matrix_a_row * sizeof(uint16_t *));
		uint16_t **b16 = malloc(matrix_b_row * sizeof(uint16_t *));
		uint16_t **d16 = malloc(matrix_d_row * sizeof(uint16_t *));
    for (int i = 0; i < matrix_a_row; i++) {
        a16[i] = malloc(matrix_a_col * sizeof(uint16_t));
        d16[i] = malloc(matrix_b_col * sizeof(uint16_t));
    }
    for (int i = 0; i < matrix_b_row; i++) {
        b16[i] = malloc(matrix_b_col * sizeof(uint16_t));
    }

		for(int i = 0; i < matrix_a_row; i++){
			for(int j = 0; j < matric_a_col; j++){
				a16[i][j] = (uint16_t)a[i*matrix_a_col+j];
			}
		}
		for(int i = 0; i < matrix_b_row; i++){
			for(int j = 0; j < matric_b_col; j++){
				b16[i][j] = (uint16_t)b[i*matrix_b_col+j];
			}
		}

    // 逐元素计算 d[i][j] = sum(a[i][k] * b[k][j]) + c[i][j]
		double sum;
    for (int i = 0; i < matrix_a_row; i++) {
        for (int j = 0; j < matrix_b_col; j++) {
            // 累加 a[i][k] * b[k][j]
            for (int k = 0; k < matrix_a_col; k++) {
                sum = float16_mac(a16[i][k], b16[k][j], sum);
            }
            // 保存结果到 d
            d16[i][j] =  float_to_fp16((float)sum);
        }
    }
		for(int i = 0; i < matrix_a_row; i++){
			for(int j = 0; j < matric_b_col; j++){
				d[i*matrix_b_col+j] = (uint8_t)d16[i][j];
			}
		}
	 	
    return d;          
}

// 创建二维 uint8_t 矩阵（实际存 uint16_t 数据）
uint8_t** create_matrix(int rows, int cols) {
    uint8_t **mat = (uint8_t**)malloc(rows * sizeof(uint8_t*));
    mat[0] = (uint8_t*)malloc(rows * cols * sizeof(uint16_t));
    for (int i = 1; i < rows; i++) {
        mat[i] = mat[i-1] + cols * sizeof(uint16_t);
    }
    memset(mat[0], 0, rows * cols * sizeof(uint16_t));
    return mat;
}

void free_matrix(uint8_t **mat) {
    if (mat) {
        free(mat[0]);
        free(mat);
    }
}

//int main() {
//    // 1. 设置矩阵维度（最少四阶）
//    int matrix_a_row = 4;
//    int matrix_a_col = 4;
//    int matrix_b_row = 4;
//    int matrix_b_col = 4;
//
//    // 2. 创建矩阵
//    uint8_t **a = create_matrix(matrix_a_row, matrix_a_col);
//    uint8_t **b = create_matrix(matrix_b_row, matrix_b_col);
//    uint8_t **c = create_matrix(matrix_a_row, matrix_b_col);
//    uint8_t **d = create_matrix(matrix_a_row, matrix_b_col);
//
//    // 3. 填充测试数据（使用 float_to_fp16 转成 FP16 位模式）
//    // a 矩阵 (4x4)
//    ((uint16_t**)a)[0][0] = float_to_fp16(1.0f); ((uint16_t**)a)[0][1] = float_to_fp16(2.0f); ((uint16_t**)a)[0][2] = float_to_fp16(3.0f); ((uint16_t**)a)[0][3] = float_to_fp16(4.0f);
//    ((uint16_t**)a)[1][0] = float_to_fp16(5.0f); ((uint16_t**)a)[1][1] = float_to_fp16(6.0f); ((uint16_t**)a)[1][2] = float_to_fp16(7.0f); ((uint16_t**)a)[1][3] = float_to_fp16(8.0f);
//    ((uint16_t**)a)[2][0] = float_to_fp16(9.0f); ((uint16_t**)a)[2][1] = float_to_fp16(10.0f);((uint16_t**)a)[2][2] = float_to_fp16(11.0f);((uint16_t**)a)[2][3] = float_to_fp16(12.0f);
//    ((uint16_t**)a)[3][0] = float_to_fp16(13.0f);((uint16_t**)a)[3][1] = float_to_fp16(14.0f);((uint16_t**)a)[3][2] = float_to_fp16(15.0f);((uint16_t**)a)[3][3] = float_to_fp16(16.0f);
//
//    // b 矩阵 (4x4)
//    ((uint16_t**)b)[0][0] = float_to_fp16(0.1f); ((uint16_t**)b)[0][1] = float_to_fp16(0.2f); ((uint16_t**)b)[0][2] = float_to_fp16(0.3f); ((uint16_t**)b)[0][3] = float_to_fp16(0.4f);
//    ((uint16_t**)b)[1][0] = float_to_fp16(0.5f); ((uint16_t**)b)[1][1] = float_to_fp16(0.6f); ((uint16_t**)b)[1][2] = float_to_fp16(0.7f); ((uint16_t**)b)[1][3] = float_to_fp16(0.8f);
//    ((uint16_t**)b)[2][0] = float_to_fp16(0.9f); ((uint16_t**)b)[2][1] = float_to_fp16(1.0f); ((uint16_t**)b)[2][2] = float_to_fp16(1.1f); ((uint16_t**)b)[2][3] = float_to_fp16(1.2f);
//    ((uint16_t**)b)[3][0] = float_to_fp16(1.3f); ((uint16_t**)b)[3][1] = float_to_fp16(1.4f); ((uint16_t**)b)[3][2] = float_to_fp16(1.5f); ((uint16_t**)b)[3][3] = float_to_fp16(1.6f);
//
//    // c 矩阵（累加项） (4x4)
//    ((uint16_t**)c)[0][0] = float_to_fp16(10.0f); ((uint16_t**)c)[0][1] = float_to_fp16(10.0f); ((uint16_t**)c)[0][2] = float_to_fp16(10.0f); ((uint16_t**)c)[0][3] = float_to_fp16(10.0f);
//    ((uint16_t**)c)[1][0] = float_to_fp16(10.0f); ((uint16_t**)c)[1][1] = float_to_fp16(10.0f); ((uint16_t**)c)[1][2] = float_to_fp16(10.0f); ((uint16_t**)c)[1][3] = float_to_fp16(10.0f);
//    ((uint16_t**)c)[2][0] = float_to_fp16(10.0f); ((uint16_t**)c)[2][1] = float_to_fp16(10.0f); ((uint16_t**)c)[2][2] = float_to_fp16(10.0f); ((uint16_t**)c)[2][3] = float_to_fp16(10.0f);
//    ((uint16_t**)c)[3][0] = float_to_fp16(10.0f); ((uint16_t**)c)[3][1] = float_to_fp16(10.0f); ((uint16_t**)c)[3][2] = float_to_fp16(10.0f); ((uint16_t**)c)[3][3] = float_to_fp16(10.0f);
//
//    // 4. 调用 tpu_systolic 计算 d = a*b + c
//    tpu_systolic(a, b, c, d,
//                 0, 0, 0, 0,
//                 matrix_a_row, matrix_a_col, matrix_b_row, matrix_b_col,
//                 0);
//
//    // 5. 打印结果
//    printf("结果矩阵 d:\n");
//    for (int i = 0; i < matrix_a_row; i++) {
//        for (int j = 0; j < matrix_b_col; j++) {
//            float val = fp16_to_float(((uint16_t**)d)[i][j]);
//            printf("%8.4f ", val);
//        }
//        printf("\n");
//    }
//
//    // 6. 验证结果是否正确（手动计算 a*b + c）
//    // 这里只列出 d[0][0] 作为示例，其他可自行计算
//    float expected[4][4] = {0};
//    // 手动计算 a*b + c
//    for (int i = 0; i < 4; i++) {
//        for (int j = 0; j < 4; j++) {
//            float sum = 0.0f;
//            for (int k = 0; k < 4; k++) {
//                sum += fp16_to_float(((uint16_t**)a)[i][k]) * fp16_to_float(((uint16_t**)b)[k][j]);
//            }
//            sum += fp16_to_float(((uint16_t**)c)[i][j]);
//            expected[i][j] = sum;
//        }
//    }
//
//    int ok = 1;
//    for (int i = 0; i < matrix_a_row; i++) {
//        for (int j = 0; j < matrix_b_col; j++) {
//            float val = fp16_to_float(((uint16_t**)d)[i][j]);
//            if (fabs(val - expected[i][j]) > 1e-1f) {
//                ok = 0;
//                printf("错误: d[%d][%d] = %f, 预期 = %f\n", i, j, val, expected[i][j]);
//            }
//        }
//    }
//    if (ok) {
//        printf("✅ 测试通过!\n");
//    } else {
//        printf("❌ 测试失败!\n");
//    }
//
//    // 7. 释放内存
//    free_matrix(a);
//    free_matrix(b);
//    free_matrix(c);
//    free_matrix(d);
//
//    return 0;
//}
