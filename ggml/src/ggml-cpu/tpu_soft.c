#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>  
#include <string.h>  
#include <assert.h>

uint8_t *tpu_systolic(uint8_t *a, uint8_t *b, uint8_t *c, uint8_t *d,
                      int nb_a_row, int nb_a_col,
                      int nb_b_row, int nb_b_col,
                      int nb_d_row, int nb_d_col,
                      int matrix_a_row, int matrix_a_col,
                      int matrix_b_row, int matrix_b_col,
                      int mode) {
    // 检查矩阵乘法维度是否匹配
    if (matrix_a_col != matrix_b_row) {
        fprintf(stderr, "[TPU] Shape mismatch: A[%d×%d] * B[%d×%d]\n",
                matrix_a_row, matrix_a_col, matrix_b_row, matrix_b_col);
        return NULL;
    }
    // 遍历 A 的行和 B 的列
    for (int i = 0; i < matrix_a_row; i++) {
        for (int j = 0; j < matrix_b_col; j++) {

            double sum = 0.0f;
            if (mode == 1 && c){
                sum = (*(float *)(c + i * nb_d_row + j * nb_d_col));
						}
            // 遍历内积维度
            for (int k = 0; k < matrix_a_col; k++) {
                float *va = (float *)((char *)a + i * nb_a_row + k * nb_a_col);
                float *vb = (float *)((char *)b + k * nb_b_row + j * nb_b_col);

                //float va = (*(float *)a_ptr);
                //float vb = (*(float *)b_ptr);

                sum += (double)(*va) * (double)(*vb);

                // 调试输出
                //fprintf(stderr, "A = [%u x %u], B = [%u x %u]\n", matrix_a_row, matrix_a_col, matrix_b_row, matrix_b_col);
                //fprintf(stderr, "A[%d][%d]=%f, B[%d][%d]=%f, sum=%f\n",
                //         i, k, *va, k, j, *vb, sum);
            }
                fprintf(stderr, "sum=%.6f\n", sum);

            // 写入结果
            uint8_t *d_ptr = d + i * nb_d_row + j * nb_d_col;
            *(float *)d_ptr = sum;
        }
    }

    return d;
}
