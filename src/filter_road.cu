
/** 
 * @file filter_road.cu
 * Routines for different ROAD calculation variants
 */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "math_constants.h"

#include <stdio.h>
#include "image.h"
#include "math.h"


__device__
void dev_calculate_road(int* in_data, float* road_data, int pos, int width, int alpha, ROADMeasure variant) {
	float w, weights[8], r1, g1, b1, r, g, b;
	//dla kazdego piksela w kwadracie
	int a = 0;

	r = (in_data[pos] & 0XFF0000) >> 16;
	g = (in_data[pos] & 0XFF00) >> 8;
	b = (in_data[pos] & 0XFF);

	for (int i = -1; i <= 1; i++)
		for (int j = -1; j <= 1; j++)
		{
			if (i == 0 && j == 0)
				continue;
			r1 = (in_data[pos + i * width + j] & 0XFF0000) >> 16;
			g1 = (in_data[pos + i * width + j] & 0XFF00) >> 8;
			b1 = (in_data[pos + i * width + j] & 0XFF);

			if (variant==ROAD)
				weights[a] = sqrtf((r - r1) * (r - r1) + (g - g1) * (g - g1) + (b - b1) * (b - b1));
			if (variant == ROAD2)
				weights[a] = (r - r1) * (r - r1) + (g - g1) * (g - g1) + (b - b1) * (b - b1);
			if (variant == ROAD_LMAX)
				weights[a] = MAX_3(fabsf(r - r1), fabsf(g - g1), fabsf(b - b1));
			if (variant == ROAD2_LMAX)
				weights[a] = MAX_3((r - r1) * (r - r1), (g - g1) * (g - g1), (b - b1) * (b - b1));
			a++;
		}

	w = 0;
	
	for (int i = 0; (i < alpha) && (i < 8); i++)
	{
		float min = weights[0];
		int tmp = 0;
		for (int j = 1; j < 8; j++)
		{
			if (weights[j] < min)
			{
				min = weights[j];
				tmp = j;
			}
		}
		w += min;
		weights[tmp] = +INFINITY;
	}
	w /= (float)(alpha);
	road_data[pos] = w;
}

__global__
void calculate_road(int* in_data, float* road_data, int width, int height, int alpha) {
	int ic = blockIdx.y * blockDim.y + threadIdx.y;
	int ir = blockIdx.x * blockDim.x + threadIdx.x;
	if (ic >= width - 1 || ir >= height - 1 || ic < 1 || ir < 1)
		return;

	int pos = ir * width + ic;
	dev_calculate_road(in_data, road_data, pos, width, alpha, ROAD);
}

__global__
void calculate_road(int* in_data, float* road_data, int width, int height, int alpha, ROADMeasure variant) {
	int ic = blockIdx.y * blockDim.y + threadIdx.y;
	int ir = blockIdx.x * blockDim.x + threadIdx.x;
	if (ic >= width - 1 || ir >= height - 1 || ic < 1 || ir < 1)
		return;

	int pos = ir * width + ic;
	dev_calculate_road(in_data, road_data, pos, width, alpha, variant);
}

float* calculate_road(const Image* in_img, const int alpha, ROADMeasure variant)
{
SET_FUNC_NAME("calculate_road");

byte*** in_data;
int num_rows, num_cols;
if (!is_rgb_img(in_img))
{
	ERROR_RET("Not a color image !", NULL);
}

if (!IS_POS(alpha))
{
	ERROR("Alpha value ( %d ) must be positive !", alpha);
	return NULL;
}

num_rows = get_num_rows(in_img);
num_cols = get_num_cols(in_img);

in_data = (byte***)get_img_data_nd(in_img);


size_t size_i = size_t(num_rows * num_cols) * sizeof(int);
size_t size_f = size_t(num_rows * num_cols) * sizeof(float);

int* int_in_data = (int*)malloc(size_i);
for (int i = 0; i < num_rows; i++) {
	for (int j = 0; j < num_cols; j++)
	{
		int_in_data[i * num_cols + j] = (((int)in_data[i][j][0]) << 16) | ((int)in_data[i][j][1] << 8) | ((int)in_data[i][j][2]);
	}
}

int* d_in_data;
cudaMalloc((void**)&d_in_data, size_i);
cudaMemcpy(d_in_data, int_in_data, size_i, cudaMemcpyHostToDevice);

float* d_road_data;
cudaMalloc((void**)&d_road_data, size_f);

dim3 blockDim(1, 128, 1);
dim3 gridDim((unsigned int)ceil((float)num_rows / (float)blockDim.x),
(unsigned int)ceil((float)num_cols / (float)blockDim.y),
1);


calculate_road << < gridDim, blockDim >> > (d_in_data, d_road_data, num_cols, num_rows, alpha, variant);

cudaDeviceSynchronize();

float* road_data = (float*)malloc(size_f);
cudaMemcpy(road_data, d_road_data, size_f, cudaMemcpyDeviceToHost);


// Free device memory

cudaFree(d_in_data);
cudaFree(d_road_data);
cudaDeviceSynchronize();
free(int_in_data);
return road_data;


}

float* calculate_road(const Image* in_img, const int alpha) {
	return calculate_road(in_img, alpha, ROAD);
}


float avg_road(const Image* in_img, const int alpha, ROADMeasure variant) {
	int num_rows = get_num_rows(in_img);
	int num_cols = get_num_cols(in_img);
	float* road_data = calculate_road(in_img, alpha, variant);
	float average = 0;
	int count = 0;
	for (int i = 1; i < num_rows - 1; i++)
		for (int j = 1; j < num_cols - 1; j++)
		{
			average += road_data[i * num_cols + j];
			count++;
		}
	return average / count;
}


Image*
filter_road ( const Image * in_img, const int alpha, const ROADMeasure variant)
{
	byte** out_data;
	int num_rows, num_cols;
	Image* out_img;
	num_rows = get_num_rows(in_img);
	num_cols = get_num_cols(in_img);

	out_img = alloc_img(PIX_GRAY, num_rows, num_cols);
	out_data = (byte**)get_img_data_nd(out_img);

	float* road_data = calculate_road(in_img, alpha, variant);
	normalize(road_data, num_rows * num_cols);
	for (int i = 0; i < num_rows; i++)
		for (int j = 0; j < num_cols; j++)
		{
			out_data[i][j] = road_data[i * num_cols + j];

		}
	free(road_data);

	return out_img;
}

