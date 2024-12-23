#ifdef CUDA
/**
 * @file filter_rlsf_mp.c
 * Routines for NLM GDP filtering of a color image
 * Multithreaded version using CUDA
 */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "math_constants.h"

#include <stdio.h>

#include "image.h"

 /**
  * @brief Implements the Robust Adaptive Denoising technique
  *
  * @param[in] variant Algorihm variant
  * @param[in] in_img Image pointer { rgb }
  * @param[in] r Radius of the Block { positive }
  * @param[in] f Radius of the patch { positive }
  * @param[in] alpha Number of pixels taked into account { positive }
  * @param[in] sigma h prameter { positive }
  * @param[in] sigmai sigma prameter { positive }
  *
  * @return Pointer to the filtered image or NULL
  *
  * @note Pixels outside the convolution area are not taken into account.
  *

  *
  * @author Kusnik Damian
  * @date 08.11.2024
  */

#define BLOCK_SIZE 4
#define ROUNDS 16


__device__
float weight(int width, int height, int* in_data, float* partial_data, int px, int py, int pz, int qx, int qy, int qz, int f, float sigma, float sigmai, int normalize)
{
	int sum = 0.0;
	float wsum = 0.0, isum = 0.0;
	float impulsity, d;// , weight;
	float r, g, b;
	//int S[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };

	for (int i = -f; i <= f; i++) {
		if (py + i < 0 || py + i >= height) continue;
		if (qy + i < 0 || qy + i >= height) continue;
		for (int j = -f; j <= f; j++) {
			if (px + j < 0 || px + j >= width) continue;
			if (qx + j < 0 || qx + j >= width) continue;
			int p = (py + i) * width + (px + j) + pz*width*height;
			int q = (qy + i) * width + (qx + j) + qz*width*height;
			//impulsity = __expf(-max(partial_data[q], partial_data[p]) / sigmai);
			//BS
			impulsity = __expf(-(partial_data[q] + partial_data[p]) / sigmai);
			r = ((in_data[p] & 0xFF0000) >> 16) - ((in_data[q] & 0xFF0000) >> 16);
			g = ((in_data[p] & 0xFF00) >> 8) - ((in_data[q] & 0xFF00) >> 8);
			b = ((in_data[p] & 0xFF) - (in_data[q] & 0xFF));
			sum = (r * r + g * g + b * b);// *S[i + 1][j + 1];

			wsum += sum * impulsity ;
			isum += impulsity;
		}
	}
	if(normalize)
		d = wsum / isum;
	else // BS variant
		d = wsum / ((2.0 * f + 1.0) * (2.0 * f + 1.0));

	// v2 BS
	//d = max((float)0, d - 3000.0);
	return __expf(-d / sigma);
}

__device__
float weight(int width, int height, int* in_data, float* partial_data, int px, int py, int qx, int qy, int f, float sigma, float sigmai, int normalize)
{
	return weight(width, height, in_data, partial_data, px, py, 0, qx, qy, 0, f, sigma, sigmai, normalize);
}



__device__
float compute_ROAD_lpsf_reachability(int* in_data, int width, int pixel_pos, int window_pos, int beta) {
	float w, weights[9], r1, g1, b1;

	int f = 1;
	int a = 0;
	float r = (in_data[pixel_pos] & 0XFF0000) >> 16;
	float g = (in_data[pixel_pos] & 0XFF00) >> 8;
	float b = (in_data[pixel_pos] & 0XFF);

	for (int i = -f; i <= f; i++)
		for (int j = -f; j <= f; j++)
		{
			r1 = (in_data[window_pos + i * width + j] & 0XFF0000) >> 16;
			g1 = (in_data[window_pos + i * width + j] & 0XFF00) >> 8;
			b1 = (in_data[window_pos + i * width + j] & 0XFF);
			weights[a] = (r - r1) * (r - r1) + (g - g1) * (g - g1) + (b - b1) * (b - b1);
			a++;
		}

	w = 0;

	//this is faster than sorting
	for (int i = 0; (i < beta) && (i < a); i++)
	{
		float min = weights[0];
		int tmp = 0;
		for (int j = 1; j < 9; j++)
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

	w /= (float)(beta);

	return w;
}


__global__
void precalculate_pixels_ROAD(int* in_data, float* pixels_ROAD, const int width, const int height, const int alpha)
{
	int ic, ir;
	int f = 1;

	ic = blockIdx.y * blockDim.y + threadIdx.y;
	ir = blockIdx.x * blockDim.x + threadIdx.x;
	if (ic >= width - f || ir >= height - f || ic < f || ir < f)
		return;
	int pos = ir * width + ic;
	pixels_ROAD[pos] = compute_ROAD_lpsf_reachability(in_data, width, pos, pos, alpha);
}

__global__
void denoise_nlm_pixelwise(int* in_data, float* partial_data, float* out_data, const int width, const int height, const int r, const int f, const float sigma, const float sigmai, int central_variant)
{
	int ic = blockIdx.y * blockDim.y + threadIdx.y;
	int ir = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int data_len = height * width;
	if (ic >= width || ir >= height)
		return;

	float sum[3];
	float wsum;
	float w;
	int index;
	float wmax = -INFINITY;
	sum[0] = 0;
	sum[1] = 0;
	sum[2] = 0;
	wsum = 0;


	int istart = max(ir - r, 0);
	int iend = min(ir + r, height - 1);
	int jstart = max(ic - r, 0);
	int jend = min(ic + r, width - 1);

	// go through all patches
	int pos = ir * width + ic;
	for (int i = istart; i <= iend; i++) { // i = y
		int offset = i * width;
		for (int j = jstart; j <= jend; j++) { // j = x

			if (ic == j && ir == i) continue;
			/* 1 - normalize, 0 not*/
			if(central_variant>9)
				w = weight(width, height, in_data, partial_data, ic, ir, j, i, f, sigma, sigmai, 1);
			else
				w = weight(width, height, in_data, partial_data, ic, ir, j, i, f, sigma, sigmai, 0);
			if (w > wmax) wmax = w;

			float impulsity = __expf(-(partial_data[offset + j] / sigmai));

			sum[0] += ((in_data[offset + j] & 0XFF0000) >> 16) * w * impulsity;
			sum[1] += ((in_data[offset + j] & 0XFF00) >> 8) * w * impulsity;
			sum[2] += (in_data[offset + j] & 0XFF) * w * impulsity;
			wsum += w * impulsity;
			index++;
		}
	}
	if (central_variant % 10)
	{
		/* central_variant:
			0. weight = 0 --> without central
			1. weight = 0.7*w
			2. weight = maxw*w
			3. weight = with central
		*/
		if (central_variant > 9)
			w = weight(width, height, in_data, partial_data, ic, ir, ic, ir, f, sigma, sigmai, 1);
		else
			w = weight(width, height, in_data, partial_data, ic, ir, ic, ir, f, sigma, sigmai, 0);

		if (central_variant % 10 == 1)
			w = 0.7 * w;
		else if (central_variant % 10 == 2)
			w = wmax;
		float impulsity = __expf(-(partial_data[pos] / sigmai));
		sum[0] += ((in_data[pos] & 0XFF0000) >> 16)* w * impulsity;
		sum[1] += ((in_data[pos] & 0XFF00) >> 8)* w * impulsity;
		sum[2] += (in_data[pos] & 0XFF) * w * impulsity;
		wsum += w * impulsity;
	}

	out_data[pos] = sum[0] / wsum;
	out_data[data_len + pos] = sum[1] / wsum;
	out_data[data_len + data_len + pos] = sum[2] / wsum;
	return;
}

__global__
void normalizePixels(float* pixel_data, int* out_data, int width, int data_len, float divider) {
	int ic = blockIdx.y * blockDim.y + threadIdx.y;
	int ir = blockIdx.x * blockDim.x + threadIdx.x;

	const unsigned int pos = ir * width + ic;
	if (ic >= width || pos >= data_len)
		return;
	out_data[pos] = ((int)(pixel_data[pos] / divider) << 16) |
		((int)(pixel_data[data_len + pos] / divider) << 8) |
		((int)(pixel_data[2 * data_len + pos] / divider));
}


__global__
void denoise_patch(int* in_data, float* partial_data, float* out_data, const int width, const int height, const int r, const int f, const float sigma, const float sigmai, int central_variant)
{
	int ic = blockIdx.y * blockDim.y + threadIdx.y;
	int ir = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int data_len = height * width;
	if (ic >= width || ir >= height)
		return;

	float sum[3][81];
	float wsum[81];
	float w;
	float impulsity;
	int index;
	float wmax = -INFINITY;
	for (int i = 0; i < 49; i++)
	{
		sum[0][i] = 0;
		sum[1][i] = 0;
		sum[2][i] = 0;
		wsum[i] = 0;
	}


	int istart = max(ir - r, 0);
	int iend = min(ir + r, height - 1);
	int jstart = max(ic - r, 0);
	int jend = min(ic + r, width - 1);

	// go through all patches
	for (int i = istart; i <= iend; i++) { // i = y
		for (int j = jstart; j <= jend; j++) { // j = x

			if (ic == j && ir == i) continue;

			if (central_variant > 9)
				w = weight(width, height, in_data, partial_data, ic, ir, j, i, f, sigma, sigmai, 1);
			else
				w = weight(width, height, in_data, partial_data, ic, ir, j, i, f, sigma, sigmai, 0);

			if (w > wmax) wmax = w;

			// Denoise patch

			int i2start = max(i - f, 0);
			int i2end = min(i + f, height - 1);
			int j2start = max(j - f, 0);
			int j2end = min(j + f, width - 1);

			index = 0;
			for (int i2 = i2start; i2 <= i2end; i2++) { // i2 = y
				int offset2 = i2 * width;
				for (int j2 = j2start; j2 <= j2end; j2++) { // j2 = x
					impulsity = __expf(-(partial_data[offset2 + j2] / sigmai));

					sum[0][index] += ((in_data[offset2 + j2] & 0XFF0000) >> 16)* w* impulsity;
					sum[1][index] += ((in_data[offset2 + j2] & 0XFF00) >> 8)* w* impulsity;
					sum[2][index] += (in_data[offset2 + j2] & 0XFF) * w * impulsity;
					wsum[index] += w * impulsity;
					index++;
				}
			}
		}
	}

	int i2start = max(ir - f, 0);
	int i2end = min(ir + f, height - 1);
	int j2start = max(ic - f, 0);
	int j2end = min(ic + f, width - 1);

	index = 0;
	
	if (central_variant % 10)
	{
		/* central_variant:
			0. weight = 0 --> without central
			1. weight = 0.7*w
			2. weight = maxw*w
			3. weight = with central
		*/
		if (central_variant > 9)
			w = weight(width, height, in_data, partial_data, ic, ir, ic, ir, f, sigma, sigmai, 1);
		else
			w = weight(width, height, in_data, partial_data, ic, ir, ic, ir, f, sigma, sigmai, 0);

		if (central_variant % 10 == 1)
			w = 0.7 * w;
		else if (central_variant % 10 == 2)
			w = wmax;
	}
	else
		w = 0;


	for (int i2 = i2start; i2 <= i2end; i2++) { // i2 = y
		int offset2 = i2 * width;
		for (int j2 = j2start; j2 <= j2end; j2++) { // j2 = x
			impulsity = __expf(-(partial_data[offset2 + j2] / sigmai));
			sum[0][index] += ((in_data[offset2 + j2] & 0XFF0000) >> 16) * w* impulsity;
			sum[1][index] += ((in_data[offset2 + j2] & 0XFF00) >> 8) * w* impulsity;
			sum[2][index] += (in_data[offset2 + j2] & 0XFF) * w * impulsity;
			wsum[index] += w * impulsity;

			atomicAdd(&out_data[offset2 + j2], sum[0][index] / wsum[index]);
			atomicAdd(&out_data[data_len + offset2 + j2], sum[1][index] / wsum[index]);
			atomicAdd(&out_data[data_len + data_len + offset2 + j2], sum[2][index] / wsum[index]);
			index++;
		}
	}

	return;
}


Image*
filter_rad(const char* variant, const Image* in_img, const int r, const int f, const int alpha, const float sigma, const float sigmai)
{
	SET_FUNC_NAME("filter_rad");
	byte*** in_data;
	byte*** out_data;
	int num_rows, num_cols;
	Image* out_img;

	if (!is_rgb_img(in_img))
	{
		ERROR_RET("Not a color image !", NULL);
	}

	if (!IS_POS(r))
	{
		ERROR("Window size ( %d ) must be positive!", r);
		return NULL;
	}

	if (!IS_POS(f))
	{
		ERROR("Patch size value ( %d ) must be positive !", f);
		return NULL;
	}
	if (!IS_POS(alpha))
	{
		ERROR("Alpha value ( %d ) must be positive !", alpha);
		return NULL;
	}

	if (!IS_POS(sigma))
	{
		ERROR("Sigma value ( %d ) must be positive !", sigma);
		return NULL;
	}

	num_rows = get_num_rows(in_img);
	num_cols = get_num_cols(in_img);

	in_data = (byte***)get_img_data_nd(in_img);
	out_img = alloc_img(PIX_RGB, num_rows, num_cols);
	out_data = (byte***)get_img_data_nd(out_img);

	size_t size_i = size_t(num_rows * num_cols) * sizeof(int);
	size_t size_f = size_t(num_rows * num_cols) * sizeof(float);

	int* int_in_data = (int*)malloc(size_i);
	for (int i = 0; i < num_rows; i++) {
		for (int j = 0; j < num_cols; j++)
		{
			int_in_data[i * num_cols + j] = (((int)in_data[i][j][0]) << 16) | ((int)in_data[i][j][1] << 8) | ((int)in_data[i][j][2]);
		}
	}

	int* d_int_out_data;
	cudaMalloc((void**)&d_int_out_data, size_i);

    float* d_out_data;
	cudaMalloc((void**)&d_out_data, size_f*3);
	cudaMemset(d_out_data, 0, 3 * size_f);

	float* d_partial_data;
	cudaMalloc((void**)&d_partial_data, size_f);
	cudaMemset(d_partial_data, 0, size_f);

	int* d_in_data;
	cudaMalloc((void**)&d_in_data, size_i);

	cudaMemcpy(d_in_data, int_in_data, size_i, cudaMemcpyHostToDevice);

	dim3 blockDim(1, 128, 1);
	dim3 gridDim((unsigned int)ceil((float)num_rows / (float)blockDim.x),
		(unsigned int)ceil((float)num_cols / (float)blockDim.y),
		1);

	precalculate_pixels_ROAD << < gridDim, blockDim >> > (d_in_data, d_partial_data, num_cols, num_rows, alpha);
	cudaDeviceSynchronize();

	/* central_variant:
	0. weight = 0 --> without central
	1. weight = 0.7*w
	2. weight = maxw*w
	3. weight = with central
	*/

	if (!strcmp(variant, "BS_NC_PIXELWISE")){
		// BS without central pixel
		denoise_nlm_pixelwise << < gridDim, blockDim >> > (d_in_data, d_partial_data, d_out_data, num_cols, num_rows, r, f, 2 * sigma * sigma, 2 * sigmai * sigmai, 0);
		cudaDeviceSynchronize();
		normalizePixels << < gridDim, blockDim >> > (d_out_data, d_int_out_data, num_cols, num_cols * num_rows, 1.0);
	}
	else if (!strcmp(variant, "BS_07C_PIXELWISE")) {
		// BS weight 0.7 Central
		denoise_nlm_pixelwise << < gridDim, blockDim >> > (d_in_data, d_partial_data, d_out_data, num_cols, num_rows, r, f, 2 * sigma * sigma, 2 * sigmai * sigmai, 1);
		cudaDeviceSynchronize();
		normalizePixels << < gridDim, blockDim >> > (d_out_data, d_int_out_data, num_cols, num_cols * num_rows, 1.0);
	}
	else if (!strcmp(variant, "BS_MAXC_PIXELWISE")) {
		// BS weight max Central
		denoise_nlm_pixelwise << < gridDim, blockDim >> > (d_in_data, d_partial_data, d_out_data, num_cols, num_rows, r, f, 2 * sigma * sigma, 2 * sigmai * sigmai, 2);
		cudaDeviceSynchronize();
		normalizePixels << < gridDim, blockDim >> > (d_out_data, d_int_out_data, num_cols, num_cols * num_rows, 1.0);
	}
	else if (!strcmp(variant, "BS_WC_PIXELWISE")) {
		// BS with central pixel
		denoise_nlm_pixelwise << < gridDim, blockDim >> > (d_in_data, d_partial_data, d_out_data, num_cols, num_rows, r, f, 2 * sigma * sigma, 2 * sigmai * sigmai, 3);
		cudaDeviceSynchronize();
		normalizePixels << < gridDim, blockDim >> > (d_out_data, d_int_out_data, num_cols, num_cols * num_rows, 1.0);
	}
	else if (!strcmp(variant, "BS_NC_PATCHWISE")) {
		denoise_patch << < gridDim, blockDim >> > (d_in_data, d_partial_data, d_out_data, num_cols, num_rows, r, f, 2 * sigma * sigma, 2 * sigmai * sigmai, 0);
		cudaDeviceSynchronize();
		normalizePixels << < gridDim, blockDim >> > (d_out_data, d_int_out_data, num_cols, num_cols * num_rows, (2 * f + 1) * (2 * f + 1));
	}
	else if (!strcmp(variant, "BS_07C_PATCHWISE")) {
		denoise_patch << < gridDim, blockDim >> > (d_in_data, d_partial_data, d_out_data, num_cols, num_rows, r, f, 2 * sigma * sigma, 2 * sigmai * sigmai, 1);
		cudaDeviceSynchronize();
		normalizePixels << < gridDim, blockDim >> > (d_out_data, d_int_out_data, num_cols, num_cols * num_rows, (2 * f + 1) * (2 * f + 1));
	}
	else if (!strcmp(variant, "BS_MAXC_PATCHWISE")) {
		denoise_patch << < gridDim, blockDim >> > (d_in_data, d_partial_data, d_out_data, num_cols, num_rows, r, f, 2 * sigma * sigma, 2 * sigmai * sigmai, 2);
		cudaDeviceSynchronize();
		normalizePixels << < gridDim, blockDim >> > (d_out_data, d_int_out_data, num_cols, num_cols * num_rows, (2 * f + 1) * (2 * f + 1));
	}
	else if (!strcmp(variant, "BS_WC_PATCHWISE")) {
		denoise_patch << < gridDim, blockDim >> > (d_in_data, d_partial_data, d_out_data, num_cols, num_rows, r, f, 2 * sigma * sigma, 2 * sigmai * sigmai, 3);
		cudaDeviceSynchronize();
		normalizePixels << < gridDim, blockDim >> > (d_out_data, d_int_out_data, num_cols, num_cols * num_rows, (2 * f + 1) * (2 * f + 1));
	} 
	else if (!strcmp(variant, "NORMALIZED_NC_PIXELWISE")) {//weight variants
		// BS without central pixel
		denoise_nlm_pixelwise << < gridDim, blockDim >> > (d_in_data, d_partial_data, d_out_data, num_cols, num_rows, r, f, 2 * sigma * sigma, 2 * sigmai * sigmai, 10);
		cudaDeviceSynchronize();
		normalizePixels << < gridDim, blockDim >> > (d_out_data, d_int_out_data, num_cols, num_cols * num_rows, 1.0);
	}
	else if (!strcmp(variant, "NORMALIZED_07C_PIXELWISE")) {
		// BS weight 0.7 Central
		denoise_nlm_pixelwise << < gridDim, blockDim >> > (d_in_data, d_partial_data, d_out_data, num_cols, num_rows, r, f, 2 * sigma * sigma, 2 * sigmai * sigmai, 11);
		cudaDeviceSynchronize();
		normalizePixels << < gridDim, blockDim >> > (d_out_data, d_int_out_data, num_cols, num_cols * num_rows, 1.0);
	}
	else if (!strcmp(variant, "NORMALIZED_MAXC_PIXELWISE")) {
		// BS weight max Central
		denoise_nlm_pixelwise << < gridDim, blockDim >> > (d_in_data, d_partial_data, d_out_data, num_cols, num_rows, r, f, 2 * sigma * sigma, 2 * sigmai * sigmai, 12);
		cudaDeviceSynchronize();
		normalizePixels << < gridDim, blockDim >> > (d_out_data, d_int_out_data, num_cols, num_cols * num_rows, 1.0);
	}
	else if (!strcmp(variant, "NORMALIZED_WC_PIXELWISE")) {
		// BS with central pixel
		denoise_nlm_pixelwise << < gridDim, blockDim >> > (d_in_data, d_partial_data, d_out_data, num_cols, num_rows, r, f, 2 * sigma * sigma, 2 * sigmai * sigmai, 13);
		cudaDeviceSynchronize();
		normalizePixels << < gridDim, blockDim >> > (d_out_data, d_int_out_data, num_cols, num_cols * num_rows, 1.0);
	}
	else if (!strcmp(variant, "NORMALIZED_NC_PATCHWISE")) {
		denoise_patch << < gridDim, blockDim >> > (d_in_data, d_partial_data, d_out_data, num_cols, num_rows, r, f, 2 * sigma * sigma, 2 * sigmai * sigmai, 10);
		cudaDeviceSynchronize();
		normalizePixels << < gridDim, blockDim >> > (d_out_data, d_int_out_data, num_cols, num_cols * num_rows, (2 * f + 1) * (2 * f + 1));
	}
	else if (!strcmp(variant, "NORMALIZED_07C_PATCHWISE")) {
		denoise_patch << < gridDim, blockDim >> > (d_in_data, d_partial_data, d_out_data, num_cols, num_rows, r, f, 2 * sigma * sigma, 2 * sigmai * sigmai, 11);
		cudaDeviceSynchronize();
		normalizePixels << < gridDim, blockDim >> > (d_out_data, d_int_out_data, num_cols, num_cols * num_rows, (2 * f + 1) * (2 * f + 1));
	}
	else if (!strcmp(variant, "NORMALIZED_WC_PATCHWISE")) {
		denoise_patch << < gridDim, blockDim >> > (d_in_data, d_partial_data, d_out_data, num_cols, num_rows, r, f, 2 * sigma * sigma, 2 * sigmai * sigmai, 13);
		cudaDeviceSynchronize();
		normalizePixels << < gridDim, blockDim >> > (d_out_data, d_int_out_data, num_cols, num_cols * num_rows, (2 * f + 1) * (2 * f + 1));
	}
	else { //if (!strcmp(variant, "NORMALIZED_MAXC_PATCHWISE")) {
		denoise_patch << < gridDim, blockDim >> > (d_in_data, d_partial_data, d_out_data, num_cols, num_rows, r, f, 2 * sigma * sigma, 2 * sigmai * sigmai, 12);
		cudaDeviceSynchronize();
		normalizePixels << < gridDim, blockDim >> > (d_out_data, d_int_out_data, num_cols, num_cols * num_rows, (2 * f + 1) * (2 * f + 1));
	}

	cudaDeviceSynchronize();

	int* int_out_data = (int*)malloc(size_i);
	cudaMemcpy(int_out_data, d_int_out_data, size_i, cudaMemcpyDeviceToHost);


	for (int i = 0; i < num_rows; i++)
		for (int j = 0; j < num_cols; j++)
		{
			out_data[i][j][0] = (int_out_data[i * num_cols + j] >> 16) & 0xFF;
			out_data[i][j][1] = (int_out_data[i * num_cols + j] >> 8) & 0xFF;
			out_data[i][j][2] = (int_out_data[i * num_cols + j]) & 0xFF;

		}

	// Free device memory
	cudaFree(d_in_data);
	cudaFree(d_out_data);
	cudaFree(d_int_out_data);
	cudaFree(d_partial_data);
	cudaDeviceSynchronize();

	free(int_in_data);
	free(int_out_data);

	return out_img;
}

#endif
