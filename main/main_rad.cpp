#include "image.h"
#include "math.h"

int main(int argc, char** argv)
{
    double elapsed_time;
    clock_t start_time;
    Image *in_img;
    Image *noisy_img;
    Image *out_img;
    int f, s_f, e_f;
	int alpha, s_alpha, e_alpha;
	int r, s_r, e_r;;
    float sigma, s_sigma, e_sigma;
	float sigmai, s_sigmai, e_sigmai;
    int i;
    int benchmark = 0;
	char* image_db_name=NULL;
	double psnr = 0.0 ;
	double* psnr_arr, * ssim_arr;
    char* variant;

    if ( argc < 4 )
    {
        printf("argc: %d\n", argc);
        fprintf(stderr, "Usage: %s <variant> <reference image { rgb }> <noisy image {rgb}> <block_radius> <patch_radius> <alpha> <sigma> <h> \n", argv[0]);
		fprintf(stderr, "Possible variants: SELFTUNE {BS|NORMALIZED}_{NC|07C|MAXC|WC}_{PIXELWISE|PATCHWISE} \n");
        exit ( EXIT_FAILURE );
    }
    if (argc == 9)
    {
		s_r = r = atoi(argv[4]);
		s_f = f = atoi(argv[5]);
		s_alpha = alpha = atoi(argv[6]);
		s_sigma = sigma = atof(argv[7]);
		s_sigmai = sigmai = atof(argv[8]);
    } 

    variant = argv[1];
    printf ( "Testing ROB Filter - variant %s...\n", variant);

    /* Read the input image */
	in_img = read_img(argv[2]);
    noisy_img = read_img(argv[3]);
	
    /* Make sure it's an rgb image */
    if (is_gray_img(in_img))
    {
        fprintf(stderr, "Input image ( %s ) must not be grayscale !", argv[2]);
        exit(EXIT_FAILURE);
    }
       

    if (argc == 4) {
        float road = avg_road(noisy_img, 4, ROAD);
        if (road < 50)
            f = 1;
        else if (road < 80)
            f = 2;
        else
            f = 3;

        alpha = 4;
        r = round(0.017 * road + 6.34);
        sigma = round(0.215 * road + 1.5);
        sigmai = 30;
        printf ( "ROAD:%f ...\n", road);
    }

    out_img = filter_rad(variant, noisy_img, r, f, alpha, sigma, sigmai);
    elapsed_time = stop_timer(start_time);

    write_img(out_img, "out.png", FMT_PNG);
    write_img(noisy_img, "noisy.ppm", FMT_PPM);

    printf("r, f, alpha, sigma, h: %d, %d, %d, %f, %f, %f -->\n", r, f, alpha, sigma, sigmai, elapsed_time);

    calculate_snr(in_img, noisy_img, NULL);
    calculate_snr(in_img, out_img, NULL);
    printf("ROB time = %f\n", elapsed_time);

    free_img(out_img);
    free_img ( in_img );
    free_img ( noisy_img );
    return EXIT_SUCCESS;
}
