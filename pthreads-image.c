#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include "image.h"
#include <pthread.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define THREAD_COUNT 8

//An array of kernel matrices to be used for image convolution.  
//The indexes of these match the enumeration from the header file. ie. algorithms[BLUR] returns the kernel corresponding to a box blur.
Matrix algorithms[]={
    {{0,-1,0},{-1,4,-1},{0,-1,0}},
    {{0,-1,0},{-1,5,-1},{0,-1,0}},
    {{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0}},
    {{1.0/16,1.0/8,1.0/16},{1.0/8,1.0/4,1.0/8},{1.0/16,1.0/8,1.0/16}},
    {{-2,-1,0},{-1,1,1},{0,1,2}},
    {{0,0,0},{0,1,0},{0,0,0}}
};


//getPixelValue - Computes the value of a specific pixel on a specific channel using the selected convolution kernel
//Paramters: srcImage:  An Image struct populated with the image being convoluted
//           x: The x coordinate of the pixel
//          y: The y coordinate of the pixel
//          bit: The color channel being manipulated
//          algorithm: The 3x3 kernel matrix to use for the convolution
//Returns: The new value for this x,y pixel and bit channel
uint8_t getPixelValue(Image* srcImage,int x,int y,int bit,Matrix algorithm){
    int px,mx,py,my,i,span;
    span=srcImage->width*srcImage->bpp;
    // for the edge pixes, just reuse the edge pixel
    px=x+1; py=y+1; mx=x-1; my=y-1;
    if (mx<0) mx=0;
    if (my<0) my=0;
    if (px>=srcImage->width) px=srcImage->width-1;
    if (py>=srcImage->height) py=srcImage->height-1;
    uint8_t result=
        algorithm[0][0]*srcImage->data[Index(mx,my,srcImage->width,bit,srcImage->bpp)]+
        algorithm[0][1]*srcImage->data[Index(x,my,srcImage->width,bit,srcImage->bpp)]+
        algorithm[0][2]*srcImage->data[Index(px,my,srcImage->width,bit,srcImage->bpp)]+
        algorithm[1][0]*srcImage->data[Index(mx,y,srcImage->width,bit,srcImage->bpp)]+
        algorithm[1][1]*srcImage->data[Index(x,y,srcImage->width,bit,srcImage->bpp)]+
        algorithm[1][2]*srcImage->data[Index(px,y,srcImage->width,bit,srcImage->bpp)]+
        algorithm[2][0]*srcImage->data[Index(mx,py,srcImage->width,bit,srcImage->bpp)]+
        algorithm[2][1]*srcImage->data[Index(x,py,srcImage->width,bit,srcImage->bpp)]+
        algorithm[2][2]*srcImage->data[Index(px,py,srcImage->width,bit,srcImage->bpp)];
    return result;
}

struct thread_data{ 
   int thread_id;
   int rowStart;
   int rowEnd;
   Image* srcImage;
   Image* destImage;
   Matrix algorithm;
};
struct thread_data thread_data_array[THREAD_COUNT];

// applyFiler: Applies a kernel matrix to a portion of an image in a specific thread
// Parameters: threadarg: Contains the various data elements required for this thread
// Returns: Nothing
void* applyFilter(void* threadarg) {
    struct thread_data* data;
    data = (struct thread_data*) threadarg;
    
    int thread_id = data->thread_id;
    int rowStart = data->rowStart;
    int rowEnd = data->rowEnd;
    Image* srcImage = data->srcImage;
    Image* destImage = data->destImage;

    int row, bit, pix;
    printf("Thread [%d] processing rows [%d] to [%d]\n", thread_id, rowStart, rowEnd);
    for (row = rowStart; row <= rowEnd && row < srcImage->height; row++) {
        for (pix=0;pix<srcImage->width;pix++) {
            for (bit=0;bit<srcImage->bpp;bit++) {
                destImage->data[Index(pix,row,srcImage->width,bit,srcImage->bpp)]=getPixelValue(srcImage,pix,row,bit,data->algorithm);
            }
        }
    }
    return NULL;
}

//convolute:  Applies a kernel matrix to an image using multiple threads
//Parameters: srcImage: The image being convoluted
//            destImage: A pointer to a  pre-allocated (including space for the pixel array) structure to receive the convoluted image.  It should be the same size as srcImage
//            algorithm: The kernel matrix to use for the convolution
//Returns: Nothing
void convolute(Image* srcImage,Image* destImage,Matrix algorithm){
    pthread_t* thread_handles;
    thread_handles = (pthread_t*)malloc(THREAD_COUNT*sizeof(pthread_t));
    
    printf("Executing pthreads version against image with height [%d] and width [%d]\n", srcImage->height, srcImage->width);
    int chunkSize = (srcImage->height + (THREAD_COUNT - 1)) / THREAD_COUNT;
    int i,j,k;
    for(i = 0; i < THREAD_COUNT; i++){
        int rowStart = chunkSize * i;
        int rowEnd = rowStart + chunkSize - 1;

        thread_data_array[i].thread_id = i;
        thread_data_array[i].rowStart = rowStart;
        thread_data_array[i].rowEnd = rowEnd;
        thread_data_array[i].srcImage = srcImage;
        thread_data_array[i].destImage = destImage;
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 3; k++) {
                thread_data_array[i].algorithm[j][k] = algorithm[j][k];
            }
        }

        pthread_create(&thread_handles[i], NULL, &applyFilter,(void*)&thread_data_array[i]);
        printf("Created thread [%d]\n", i);
    }
    for (i = 0; i < THREAD_COUNT; i++){
        pthread_join(thread_handles[i], NULL);
        printf("Finished thread [%d]\n", i);
    }
    free(thread_handles);
}

//Usage: Prints usage information for the program
//Returns: -1
int Usage(){
    printf("Usage: image <filename> <type>\n\twhere type is one of (edge,sharpen,blur,gauss,emboss,identity)\n");
    return -1;
}

//GetKernelType: Converts the string name of a convolution into a value from the KernelTypes enumeration
//Parameters: type: A string representation of the type
//Returns: an appropriate entry from the KernelTypes enumeration, defaults to IDENTITY, which does nothing but copy the image.
enum KernelTypes GetKernelType(char* type){
    if (!strcmp(type,"edge")) return EDGE;
    else if (!strcmp(type,"sharpen")) return SHARPEN;
    else if (!strcmp(type,"blur")) return BLUR;
    else if (!strcmp(type,"gauss")) return GAUSE_BLUR;
    else if (!strcmp(type,"emboss")) return EMBOSS;
    else return IDENTITY;
}

//main:
//argv is expected to take 2 arguments.  First is the source file name (can be jpg, png, bmp, tga).  Second is the lower case name of the algorithm.
int main(int argc,char** argv){
    long t1,t2,t3,t4;
    t1=time(NULL);

    stbi_set_flip_vertically_on_load(0); 
    if (argc!=3) return Usage();
    char* fileName=argv[1];
    if (!strcmp(argv[1],"pic4.jpg")&&!strcmp(argv[2],"gauss")){
        printf("You have applied a gaussian filter to Gauss which has caused a tear in the time-space continum.\n");
    }
    enum KernelTypes type=GetKernelType(argv[2]);

    Image srcImage,destImage,bwImage;   
    srcImage.data=stbi_load(fileName,&srcImage.width,&srcImage.height,&srcImage.bpp,0);
    if (!srcImage.data){
        printf("Error loading file %s.\n",fileName);
        return -1;
    }
    destImage.bpp=srcImage.bpp;
    destImage.height=srcImage.height;
    destImage.width=srcImage.width;
    destImage.data=malloc(sizeof(uint8_t)*destImage.width*destImage.bpp*destImage.height);
    convolute(&srcImage,&destImage,algorithms[type]);
    t2=time(NULL);
    printf("Image processing took %ld seconds\n",t2-t1);

    stbi_write_png("output.png",destImage.width,destImage.height,destImage.bpp,destImage.data,destImage.bpp*destImage.width);
    t3=time(NULL);
    printf("Image write took %ld seconds\n",t3-t2);

    stbi_image_free(srcImage.data);
    free(destImage.data);
    t4=time(NULL);
    printf("Total execution took %ld seconds\n",t4-t1);
    return 0;
}