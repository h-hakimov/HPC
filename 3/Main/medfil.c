#include <stdio.h>
#include <iostream>
#include "MedianFilter.h"
#include <time.h>
#include <ctime>
#include "Bitmap.h"
#define TILE_SIZE 4 
#define ITERATIONS ( 1 )

int CompareBitmaps( Bitmap* inputA, Bitmap* inputB ){
	int differentpixels = 0; //Initializing the diffrerce Variable.
	if((inputA->Height() != inputB->Height()) || (inputA->Width() != inputB->Width())) // Check the condition for height and width matching.
		return -1;
	for(int height=1; height<inputA->Height()-1; height++){
		for(int width=1; width<inputA->Width()-1; width++){
			if(inputA->GetPixel(width, height) != inputB->GetPixel(width, height))
				differentpixels++; // increment the differences.
		}
	}
	return differentpixels;
}

void MedianFilterCPU( Bitmap* image, Bitmap* outputImage )
{
	unsigned char filterVector[9] = {0,0,0,0,0,0,0,0,0}; //Taking the filter initialization.
	for(int row=0; row<image->Height(); row++){
		for(int col=0; col<image->Width(); col++){
			if((row==0) || (col==0) || (row==image->Height()-1) || (col==image->Width()-1)) //Check the boundry condition.
				outputImage->SetPixel(col, row, 0);
			else {
				for (int x = 0; x < WINDOW_SIZE; x++) {
					for (int y = 0; y < WINDOW_SIZE; y++){
						filterVector[x*WINDOW_SIZE+y] = image->GetPixel( (col+y-1),(row+x-1)); //Fill the Filter Vector
					}
				}



				for (int i = 0; i < 9; i++) {
					for (int j = i + 1; j < 9; j++) {
						if (filterVector[i] > filterVector[j]) {
							char tmp = filterVector[i];
							filterVector[i] = filterVector[j];
							filterVector[j] = tmp;
						}
					}
				}
				outputImage->SetPixel(col, row, filterVector[4]); //Finally assign value to output pixels
			}
		}
	}
}

int main()
{
	Bitmap* originalImage = new Bitmap();
	Bitmap* resultImageCPU = new Bitmap();

	float tcpu; //timing variables.
	clock_t cpu_start, end;
	

	originalImage->Load("Lab/image22.bmp");
	resultImageCPU->Load("Lab/image22.bmp");
	std::cout << "Operating on a " << originalImage->Width() << " x " << originalImage->Height() << " image..." << std::endl;


	cpu_start = clock(); //Stat the clock
	for (int i = 0; i < ITERATIONS; i++)
	{
		MedianFilterCPU(originalImage, resultImageCPU);
	}
	end = clock();
	tcpu = ((float)(end-cpu_start) + 1) * 1000 / (float)CLOCKS_PER_SEC/ITERATIONS;
	std::cout<< "CPU time " << tcpu<<"ms"<<std::endl;
	


	resultImageCPU->Save("Lab/image22_cpu.bmp");
	
}