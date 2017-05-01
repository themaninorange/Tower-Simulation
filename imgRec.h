/*
 add makePicture() after glutSwapBuffers()
 
 the first string is the output folder, which must be created before you start the application, otherwise it errors out.
 
 the next two integers are the screen size, x,y respectively.
 
 if the image is flipped, set the boolean to true, it will change it back to.
 
 this also requires C++, so please change your compiler to nvcc or g++ for compilation. 
 please also add -std=c++0x to the compile line in order to get string concat working.
 example: g++ -std=c++0x ParchieCircleWall.c -o tempCW -lglut -lm -lGLU -lGL && ./tempCW
 
 ffmpeg "compile" line. cd to record directory execute:
 ffmpeg -y -r 60 -f image2 -i ./%d.tga -c:v libx264 -qp 0 -preset ultrafast -f mp4 test.mp4
 
 tweak as necessary.
 
 please also include stb_image_write.h in the same folder as this h file, it is required for making image saving work.
 Location: https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
*/


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include <string> 

int frame = 0;

void makePicture(std::string folder,int windowX, int windowY,bool flip)
{
	unsigned char * pixels = (unsigned char*)calloc(windowX*windowY*3, sizeof(unsigned char));// new unsigned char[XWindowSize*YWindowSize*3];
	glReadPixels(0,0,(int)windowX, (int)windowY, GL_RGB, GL_UNSIGNED_BYTE, pixels);
	if(flip)
	{
	unsigned char * pixels2 = (unsigned char*)calloc(windowX*windowY*3, sizeof(unsigned char));
		for (int i = 0; i < windowX * 3; i++)
		{
			for(int j = 0; j < windowY; j++)
			{
				pixels2[i + windowX* 3 * j] = pixels[i+ windowX* 3 * (windowY - j)];
			}
		}
	pixels = pixels2;
	}
	if(!stbi_write_tga(std::string(folder+ "/" + std::to_string(frame)  + ".tga").c_str(), (int)windowX, (int)windowY, 3, pixels))
	{
		std::cout << "Error recording screenshot" << std::endl;
	}
	frame++;
}
