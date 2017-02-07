/*
Joseph Brown
Homework1

CPU version of problem 2.
*/

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int N = 20;
	//Sets N to the default value of 20.

float *X_CPU,  *Y_CPU,  *Z_CPU;
float *vX_CPU, *vY_CPU, *vZ_CPU;
float *aX_CPU, *aY_CPU, *aZ_CPU;
	//Points to memory for these three variables globally.
	//Declaring this in AllocateMemory() would result in
	// these values being local to that function.

void AllocateMemory(){
	X_CPU = (float*)malloc(N*sizeof(float));
	Y_CPU = (float*)malloc(N*sizeof(float));
	Z_CPU = (float*)malloc(N*sizeof(float));
}	//Saves the appropriate memory chunk for later use.
	//References the globally defined variables.

void Initialize(){
	int i;
	for(i = 0; i < N; i++){
		X_CPU[i] = 2.0*i/N-1;
		Y_CPU[i] = sqrt(1-X_CPU[i]*X_CPU[i])*sin(i);
		Z_CPU[i] = sqrt(1-X_CPU[i]*X_CPU[i])*cos(i);
	}	//Sets these arrays to the values 1..N.
}

void CleanUp(float  *X_CPU, float  *Y_CPU, float  *Z_CPU,
	     float *vX_CPU, float *vY_CPU, float *vZ_CPU,
	     float *aX_CPU, float *aY_CPU, float *aZ_CPU){
	free(X_CPU);
	free(Y_CPU);
	free(Z_CPU);

	free(vX_CPU);
	free(vY_CPU);
	free(vZ_CPU);

	free(aX_CPU);
	free(aY_CPU);
	free(aZ_CPU);
}	//Frees the memory for the three relevant global variables.

void VectorAddition(float *X, float *Y, float *Z, int n){
	int i;
	for(i = 0; i < n; i++){
		Z[i] = X[i] + Y[i];
	}
}	//Takes the component-wise sum of the first n 
	// values of two vectors, A and B, and stores them in the
	// corresponding values of a third vector, C.

void CalcAccel(float *X,  float *Y,  float *Z, 
	       float *aX, float *aY, float *aZ, int n){
	int i, j;
	float G = 1;
	float m2 = 1;

	for(i = 0; i < n ; i++){
		aX[i] = 0;
		aY[i] = 0;
		aZ[i] = 0;
		for(j = 0; j < n ; j++){
			if(i != j){
				aX[i] = aX[i] + G*m2/pow(X[i] - X[j], 2);
				aY[i] = aY[i] + G*m2/pow(Y[i] - Y[j], 2);
				aZ[i] = aZ[i] + G*m2/pow(Z[i] - Z[j], 2);
			}
		}	//Loop through each *other* particle to sum forces.
	}	//Loop through each particle to calculate new accel.
}

void CalcVeloc(float *aX, float *aY, float *aZ,
	       float *vX, float *vY, float *vZ, dt, n){
	int i;
	for(i = 0; i < n ; i++){
		vX[i] = vX[i] + aX[i]*dt;
		vY[i] = vY[i] + aY[i]*dt;
		vZ[i] = vZ[i] + aZ[i]*dt;
	}
	
}

void CalcPosit(float *vX, float *vY, float *vZ,
	       float  *X, float  *Y, float  *Z, dt, n){
	int i;
	for(i = 0; i < n ; i++){
		X[i] = X[i] + vX[i]*dt;
		Y[i] = Y[i] + vY[i]*dt;
		Z[i] = Z[i] + vZ[i]*dt;
	}
}

int main(int argc, char *argv[]){
	//I would like this program to accept command line
	// arguments.  Simply run "./VectorAdditionCPU.cu #"
	// to run the same program with a different parameter.
	
	if(argc == 2){
		char *ptr;
		N = strtol(argv[1], &ptr, 10);
	}
	else if(argc > 2){
		printf("One or zero arguments expected.");
		return(1);
	}

	AllocateMemory();
	Initialize();
//	gettimeofday(&start,NULL); // */ 
//	VectorAddition(X_CPU, Y_CPU, Z_CPU, N);
		//
/*	gettimeofday(&end, NULL);
	float time = (end.tv_sec*1000000 + end.tv_usec*1) - 
		     (start.tv_sec*1000000 + start.tv_usec*1);
		     //tv_sec is in seconds, while tv_usec
		     // is in microseconds, so they need to
		     // be scaled appropriately.
		     //Then, it's a matter of subtracting
		     // the value of the start from the
		     // value of the end.
	printf("CPU Time in milliseconds= %.10f\n", (time/1000.0));// */

	int i;
	for(i = 0; i < 5; i++){
		printf("X[%d] = %.5f   Y[%d] = %.5f   Z[%d] = %.5f\n",
			  i,    X_CPU[i], i,    Y_CPU[i], i,    Z_CPU[i]);
	}	
	for(int i = 5; i < N-1; i++){
		printf("A[%d] = %.5f   B[%d] = %.5f   C[%d] = %.5f\n",
			  i,   A_CPU[i], i,    B_CPU[i], i,    C_CPU[i]);
	}// */
	printf("...\n");// */
	printf("X[%d] = %.5f   Y[%d] = %.5f   Z[%d] = %.5f\n",
		  N-1,  X_CPU[N-1],N-1, Y_CPU[N-1],N-1, Z_CPU[N-1]);	
	CleanUp(X_CPU, Y_CPU, Z_CPU);

	return(0);
}


// Joseph Brown
// Mathematical Models, SPRING 2017
// 1-22-2017
