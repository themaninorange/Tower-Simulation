/*
Joseph Brown

nvcc newstructure.cu -o temp -lglut -lm -lGLU -lGL -std=c++11
*/

#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <random>

#define PI 3.141592654

#define N 30
#define PROB 0.3 
#define MAXCONNECTIONS 20

#define XWindowSize 600
#define YWindowSize 600

#define STOP_TIME 1.0
#define DT        0.005
#define MAXTRIES  50
#define MAXVEL	  0.1

#define GRAVITY 0.1 
/*
#define MASSBODY1 10.0
#define MASSBODY2 10.0*/
#define ALLMASS 10
#define DRAW 10

#define rnd( x ) (x * rand() / RAND_MAX)

using namespace std;
//std::random_device generator;
//std::uniform_real_distribution<float> unif_dist(0.0,1.0);// */

// Globals
double G_const, H_const, p_const, q_const, k_const, k_anchor, rod_proportion, dampening, timerunning;

int numblocks, blocksize, attempt;

struct stat st = {0};

struct Connection {
	double px, py, pz;	//Position
	double vx, vy, vz;	//Velocity
	double fx, fy, fz;	//Net force
	double mass, radius;
	bool anchor;		//Establishes whether each node can move

};

Connection *cnx_CPU, *cnx_GPU;

int	*numcon_CPU;
int	*beami_CPU;
double	*beamk_CPU;
double	*beaml_CPU;
bool	*fail_CPU;

int	*numcon_GPU;
int	*beami_GPU;
double	*beamk_GPU;
double	*beaml_GPU;
bool	*fail_GPU;

char timestring[16] ;
char fastnessstring[18] ;	

bool written = false;

/*
EXAMPLE:

If conn_index[0] is zero, and conn_index[1] is 6, this means that i_conns entries 0 through 5 will be 
	connections to node zero.  If conn_index[2] is 9, then i_conns entries 6 through 8 will be
	connections to node one.

i_conns will then have length equal to the total number of edges, and its entries will be which nodes
	are connected to the node associated with that index range.  If i_conns[0] through i_conns[9] are
		 1,   2,   5,   6,   7,   10,  0,   2,   5,          ...
		|    connected to node 0     | connected to node 1 | ...
	we can loop through i_conns[6] through i_conns[8] to find the connections for node 1.

i_kconst will have the same structure as i_conns, but the entries will be the spring constants associated with
	the connections in i_conns.  If i_kconst[0] through i_kconst[9] are
		 3.5, 5.5, 10,  4,   0.5, 12.1,7.2, 17,  3.2	     ...
		| ^  |					| ^  |
		  |					  |	
		  |					Value of spring constant for connection between 1 and 5.
		Value of spring constant for connection between 0 and 1.

i_lengths will also have information about connections, like i_kconst, but the entries will be the lengths of the springs.
*/


void AllocateMemory(){
	
	printf("passA1\n");
	cnx_CPU = (Connection*)malloc( sizeof(Connection) * N);
	cudaMalloc((void**)&cnx_GPU, sizeof(Connection) * N);

	numcon_CPU  = (int *)malloc(N*sizeof(int));
	printf("passA2\n");

	beami_CPU = (int    *)malloc(N * MAXCONNECTIONS * sizeof(int)   );
        beamk_CPU = (double *)malloc(N * MAXCONNECTIONS * sizeof(double));
        beaml_CPU = (double *)malloc(N * MAXCONNECTIONS * sizeof(double));
        fail_CPU  = (bool   *)malloc(N * MAXCONNECTIONS * sizeof(bool)  );


	cudaMalloc((void**)&numcon_GPU, N* sizeof(int));

        cudaMalloc((void**)&beami_GPU, N * MAXCONNECTIONS * sizeof(int)   );
        cudaMalloc((void**)&beamk_GPU, N * MAXCONNECTIONS * sizeof(double));
        cudaMalloc((void**)&beaml_GPU, N * MAXCONNECTIONS * sizeof(double));
        cudaMalloc((void**)&fail_GPU,  N * MAXCONNECTIONS * sizeof(bool)  );

	printf("passA3\n");

}

void create_beams(){

	float prob = PROB;

	srand(time(NULL));
	//srand(9001);
	
	int i,j;
	for(i = 0 ; i < N ; i++){
		numcon_CPU[i] = 0;
	}
	
	for(i = 0 ; i < N ; i++){
		for(j = i + 1; j < N ; j++){
			if(((!cnx_CPU[i].anchor) || (!cnx_CPU[j].anchor))&&
			   (numcon_CPU[i]<MAXCONNECTIONS)&&
			   (numcon_CPU[j]<MAXCONNECTIONS)&&
			   (rand() < prob*RAND_MAX)){

				beami_CPU[i*MAXCONNECTIONS + numcon_CPU[i]] = j;
					//Set which node i is being connected to.
				
				if((cnx_CPU[j].anchor)||(cnx_CPU[i].anchor)){
					beamk_CPU[i*MAXCONNECTIONS + numcon_CPU[i]] = k_anchor;
					beaml_CPU[i*MAXCONNECTIONS + numcon_CPU[i]] = 2.0;
	
				} else {
					beamk_CPU[i*MAXCONNECTIONS + numcon_CPU[i]] = rnd(100.0f) + 50.0;
					beaml_CPU[i*MAXCONNECTIONS + numcon_CPU[i]] = rnd(0.6f) + 0.2;
				} //Create values for the connection just established.


				beami_CPU[j*MAXCONNECTIONS + numcon_CPU[j]] = i;
					//Set which node j is being connected to.
				beamk_CPU[j*MAXCONNECTIONS + numcon_CPU[j]] = beamk_CPU[i*MAXCONNECTIONS + numcon_CPU[i]];
				beaml_CPU[j*MAXCONNECTIONS + numcon_CPU[j]] = beaml_CPU[i*MAXCONNECTIONS + numcon_CPU[i]];
					//Set these the same as the values for i.
				numcon_CPU[j] += 1;	
					//Increase number of connections to j by one.	
				numcon_CPU[i] += 1;	
					//Increase number of connections to i by one.
			}
		}
	}
	/*for(i = 0; i < N; i++){
		for(j = 0; j < numcon_CPU[i]; j++){
			printf("nodes connected: %d, %d\n", i, beami_CPU[i*MAXCONNECTIONS + j]);	
			printf("constant:        %.3f\n", beamk_CPU[i*MAXCONNECTIONS + j]);
			printf("length:          %.3f\n\n", beaml_CPU[i*MAXCONNECTIONS + j]);
		}
	}*/
}// */

double springmass(double k, double l){
	return(0.1*sqrt(k)*l);
}//Determines spring mass based on springiness and length.
 // This can be replaced with a realistic function later.

/*__global__ void setLimit(double* beami, double* beamk, double* beaml, double* limits, int* numcon){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i < N){
		int j;
		for(j = 0; j < numcon[i]; j++){
			limits[i*MAXCONNECTIONS + j] = 0.1*beaml[i*MAXCONNECTIONS + j];
		}
	}
}*/

void set_initial_conditions()
{
	blocksize = 1024;
	numblocks = (N - 1)/blocksize + 1;
	timerunning = 0;
	attempt = 0;
	printf("pass0\n");
	G_const = 1;
	k_const = 1;
	k_anchor = 4000;
	rod_proportion = 10;
	dampening = 20;

	int i, j;

	for(i = 0; i < N; i++){
		cnx_CPU[i].px = 2.0*i/N-1;
		cnx_CPU[i].py = sqrt(1-cnx_CPU[i].px*cnx_CPU[i].px)*sin(i);
		cnx_CPU[i].pz = sqrt(1-cnx_CPU[i].px*cnx_CPU[i].px)*cos(i);
	}	//Initialize nodes around the unit sphere
	
	for(i = 0; i < N; i++){
		cnx_CPU[i].vx = 0;
		cnx_CPU[i].vy = 0;
		cnx_CPU[i].vz = 0;
	}// */	//Initialize sphere velocities to zero

	for(i = 0; i < 6; i++){
		cnx_CPU[i].px = sin(i+1);
		cnx_CPU[i].py = -1;
		cnx_CPU[i].pz = cos(i+1);
		
		cnx_CPU[i].anchor = true;	

		cnx_CPU[i].vx = 0;
		cnx_CPU[i].vy = 0;
		cnx_CPU[i].vz = 0;
	} // initializes anchors in nearly a hexagon.  labels them anchors.

	create_beams();
	
	for(i = 0; i < N; i++){
		//cnx_CPU[i].mass = ALLMASS;
		cnx_CPU[i].mass = 0;
		for(j = 0; j < numcon_CPU[i]; j++){
			cnx_CPU[i].mass += springmass(beamk_CPU[i*MAXCONNECTIONS + j], beaml_CPU[i*MAXCONNECTIONS]);
		}// */
		printf("mass[i]: %.3f\n", cnx_CPU[i].mass);
	}	//Assigns masses to nodes based on beam mass
	
	for(i = 0; i < N; i++){
		cnx_CPU[i].radius = 0.01*pow(cnx_CPU[i].mass, 1.0/3);
	}	//Assigns radius to spheres based on cnx_CPU.mass.
	
	
	cudaMemcpy(cnx_GPU,    cnx_CPU,                   N*sizeof(Connection), cudaMemcpyHostToDevice);
	cudaMemcpy(numcon_GPU, numcon_CPU, 	 	  N*sizeof(int),        cudaMemcpyHostToDevice);
	cudaMemcpy(beami_GPU,  beami_CPU,  MAXCONNECTIONS*N*sizeof(int),        cudaMemcpyHostToDevice);
	cudaMemcpy(beamk_GPU,  beamk_CPU,  MAXCONNECTIONS*N*sizeof(double),     cudaMemcpyHostToDevice);
	cudaMemcpy(beaml_GPU,  beaml_CPU,  MAXCONNECTIONS*N*sizeof(double),     cudaMemcpyHostToDevice);


	for (i = 0; i < N; i++){
		printf("numcon[%d]: %d\n", i, numcon_CPU[i]);
	}
/*	for (i = 0; i < N; i++){
		for (j = 0; j < numcon_CPU[i] ; j++){
			if(beami_CPU[i*MAXCONNECTIONS + j] > i){
				printf("node1: %d  node2: %d  k:   %.3f  len: %.3f\n", 
					i, beami_CPU[i*MAXCONNECTIONS + j], beamk_CPU[i*MAXCONNECTIONS+j], beaml_CPU[i*MAXCONNECTIONS+j] );
			}
		}
	}*/
}

void displayText( float x, float y, float z, int r, int g, int b, const char *string ) {
	int j = strlen( string );
 
	glColor3f( r, g, b );
	glRasterPos3f( x, y, z);
	for( int i = 0; i < j; i++ ) {
		glutBitmapCharacter( GLUT_BITMAP_TIMES_ROMAN_24, string[i] );
	}
}

double fastest_cnxn(Connection *cnx){
	int i;
	double max_fastness = 0;
	for(i = 0; i < N; i++){
		max_fastness = max(max_fastness, sqrt(cnx[i].vx*cnx[i].vx+
						      cnx[i].vy*cnx[i].vy+
						      cnx[i].vz*cnx[i].vz));
	}
	return(max_fastness);
}

void draw_picture()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);

	glBegin(GL_QUADS);
		glColor3f(0.2f, 1.0f, 0.5f);     // Red
		glVertex3f( -2.0f, -1.0f, 2.0f); 
		glColor3f(0.2f, 0.2f, 0.3f);     // Blue
		glVertex3f(-2.0f, -1.0f, -2.0f);
		glColor3f(0.2f, 0.2f, 0.6f);     // Green
		glVertex3f(2.0f, -1.0f, -2.0f);
		glColor3f(0.2f, 1.0f, 0.3f);     // Green
		glVertex3f(2.0f, -1.0f, 2.0f);
	glEnd(); // draw "ground"
	
	

	int i, j;
	for(i = 0 ; i < N ; i++){
		for(j = 0; j < numcon_CPU[i] ; j++){
			if(beami_CPU[i*MAXCONNECTIONS + j] > i){
				if(fail_CPU[i*MAXCONNECTIONS + j]){
					glColor3f(0.6f, 0.0f, 0.0f);     // bludd

				} else {
					glColor3f(0.4f, 1.0f, 1.0f);     // terkwaass
				}
				glBegin(GL_LINES);
					glVertex3f(cnx_CPU[i].px, 
						   cnx_CPU[i].py,
						   cnx_CPU[i].pz);
						 //Position of i.
					glVertex3f(cnx_CPU[beami_CPU[i*MAXCONNECTIONS + j]].px, 
						   cnx_CPU[beami_CPU[i*MAXCONNECTIONS + j]].py, 
						   cnx_CPU[beami_CPU[i*MAXCONNECTIONS + j]].pz);
						 //Position of node i is connected to.
				glEnd();
			}
		}
	}	// draw connections between nodes
	
	for(i = 0 ; i < N ; i++){
		glColor3d(1.0,1.0,0.5);
		glPushMatrix();
		glTranslatef(cnx_CPU[i].px, cnx_CPU[i].py, cnx_CPU[i].pz);
		glutSolidSphere(cnx_CPU[i].radius,20,20);
		glPopMatrix();
	}	// draw intersections
	
	snprintf(timestring, 16, "time: %4.4f", STOP_TIME*attempt + timerunning);
	
	if(((int)(timerunning/DT))%10 == 0){
		snprintf(fastnessstring, 18, "max speed: %3.3f", fastest_cnxn(cnx_CPU));
	}
	glClear(GL_DEPTH_BUFFER_BIT);
	displayText( 1, -1.8, 0, 1.0, 1.0, 1.0, timestring);
	displayText( 0.59, -1.9, 0, 1.0, 1.0, 1.0, fastnessstring);
}

double dot_prod(double x1, double y1, double z1, double x2, double y2, double z2){
	return(x1*x2+y1*y2+z1*z2);
}

__device__ double dev_dot_prod(double x1, double y1, double z1, double x2, double y2, double z2){
	return(x1*x2+y1*y2+z1*z2);
}

__global__ void setForces(Connection *cnx){
	
	int i = threadIdx.x + blockDim.x*blockIdx.x;

	if(i < N){
		cnx[i].fx = 0.0;
		cnx[i].fy = -GRAVITY*cnx[i].mass;	//Throw gravity into the mix
		cnx[i].fz = 0.0;
		//printf("core %d checking in.\n", i);
	}
}

__global__ void calcForces(Connection *cnx, int *numcon, int *beami, double *beamk, double *beaml, bool *fail,  double dampening){

	int i = threadIdx.x + blockDim.x*blockIdx.x;

	if(i < N){
	
		double dvx, dvy, dvz, dpx, dpy, dpz, dD; 
		int    j, node2;
		double r, relative_v;	
	
		for(j = 0; j < numcon[i]; j++){
		//iterate through all nodes connected to this position.

			node2 = beami[i*MAXCONNECTIONS + j];
			dpx = cnx[node2].px - cnx[i].px;
			dpy = cnx[node2].py - cnx[i].py;
			dpz = cnx[node2].pz - cnx[i].pz;
			//determine relative position
        
			dvx = cnx[node2].vx - cnx[i].vx;
			dvy = cnx[node2].vy - cnx[i].vy;
			dvz = cnx[node2].vz - cnx[i].vz;
			//determine relative velocity
        
			r = sqrt(dpx*dpx+dpy*dpy+dpz*dpz);
			//magnitude of relative position
        
			relative_v = dev_dot_prod(dpx,dpy,dpz,dvx,dvy,dvz)/r;
			//magnitude of relative velocity,
			// projected onto the relative position.
			//I.e., relative velocity along the spring.
        
			dD= r - beaml[i*MAXCONNECTIONS + j];
			fail[i*MAXCONNECTIONS + j] = (10*abs(dD) > beaml[i*MAXCONNECTIONS + j]);
			//difference between relative position and default length of spring
			cnx[i].fx += (dD*beamk[i*MAXCONNECTIONS + j] + dampening*relative_v)*dpx/r;
			cnx[i].fy += (dD*beamk[i*MAXCONNECTIONS + j] + dampening*relative_v)*dpy/r;
			cnx[i].fz += (dD*beamk[i*MAXCONNECTIONS + j] + dampening*relative_v)*dpz/r;
		
			/*printf("node1.px: %.3f\nnode2.px: %.3f\ndD:       %.3f\nbeamk:    %.3f\ndampen:   %.3f\n", 
				cnx[i].px,      cnx[node2].px,  dD,             beamk[MAXCONNECTIONS*i + j],
											                  dampening*relative_v);*/	
		}			
	}
}// */

__global__ void calcVP(Connection *cnx) {

	int i = threadIdx.x + blockDim.x*blockIdx.x;
	double dt = DT;
	if(i < N){

		if(!cnx[i].anchor){
			cnx[i].vx = cnx[i].vx + cnx[i].fx*dt/cnx[i].mass;
			cnx[i].vy = cnx[i].vy + cnx[i].fy*dt/cnx[i].mass;
			cnx[i].vz = cnx[i].vz + cnx[i].fz*dt/cnx[i].mass;
		}
		
        
		if(!cnx[i].anchor){
			cnx[i].px = cnx[i].px + cnx[i].vx*dt;
			cnx[i].py = cnx[i].py + cnx[i].vy*dt;
			cnx[i].pz = cnx[i].pz + cnx[i].vz*dt;
		}
	}
} // */

void update(int value){


	if(timerunning < STOP_TIME){

		setForces<<<numblocks, blocksize>>>(cnx_GPU);
		calcForces<<<numblocks, blocksize>>>(cnx_GPU, numcon_GPU, beami_GPU, beamk_GPU, beaml_GPU, fail_GPU, dampening);
			//ARGS: (Connection *cnx, int *numcon, int *beami, double *beamk, double *beaml, double dampening)

		calcVP<<<numblocks, blocksize>>>(cnx_GPU);
		
		cudaMemcpy( cnx_CPU,  cnx_GPU, N*sizeof(Connection),                cudaMemcpyDeviceToHost);
		cudaMemcpy(fail_CPU, fail_GPU, MAXCONNECTIONS*N*sizeof(bool),       cudaMemcpyDeviceToHost);
	
		glutPostRedisplay();
		glutTimerFunc(1, update, 0);
		//printf("cnx[9] velocity: %.3f\n", cnx_CPU[9].vy);
		timerunning += DT;
	}
	else{		
		if (fastest_cnxn(cnx_CPU) < MAXVEL){
			if(!written){

				int result = -1;
				int i = 0;
				int j;
				char* filetry;
				
				while (result == -1){
					i += 1;
					asprintf(&filetry, "stablestructs1/trial%d", i);
					result = mkdir(filetry, ACCESSPERMS);
 				}
		
				char* nodepos; 
				char* nodecon;
				char* datarow;
				//char datarow[30];

				asprintf(&nodepos, "%s/nodepos.txt", filetry);
				asprintf(&nodecon, "%s/nodecon.txt", filetry);

				FILE *filecon;

				filecon = fopen(nodepos, "w+");
					fprintf(filecon, "px\tpy\tpz\n");
					for (i = 0; i < N ; i++){
						fprintf(filecon, "%2.3f\t%2.3f\t%2.3f\t%s\n", 
								 cnx_CPU[i].px, cnx_CPU[i].py, cnx_CPU[i].pz, cnx_CPU[i].anchor ? "T" : "F" );
					}
				fclose(filecon);

				filecon = fopen(nodecon, "w+");
					fprintf(filecon, "Nodes: %d\nMax Connections: %d\n\n", N, MAXCONNECTIONS);
					fprintf(filecon, "beami\n");
					for (i = 0; i < N ; i++){
						asprintf(&datarow, "");
						for (j = 0; j < numcon_CPU[i]; j++){
							asprintf(&datarow, "%s\t%d", datarow, beami_CPU[MAXCONNECTIONS*i + j]);
						}
						asprintf(&datarow, "%s\n", datarow);
						fprintf(filecon, "%s", datarow);
					}

					fprintf(filecon, "\nbeamk\n");
					for (i = 0; i < N ; i++){
						asprintf(&datarow, "");
						for (j = 0; j < numcon_CPU[i]; j++){
							asprintf(&datarow, "%s\t%.3f", datarow, beamk_CPU[MAXCONNECTIONS*i + j]);
						}
						asprintf(&datarow, "%s\n", datarow);
						fprintf(filecon, "%s", datarow);
					}

					fprintf(filecon, "\nbeaml\n");
					for (i = 0; i < N ; i++){
						asprintf(&datarow, "");
						for (j = 0; j < numcon_CPU[i]; j++){
							asprintf(&datarow, "%s\t%.3f", datarow, beaml_CPU[MAXCONNECTIONS*i + j]);
						}
						asprintf(&datarow, "%s\n", datarow);
						fprintf(filecon, "%s", datarow);
					}

					fprintf(filecon, "\nfail\n");
					for (i = 0; i < N ; i++){
						asprintf(&datarow, "");
						for (j = 0; j < numcon_CPU[i]; j++){
							asprintf(&datarow, "%s\t%s", datarow, fail_CPU[MAXCONNECTIONS*i + j] ? "T" : "F");
						}
						asprintf(&datarow, "%s\n", datarow);
						fprintf(filecon, "%s", datarow);
					}

				fclose(filecon);

 				written = true;
			}
		} else if (attempt < MAXTRIES){
			attempt += 1;
			timerunning = 0;

			glutTimerFunc(1, update, 0);
		}
	}
}

void Display(void)
{
	//gluLookAt(0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	
	draw_picture();
	glutSwapBuffers();
	glFlush();// */
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 150.0);

	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char *argv[])
{
	AllocateMemory();

	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("2 Body 3D");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);

	set_initial_conditions();
	gluLookAt(0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glutDisplayFunc(Display);
	glutTimerFunc(1, update, 0);
	glutReshapeFunc(reshape);

	glutMainLoop();

	return 0;
}

