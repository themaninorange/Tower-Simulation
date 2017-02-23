/*
Joseph Brown

nvcc simintersections.cu -o temp -lglut -lm -lGLU -lGL -std=c++11
*/

#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <random>

#define PI 3.141592654

#define N 400
#define MAXCONNECTIONS 20

#define XWindowSize 600
#define YWindowSize 600

#define STOP_TIME 1000.0
#define DT        0.005

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
double G_const, H_const, p_const, q_const, k_const, k_anchor, rod_proportion, dampening;

int numblocks, blocksize;

struct Connection {
	double px, py, pz;	//Position
	double vx, vy, vz;	//Velocity
	double fx, fy, fz;	//Net force
	double mass, radius;
	bool anchor;		//Establishes whether each node can move

};

Connection *cnx_CPU, *cnx_GPU;

int	*conn_index_CPU;	//List of where to look in i_conns for connections.
int 	*i_conns_CPU;		//List of nodes connected at appropriate index.
double 	*i_kconst_CPU;		//List of "springiness" constant associated with the above connections.
double  *i_lengths_CPU;		//List of default lengths for the above connections.*/

int 	*conn_index_GPU;	//List of where to look in i_conns for connections.
int 	*i_conns_GPU;		//List of nodes connected at appropriate index.
double 	*i_kconst_GPU;		//List of "springiness" constant associated with the above connections.
double  *i_lengths_GPU;		//List of default lengths for the above connections.*/

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
	cnx_CPU = (Connection*)malloc( sizeof(Connection) * N);
	cudaMalloc((void**)&cnx_GPU, sizeof(Connection) * N);
	conn_index_CPU = (int*)malloc(sizeof(int) * (N+1));
	cudaMalloc((void**)&conn_index_GPU, sizeof(int) * (N+1));

	

	if(N < 20){
		  i_conns_CPU =    (int*)malloc(N*N*sizeof(int));
		 i_kconst_CPU = (double*)malloc(N*N*sizeof(double));
		i_lengths_CPU = (double*)malloc(N*N*sizeof(double));	

		cudaMalloc(  (void**)&i_conns_GPU, sizeof(int) * N * N);
		cudaMalloc( (void**)&i_kconst_GPU, sizeof(double) * N * N);
		cudaMalloc((void**)&i_lengths_GPU, sizeof(double) * N * N);// */
	}
	else{
		  i_conns_CPU =    (int*)malloc(20*N*sizeof(int));
		 i_kconst_CPU = (double*)malloc(20*N*sizeof(double));
		i_lengths_CPU = (double*)malloc(20*N*sizeof(double));

		cudaMalloc(  (void**)&i_conns_GPU, sizeof(int) * 20 * N);
		cudaMalloc( (void**)&i_kconst_GPU, sizeof(double) * 20 * N);
		cudaMalloc((void**)&i_lengths_GPU, sizeof(double) * 20 * N);// */

	}

}

void create_connections(){

	int index = 0;
	int numconns;
	float prob = 0.02;

	srand(time(NULL));
	
	int i,j;
	conn_index_CPU[0] = 0;

	for(i = 0 ; i < N ; i++){
		numconns = 0;
		for(j = i+1; j < N ; j++){
			if(!cnx_CPU[i].anchor || !cnx_CPU[j].anchor){
				if((numconns < MAXCONNECTIONS) && (rand() < prob*RAND_MAX)){
					i_conns_CPU[index] = j;
					i_kconst_CPU[index] = rnd(10.0) + 1;
					i_lengths_CPU[index] = rnd(0.1) + 0.2;
					index += 1;
					numconns += 1;
				}
			}
		}
		conn_index_CPU[i+1] = conn_index_CPU[i] + numconns;
		//The next index should begin after all of the connections this node has.
	} 

}// */

void set_initial_conditions()
{
	blocksize = 1024;
	numblocks = (N - 1)/blocksize + 1;
	
	create_connections();
	G_const = 1;
	k_const = 1;
	k_anchor = 4000;
	rod_proportion = 10;
	dampening = 0.3;
	int i, j;
	for(i = 0; i < N; i++){
		cnx_CPU[i].mass = ALLMASS;
	}	//Assigns masses to spheres.
	
	for(i = 0; i < N; i++){
		cnx_CPU[i].px = 2.0*i/N-1;
		cnx_CPU[i].py = sqrt(1-cnx_CPU[i].px*cnx_CPU[i].px)*sin(i);
		cnx_CPU[i].pz = sqrt(1-cnx_CPU[i].px*cnx_CPU[i].px)*cos(i);
	}	//Initialize spheres around the unit sphere
	
	/*for(i = 0; i < N; i++){
		cnx_CPU[i].vx = cnx_CPU[i].py + 0.5;
		cnx_CPU[i].vy = cnx_CPU[i].pz + 0.5;
		cnx_CPU[i].vz = cnx_CPU[i].px + 0.5;
	}// */	//Initialize sphere velocities to something nonzero

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

		for(j = conn_index_CPU[i]; j < conn_index_CPU[i+1]; j++){


			i_lengths_CPU[j] = 2.0;
			i_kconst_CPU[j] = k_anchor;
		}		
	}
	

	
	for(i = 0; i < N; i++){
		cnx_CPU[i].radius = 0.01*pow(cnx_CPU[i].mass, 1.0/3);
	}	//Assigns radius to spheres based on cnx_CPU.mass.
	
	
	/*cnx_CPU.px[0] = 0;
	cnx_CPU.py[0] = 0;
	cnx_CPU.pz[0] = 0;// */
	
	/*for(i = 0; i < N; i++){
		cnx_CPU[i].vx = cnx_CPU[i].py*3;
		cnx_CPU[i].vy = -cnx_CPU[i].pz*3;
		cnx_CPU[i].vz = cnx_CPU[i].py-cnx_CPU[i].px;
	}// */	//Initialize sphere velocities to something nonzero
	

	
	/*cnx_CPU[0].vx = 0;
	cnx_CPU[0].vy = 0;
	cnx_CPU[0].vz = 0;
	*/
	
	//Note: Forces are not initialized as they are set to zero at each timestep.

	cudaMemcpy(conn_index_GPU, conn_index_CPU, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cnx_GPU, cnx_CPU, N*sizeof(Connection), cudaMemcpyHostToDevice);
	
	if(N < 20){
		
		cudaMemcpy(  i_conns_GPU,   i_conns_CPU, N*N*sizeof(int),    cudaMemcpyHostToDevice);
		cudaMemcpy( i_kconst_GPU,  i_kconst_CPU, N*N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(i_lengths_GPU, i_lengths_CPU, N*N*sizeof(double), cudaMemcpyHostToDevice);

	}
	else{

		cudaMemcpy(  i_conns_GPU,   i_conns_CPU, 20*N*sizeof(int),    cudaMemcpyHostToDevice);
		cudaMemcpy( i_kconst_GPU,  i_kconst_CPU, 20*N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(i_lengths_GPU, i_lengths_CPU, 20*N*sizeof(double), cudaMemcpyHostToDevice);
	}
	
	cudaDeviceSynchronize();

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
		for(j = conn_index_CPU[i]; j < conn_index_CPU[i+1] ; j++){
			glBegin(GL_LINES);
				glVertex3f(cnx_CPU[i].px, cnx_CPU[i].py, cnx_CPU[i].pz);
				glVertex3f(cnx_CPU[i_conns_CPU[j]].px, cnx_CPU[i_conns_CPU[j]].py, cnx_CPU[i_conns_CPU[j]].pz);
			glEnd();
		}
	}	// draw connections between nodes
	
	for(i = 0 ; i < N ; i++){
		glColor3d(1.0,1.0,0.5);
		glPushMatrix();
		glTranslatef(cnx_CPU[i].px, cnx_CPU[i].py, cnx_CPU[i].pz);
		glutSolidSphere(cnx_CPU[i].radius,20,20);
		glPopMatrix();
	}	// draw intersections
}

double dot_prod(double x1, double y1, double z1, double x2, double y2, double z2){
	return(x1*x2+y1*y2+z1*z2);
}

__device__ double dev_dot_prod(double x1, double y1, double z1, double x2, double y2, double z2){
	return(x1*x2+y1*y2+z1*z2);
}

__global__ void stepForward(Connection *cnx, int *conns,  double *kconst,
			 //(cnx_GPU,         i_conns_GPU, i_kconst_GPU, 

			    double *lengths, int *index,     double dampening) {
			 // i_lengths_GPU,   conn_index_GPU, dampening)

	int pos = threadIdx.x + blockDim.x*blockIdx.x;

	if(pos < N){
		double dvx, dvy, dvz, dpx, dpy, dpz, dD; 
		double dt = DT;
		int    i,j, node2;
		double r, relative_v;	
		dt = DT;
        
		cnx[pos].fx = 0.0;
		cnx[pos].fy = -GRAVITY;	//Throw gravity into the mix
		cnx[pos].fz = 0.0;
        
		__syncthreads();
		
		for(j = index[pos]; j < index[pos+1]; j++){
		//iterate through all nodes connected to this position.
        
			node2 = conns[j];
        
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
        
			dD= r - lengths[j];
			//difference between relative position and default length of spring
			
			cnx[i].fx = cnx[i].fx + (dD*kconst[j] + dampening*relative_v)*dpx/r;
			cnx[i].fy = cnx[i].fy + (dD*kconst[j] + dampening*relative_v)*dpy/r;
			cnx[i].fz = cnx[i].fz + (dD*kconst[j] + dampening*relative_v)*dpz/r;
			//adds the appropriate forces from this particle interaction to i
		                                                                 
			cnx[node2].fx = cnx[node2].fx - (dD*kconst[j] + dampening*relative_v)*dpx/r;
			cnx[node2].fy = cnx[node2].fy - (dD*kconst[j] + dampening*relative_v)*dpy/r;
			cnx[node2].fz = cnx[node2].fz - (dD*kconst[j] + dampening*relative_v)*dpz/r; // 
			//adds the appropriate forces from this particle interaction to i
		}			
		__syncthreads(); 
        
		if(!cnx[i].anchor){
			cnx[i].vx = cnx[i].vx + cnx[i].fx*dt;
			cnx[i].vy = cnx[i].vy + cnx[i].fy*dt;
			cnx[i].vz = cnx[i].vz + cnx[i].fz*dt;
		}
		
        
		if(!cnx[i].anchor){
			cnx[i].px = cnx[i].px + cnx[i].vx*dt;
			cnx[i].py = cnx[i].py + cnx[i].vy*dt;
			cnx[i].pz = cnx[i].pz + cnx[i].vz*dt;
		}
	}
} // */

void n_body()
{
	double dvx, dvy, dvz, dpx, dpy, dpz, dD; 
	double dt = DT;
	int    i,j, node2;
	double r, relative_v;
	//int debug_int = 0;	

	for(i=0; i<N; i++)
	{
		cnx_CPU[i].fx = 0.0;
		cnx_CPU[i].fy = 0.0;
		cnx_CPU[i].fz = 0.0;
	}

	//Get forces
	for(i = 0 ; i < N ; i++){
		for(j = conn_index_CPU[i]; j < conn_index_CPU[i+1] ; j++){
		//iterate through all nodes connected to node i

			node2 = i_conns_CPU[j];
			
			dpx = cnx_CPU[node2].px - cnx_CPU[i].px;
			dpy = cnx_CPU[node2].py - cnx_CPU[i].py;
			dpz = cnx_CPU[node2].pz - cnx_CPU[i].pz;
			//determine relative position

			dvx = cnx_CPU[node2].vx - cnx_CPU[i].vx;
			dvy = cnx_CPU[node2].vy - cnx_CPU[i].vy;
			dvz = cnx_CPU[node2].vz - cnx_CPU[i].vz;
			//determine relative velocity

			r = sqrt(dpx*dpx+dpy*dpy+dpz*dpz);
			//magnitude of relative position

			relative_v = dot_prod(dpx,dpy,dpz,dvx,dvy,dvz)/r;
			//magnitude of relative velocity,
			// projected onto the relative position.
			//I.e., relative velocity along the spring.

			dD= r - i_lengths_CPU[j];
			//difference between relative position and default length of spring
			
			cnx_CPU[i].fx = cnx_CPU[i].fx + (dD*i_kconst_CPU[j] + dampening*relative_v)*dpx/r;
			cnx_CPU[i].fy = cnx_CPU[i].fy + (dD*i_kconst_CPU[j] + dampening*relative_v)*dpy/r;
			cnx_CPU[i].fz = cnx_CPU[i].fz + (dD*i_kconst_CPU[j] + dampening*relative_v)*dpz/r;
			//adds the appropriate forces from this particle interaction to i
		                                                                 
			cnx_CPU[node2].fx = cnx_CPU[node2].fx - (dD*i_kconst_CPU[j] + dampening*relative_v)*dpx/r;
			cnx_CPU[node2].fy = cnx_CPU[node2].fy - (dD*i_kconst_CPU[j] + dampening*relative_v)*dpy/r;
			cnx_CPU[node2].fz = cnx_CPU[node2].fz - (dD*i_kconst_CPU[j] + dampening*relative_v)*dpz/r;
			//adds the appropriate forces from this particle interaction to i
		}

		cnx_CPU[i].fy = cnx_CPU[i].fy - GRAVITY;
	}
	
	//Update velocity
	for(i = 0; i < N; i++){
		if(!cnx_CPU[i].anchor){
			cnx_CPU[i].vx = cnx_CPU[i].vx + cnx_CPU[i].fx*dt;
			cnx_CPU[i].vy = cnx_CPU[i].vy + cnx_CPU[i].fy*dt;
			cnx_CPU[i].vz = cnx_CPU[i].vz + cnx_CPU[i].fz*dt;
		}
	}// 
	
	//Move elements
	for(i=0; i<N; i++)
	{
		if(!cnx_CPU[i].anchor){
			cnx_CPU[i].px = cnx_CPU[i].px + cnx_CPU[i].vx*dt;
			cnx_CPU[i].py = cnx_CPU[i].py + cnx_CPU[i].vy*dt;
			cnx_CPU[i].pz = cnx_CPU[i].pz + cnx_CPU[i].vz*dt;
		}
	}
	
}// */

void update(int value){


	//n_body();
	//cudaMemcpy(cnx_GPU, cnx_CPU, N*sizeof(Connection), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	stepForward<<<numblocks, blocksize>>>(cnx_GPU, i_conns_GPU, i_kconst_GPU, i_lengths_GPU, conn_index_GPU, dampening);
	cudaDeviceSynchronize();
	cudaMemcpy(cnx_CPU, cnx_GPU, N*sizeof(Connection), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	glutPostRedisplay();
	glutTimerFunc(1, update, 0);
	printf("cnx[9] velocity: %.3f\n", cnx_CPU[9].vy);

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

