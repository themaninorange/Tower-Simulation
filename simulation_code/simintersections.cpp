/*
Joseph Brown

g++ simintersections.cpp -o temp -lglut -lm -lGLU -lGL -std=c++11
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
#define MAXCONNECTIONS 10

#define XWindowSize 600
#define YWindowSize 600

#define STOP_TIME 1000.0
#define DT        0.05

#define GRAVITY 1.0 
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
double px[N], py[N], pz[N], vx[N], vy[N], vz[N], fx[N], fy[N], fz[N], mass[N], radii[N], G_const, H_const, p_const, q_const, k_const, k_anchor, rod_proportion, dampening;

int 	conn_index[N+1];//List of where to look in i_conns for connections.
int 	*i_conns;	//List of nodes connected at appropriate index.
double 	*i_kconst;	//List of "springiness" constant associated with the above connections.
double  *i_lengths;	//List of default lengths for the above connections.*/
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

bool anchor[N];	//Establishes whether each node is anchored.

void AllocateMemory(){
	if(N < 20){
		i_conns = (int*)malloc(N*N*sizeof(int));
		i_kconst = (double*)malloc(N*N*sizeof(double));
		i_lengths = (double*)malloc(N*N*sizeof(double));

	}
	else{
		i_conns = (int*)malloc(20*N*sizeof(int));
		i_kconst = (double*)malloc(20*N*sizeof(double));
		i_lengths = (double*)malloc(20*N*sizeof(double));
	}
}

void create_connections(){

	printf("pass0");
	int index = 0;
	int numconns;
	float prob = 0.022;
	double kconst, length;
	
	printf("pass1");
	srand(time(NULL));
	
	int i,j;
	printf("pass2");
	for(i = 0 ; i < N ; i++){
		numconns = 0;
		for(j = i+1; j < N ; j++){
			if(!anchor[i] || !anchor[j]){
				if(rand() < prob*RAND_MAX){
					i_conns[index] = j;
					i_kconst[index] = rnd(10.0)+ 1;
					i_lengths[index] = rnd(0.1)+ 0.2;
					index += 1;
					numconns += 1;
				}
			}
		}
		conn_index[i+1] = conn_index[i] + numconns;
		//The next index should begin after all of the connections this node has.
	} 
	/*for(i = 0 ; i < N ; i++){
		for(j = conn_index[i]; j < conn_index[i+1] ; j++){
			printf("node1: %d\nnode2: %d\n\n", i, i_conns[j]);
		}
	}// */
	
}// */

void set_initial_conditions()
{
	create_connections();
	G_const = 1;
	k_const = 1;
	k_anchor = 4000;
	rod_proportion = 10;
	dampening = 0.03;
	int i, j;
	for(i = 0; i < N; i++){
		mass[i] = ALLMASS;
	}	//Assigns masses to spheres.
	
	//mass[0] = 1000;
	for(i = 0; i < N; i++){
		px[i] = 2.0*i/N-1;
		py[i] = sqrt(1-px[i]*px[i])*sin(i);
		pz[i] = sqrt(1-px[i]*px[i])*cos(i);
	}	//Initialize spheres around the unit sphere
	
	
		
	for(i = 0; i < 6; i++){
		px[i] = sin(i+1);
		py[i] = -1;
		pz[i] = cos(i+1);
		
		anchor[i] = true;	

		vx[i] = 0;
		vy[i] = 0;
		vz[i] = 0;

		for(j = conn_index[i]; j < conn_index[i+1]; j++){
			i_lengths[j] = 2.0;
			i_kconst[j] = 500;
		}		
	}
	
	for(i = 0; i < N; i++){
		radii[i] = 0.01*pow(mass[i], 1.0/3);
	}	//Assigns radii to spheres based on mass.
	
	
	/*px[0] = 0;
	py[0] = 0;
	pz[0] = 0;// */
	
	/*for(i = 0; i < N; i++){
		vx[i] = py[i] + 0.5;
		vy[i] = pz[i] + 0.5;
		vz[i] = px[i] + 0.5;
	}// */	//Initialize sphere velocities to something nonzero
	
	/*for(i = 0; i < N; i++){
		vx[i] = py[i]*3;
		vy[i] = -pz[i]*3;
		vz[i] = py[i]-px[i];
	}// */	//Initialize sphere velocities to something nonzero
	
	/*for(i = 0; i < N; i++){
		vx[i] = 0;
		vy[i] = 0;
		vz[i] = 0;
	}// */	//Initialize sphere velocities to zero
	
	vx[0] = 0;
	vy[0] = 0;
	vz[0] = 0;
	
	//Note: Forces are not initialized as they are set to zero at each timestep.
	
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
		for(j = conn_index[i]; j < conn_index[i+1] ; j++){
			glBegin(GL_LINES);
				glVertex3f(px[i], py[i], pz[i]);
				glVertex3f(px[i_conns[j]], py[i_conns[j]], pz[i_conns[j]]);
			glEnd();
		}
	}	// draw connections between nodes
	
	for(i = 0 ; i < N ; i++){
		glColor3d(1.0,1.0,0.5);
		glPushMatrix();
		glTranslatef(px[i], py[i], pz[i]);
		glutSolidSphere(radii[i],20,20);
		glPopMatrix();
	}	// draw intersections
}

double dot_prod(double x1, double y1, double z1, double x2, double y2, double z2){
	return(x1*x2+y1*y2+z1*z2);
}

int n_body()
{
	double fx[N], fy[N], fz[N], dvx, dvy, dvz, dpx, dpy, dpz, rp1, rq1, feynFac, dD; 
	double dt = DT;
	double time = 0.0;
	int    i,j, node2;
	double r, relative_v;	
	dt = DT;

	for(i=0; i<N; i++)
	{
		fx[i] = 0.0;
		fy[i] = 0.0;
		fz[i] = 0.0;
	}

	//Get forces
	for(i = 0 ; i < N ; i++){
		for(j = conn_index[i]; j < conn_index[i+1] ; j++){
		//iterate through all nodes connected to node i

			node2 = i_conns[j];
			
			dpx = px[node2] - px[i];
			dpy = py[node2] - py[i];
			dpz = pz[node2] - pz[i];
			//determine relative position

			dvx = vx[node2] - vx[i];
			dvy = vy[node2] - vy[i];
			dvz = vz[node2] - vz[i];
			//determine relative velocity

			r = sqrt(dpx*dpx+dpy*dpy+dpz*dpz);
			//magnitude of relative position

			relative_v = dot_prod(dpx,dpy,dpz,dvx,dvy,dvz)/r;
			//magnitude of relative velocity,
			// projected onto the relative position.
			//I.e., relative velocity along the spring.

			dD= r - i_lengths[j];
			//difference between relative position and default length of spring
			
			fx[i] = fx[i] + (dD*i_kconst[j] + dampening*relative_v)*dpx/r;
			fy[i] = fy[i] + (dD*i_kconst[j] + dampening*relative_v)*dpy/r;
			fz[i] = fz[i] + (dD*i_kconst[j] + dampening*relative_v)*dpz/r;
		                                                                 
			fx[node2] = fx[node2] - (dD*i_kconst[j] + dampening*relative_v)*dpx/r;
			fy[node2] = fy[node2] - (dD*i_kconst[j] + dampening*relative_v)*dpy/r;
			fz[node2] = fz[node2] - (dD*i_kconst[j] + dampening*relative_v)*dpz/r; // */
		}
	}
	
	//Update velocity
	for(i = 0; i < N; i++){
		if(!anchor[i]){
			vx[i] = vx[i] + fx[i]*dt;
			vy[i] = vy[i] + fy[i]*dt;
			vz[i] = vz[i] + fz[i]*dt;
		}
	}// */
	
	//Move elements
	for(i=0; i<N; i++)
	{
		if(!anchor[i]){
			px[i] = px[i] + vx[i]*dt;
			py[i] = py[i] + vy[i]*dt;
			pz[i] = pz[i] + vz[i]*dt;
		}
	}
}

void update(int value){
	n_body();
	glutPostRedisplay();
	glutTimerFunc(16, update, 0);
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
	glutTimerFunc(16, update, 0);
	glutReshapeFunc(reshape);

	glutMainLoop();

	return 0;
}

