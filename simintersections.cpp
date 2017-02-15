/*
Joseph Brown

g++ simintersections.c -o temp -lglut -lm -lGLU -lGL -std=c++11
*/

#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <random>

#define PI 3.141592654

#define N 100
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

//std::random_device generator;
std::uniform_real_distribution<float> unif_dist(0.0,1.0);// */

// Globals
double px[N], py[N], pz[N], vx[N], vy[N], vz[N], fx[N], fy[N], fz[N], mass[N], radii[N], G_const, H_const, p_const, q_const, k_const, k_anchor, rod_proportion, dampening;

int 	conn_index[N]; 		//List of where to look in i_conns for connections.
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

/*void create_connections(){

	int index = 0;
	int numconns;
	float prob = 0.75;
	double kconst, length;
	
	
	
	int i,j;
	for(i = 0 ; i < N ; i++){
		numconns = 0;
		for(j = i+1; j < N ; j++){
			if(!anchor[i] || !anchor[j]){
				if(unif_dist(generator) < prob){
					i_conns[index] = j;
					index += 1;
					numconns += 1;
				}
			}
		}
		if(i!=N-1){
			conn_index[i+1] = conn_index[i] + numconns;
			//The next index should begin after all of the connections this node has.
		}
	} 
}// */

void set_initial_conditions()
{
	G_const = 1;
	k_const = 1;
	k_anchor = 4000;
	rod_proportion = 10;
	dampening = 0.03;	int i;
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

		mass[i] = 10000;
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
		for(j = i + 1 ; j < N ; j++){
			glBegin(GL_LINES);
				glVertex3f(px[i], py[i], pz[i]);
				glVertex3f(px[j], py[j], pz[j]);
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

/*glBegin(GL_TRIANGLES);           // Begin drawing the pyramid with 4 triangles
      // Front
      glColor3f(1.0f, 0.0f, 0.0f);     // Red
      glVertex3f( 0.0f, 1.0f, 0.0f); 
      glColor3f(0.0f, 0.0f, 1.0f);     // Blue
      glVertex3f(1.0f, -1.0f, 1.0f);
      glColor3f(0.0f, 1.0f, 0.0f);     // Green
      glVertex3f(-1.0f, -1.0f, 1.0f);
 
      // Right
      glColor3f(1.0f, 0.0f, 0.0f);     // Red
      glVertex3f(0.0f, 1.0f, 0.0f);
      glColor3f(0.0f, 0.0f, 1.0f);     // Blue
      glVertex3f(1.0f, -1.0f, 1.0f);
      glColor3f(0.0f, 1.0f, 0.0f);     // Green
      glVertex3f(1.0f, -1.0f, -1.0f);
 
      // Back
      glColor3f(1.0f, 0.0f, 0.0f);     // Red
      glVertex3f(0.0f, 1.0f, 0.0f);
      glColor3f(0.0f, 1.0f, 0.0f);     // Green
      glVertex3f(1.0f, -1.0f, -1.0f);
      glColor3f(0.0f, 0.0f, 1.0f);     // Blue
      glVertex3f(-1.0f, -1.0f, -1.0f);
 
      // Left
      glColor3f(1.0f,0.0f,0.0f);       // Red
      glVertex3f( 0.0f, 1.0f, 0.0f);
      glColor3f(0.0f,0.0f,1.0f);       // Blue
      glVertex3f(-1.0f,-1.0f,-1.0f);
      glColor3f(0.0f,1.0f,0.0f);       // Green
      glVertex3f(-1.0f,-1.0f, 1.0f);
glEnd();   // Done drawing the pyramid */
	
	
	glutSwapBuffers();
}

double dot_prod(double x1, double y1, double z1, double x2, double y2, double z2){
	return(x1*x2+y1*y2+z1*z2);
}

int n_body()
{
	double fx[N], fy[N], fz[N], dvx, dvy, dvz, dpx, dpy, dpz, rp1, rq1, feynFac, dD; 
	double dt = DT;
	int    tdraw = 0; 
	int    tprint = 0;
	double time = 0.0;
	int    i,j;
	double r, relative_v;	
	dt = DT;

	while(time < STOP_TIME)
	{
		for(i=0; i<N; i++)
		{
			fx[i] = 0.0;
			fy[i] = 0.0;
			fz[i] = 0.0;
		}

		//Get forces
		for(i=0; i<N; i++)
		{
			for(j=i+1; j<N; j++)
			{
				dpx = px[j] - px[i];
				dpy = py[j] - py[i];
				dpz = pz[j] - pz[i];
				
				dvx = vx[j] - vx[i];
				dvy = vy[j] - vy[i];
				dvz = vz[j] - vz[i];

				r = sqrt(dpx*dpx+dpy*dpy+dpz*dpz);

				relative_v = dot_prod(dpx,dpy,dpz,dvx,dvy,dvz)/r;

				dD= r - (radii[i]+radii[j])*rod_proportion;
				
				fx[i] = fx[i] + (dD*k_const + dampening*relative_v)*dpx/r;
				fy[i] = fy[i] + (dD*k_const + dampening*relative_v)*dpy/r;
				fz[i] = fz[i] + (dD*k_const + dampening*relative_v)*dpz/r;
			                                                                 
				fx[j] = fx[j] - (dD*k_const + dampening*relative_v)*dpx/r;
				fy[j] = fy[j] - (dD*k_const + dampening*relative_v)*dpy/r;
				fz[j] = fz[j] - (dD*k_const + dampening*relative_v)*dpz/r; // */

				/*fx[i] = fx[i] + ((dD*dpx>0)?1:-1)*(fabs((dD)*k_const*dpx/r) - dampening*fabs(relative_v));
				fy[i] = fy[i] + ((dD*dpy>0)?1:-1)*(fabs((dD)*k_const*dpy/r) - dampening*fabs(relative_v));
				fz[i] = fz[i] + ((dD*dpz>0)?1:-1)*(fabs((dD)*k_const*dpz/r) - dampening*fabs(relative_v));
			                                                      
				fx[j] = fx[j] - ((dD*dpx>0)?1:-1)*(fabs((dD)*k_const*dpx/r) - dampening*fabs(relative_v));
				fy[j] = fy[j] - ((dD*dpy>0)?1:-1)*(fabs((dD)*k_const*dpy/r) - dampening*fabs(relative_v));
				fz[j] = fz[j] - ((dD*dpz>0)?1:-1)*(fabs((dD)*k_const*dpz/r) - dampening*fabs(relative_v)); // */
				// Ideally ensures that the dampening never reverses the direction of the force.
				// Does not work.
				
				if(j == 6 && time > 100*DT){
					printf(
"dampening: %.3f\nrelvel: %.3f\nprimary force: %.3f\ntruthiness: %d\nr: %.3f\n", 
((dD>0)?1:-1)*dampening*relative_v, relative_v, k_const*dD, ((dD>0)?1:-1), r); // */
					/*printf(
"dpx: %.3f\ndpy: %.3f\ndpz: %.3f\ndvx: %.3f\ndvy: %.3f\ndvz: %.3f\nrelative_v: %.3f\n", 
dpx, dpy, dpz, dvx, dvy, dvz, relative_v); // */

					time = 0.0;
				}
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

		if(tdraw == DRAW) 
		{
			draw_picture();
			tdraw = 0;
		}

		time += dt;
		tdraw++;
		tprint++;
	}
}

void control()
{	
	int    tdraw = 0;
	double  time = 0.0;

	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);

	set_initial_conditions();
	
	draw_picture();
	
	n_body();
	
	printf("\n DONE \n");
	while(1);
}

void Display(void)
{
	gluLookAt(0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glutSwapBuffers();
	glFlush();
	control();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 150.0);

	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	printf("pass\n");
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
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}

