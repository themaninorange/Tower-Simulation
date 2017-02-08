/*
Joseph Brown

gcc simintersections.c -o temp -lglut -lm -lGLU -lGL
*/

#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

#define PI 3.141592654

#define N 7

#define XWindowSize 1024
#define YWindowSize 1024

#define STOP_TIME 1000.0
#define DT        0.05

#define GRAVITY 1.0 
/*
#define MASSBODY1 10.0
#define MASSBODY2 10.0*/
#define ALLMASS 10

#define DRAW 10

// Globals
double px[N], py[N], pz[N], vx[N], vy[N], vz[N], fx[N], fy[N], fz[N], mass[N], radii[N], G_const, H_const, p_const, q_const, k_const, k_anchor, rod_proportion, dampening; 

bool anchorx[N], anchory[N], anchorz[N];

void set_initial_conditions()
{
	G_const = 1;
	H_const = 0;
	p_const = 2;
	q_const = 1;
	k_const = 1;
	k_anchor = 4000;
	rod_proportion = 10;
	dampening = 0.01;	int i;
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
		
		anchorx[i] = true;
		anchory[i] = true;
		anchorz[i] = true;
	
		vx[i] = 0;
		vy[i] = 0;
		vz[i] = 0;

		mass[i] = 10000;
	}
	
	px[6] = 0;
	pz[6] = 0;
	py[6] = 1;

	vx[6] = 0;
	vz[6] = 0;
	vy[6] = 0;
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
	
	int i;
	for(i = 0 ; i < N ; i++){
		glColor3d(1.0,1.0,0.5);
		glPushMatrix();
		glTranslatef(px[i], py[i], pz[i]);
		glutSolidSphere(radii[i],20,20);
		glPopMatrix();
	}

	glBegin(GL_QUADS);
		glColor3f(0.2f, 1.0f, 0.5f);     // Red
		glVertex3f( -2.0f, -1.0f, 2.0f); 
		glColor3f(0.2f, 0.2f, 0.3f);     // Blue
		glVertex3f(-2.0f, -1.0f, -2.0f);
		glColor3f(0.2f, 0.2f, 0.6f);     // Green
		glVertex3f(2.0f, -1.0f, -2.0f);
		glColor3f(0.2f, 1.0f, 0.3f);     // Green
		glVertex3f(2.0f, -1.0f, 2.0f);
	glEnd();


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
				
				fx[i] = fx[i] + ((dD*dpx>0)?1:-1)*(fabs((dD)*k_const*dpx/r) - dampening*fabs(relative_v));
				fy[i] = fy[i] + ((dD*dpy>0)?1:-1)*(fabs((dD)*k_const*dpy/r) - dampening*fabs(relative_v));
				fz[i] = fz[i] + ((dD*dpz>0)?1:-1)*(fabs((dD)*k_const*dpz/r) - dampening*fabs(relative_v));
			                                                      
				fx[j] = fx[j] - ((dD*dpx>0)?1:-1)*(fabs((dD)*k_const*dpx/r) - dampening*fabs(relative_v));
				fy[j] = fy[j] - ((dD*dpy>0)?1:-1)*(fabs((dD)*k_const*dpy/r) - dampening*fabs(relative_v));
				fz[j] = fz[j] - ((dD*dpz>0)?1:-1)*(fabs((dD)*k_const*dpz/r) - dampening*fabs(relative_v));
				
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
			if(!anchorx[i]){
				vx[i] = vx[i] + fx[i]*dt;
			}
			if(!anchorx[i]){
				vy[i] = vy[i] + fy[i]*dt;
			}
			if(!anchorx[i]){
				vz[i] = vz[i] + fz[i]*dt;
			}
		}// */
		
		//Move elements
		for(i=0; i<N; i++)
		{
			if(!anchorx[i]){
				px[i] = px[i] + vx[i]*dt;
			}
			if(!anchory[i]){
				py[i] = py[i] + vy[i]*dt;
			}
			if(!anchorz[i]){
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

