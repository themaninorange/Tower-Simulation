/*
Joseph Brown

gcc simintersections.c -o temp -lglut -lm -lGLU -lGL
*/

#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PI 3.141592654

#define N 100

#define XWindowSize 700
#define YWindowSize 700

#define STOP_TIME 100.0
#define DT        0.00005

#define GRAVITY 1.0 
/*
#define MASSBODY1 10.0
#define MASSBODY2 10.0*/
#define ALLMASS 10

#define DRAW 10

// Globals
double px[N], py[N], pz[N], vx[N], vy[N], vz[N], fx[N], fy[N], fz[N], mass[N], radii[N], G_const, H_const, p_const, q_const, k_const, rod_proportion; 

void set_initial_conditions()
{
	G_const = 1;
	H_const = 0;
	p_const = 2;
	q_const = 1;
	k_const = 400;
	rod_proportion = 2;
	int i;
	for(i = 0; i < N; i++){
		mass[i] = ALLMASS;
	}	//Assigns masses to spheres.
	
	mass[0] = 1000;
	
	for(i = 0; i < N; i++){
		radii[i] = 0.01*pow(mass[i], 1.0/3);
	}	//Assigns radii to spheres based on mass.
	
	for(i = 0; i < N; i++){
		px[i] = 2.0*i/N-1;
		py[i] = sqrt(1-px[i]*px[i])*sin(i);
		pz[i] = sqrt(1-px[i]*px[i])*cos(i);
	}	//Initialize spheres around the unit sphere
	
	//:px[0] = 0;
	py[0] = 0;
	pz[0] = 0;// */
	
	/*for(i = 0; i < N; i++){
		vx[i] = py[i] + 0.5;
		vy[i] = pz[i] + 0.5;
		vz[i] = px[i] + 0.5;
	}// */	//Initialize sphere velocities to something nonzero
	
	for(i = 0; i < N; i++){
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
	
	glutSwapBuffers();
}

int n_body()
{
	double fx[N], fy[N], fz[N], dx, dy, dz, rp1, rq1, feynFac, dD; 
	double dt = DT;
	int    tdraw = 0; 
	int    tprint = 0;
	double time = 0.0;
	int    i,j;
	double r;
	
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
				dx = px[j] - px[i];
				dy = py[j] - py[i];
				dz = pz[j] - pz[i];
				
				r = sqrt(dx*dx+dy*dy+dz*dz);
				dD= r - (radii[i]+radii[j])*rod_proportion;
				
				fx[i] = fx[i] + k_const*dx*dD/r;
				fy[i] = fy[i] + k_const*dy*dD/r;
				fz[i] = fz[i] +	k_const*dz*dD/r;
			
				fx[j] = fx[j] - k_const*dx*dD/r;
				fy[j] = fy[j] - k_const*dy*dD/r;
				fz[j] = fz[j] - k_const*dz*dD/r;
			}
		}
		
		//Update velocity
		for(i = 0; i < N; i++){
			vx[i] = vx[i] + fx[i]*dt;
			vy[i] = vy[i] + fy[i]*dt;
			vz[i] = vz[i] + fz[i]*dt;
		}// */
		
		//Move elements
		for(i=0; i<N; i++)
		{
			px[i] = px[i] + vx[i]*dt;
			py[i] = py[i] + vy[i]*dt;
			pz[i] = pz[i] + vz[i]*dt;
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

glBegin(GL_TRIANGLES);           // Begin drawing the pyramid with 4 triangles
      // Front
      glColor3f(1.0f, 0.0f, 0.0f);     // Red
      glVertex3f( 0.0f, 1.0f, 0.0f);
      glColor3f(0.0f, 1.0f, 0.0f);     // Green
      glVertex3f(-1.0f, -1.0f, 1.0f);
      glColor3f(0.0f, 0.0f, 1.0f);     // Blue
      glVertex3f(1.0f, -1.0f, 1.0f);
 
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
glEnd();   // Done drawing the pyramid

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

