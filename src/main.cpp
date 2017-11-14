
// The loaders are included by glfw3 (glcorearb.h) if we are not using glew.
#ifdef USE_GLEW
#include <GL/glew.h>
#endif
#include <GLFW/glfw3.h>

// Includes
#include <omp.h>
#include "trimesh.hpp"
#include "shader.hpp"
#include <cstring> // memcpy
#include <random>
#include <algorithm>
#include <omp.h>

// Constants
#define WIN_WIDTH 800
#define WIN_HEIGHT 800

#define PI 4.0*atan(1.0)
#define GRAVITY -9.8
#define MAXDT .25
#define MINHEIGHT 0.01
#define NUMPRESETS 6

#define DEBUG 0
#define USE_BVH 0
#define USE_OMP 1

using std::vector;
using std::max;

typedef Vec<2,float> Vec2f; // Declare to use inside of deformableMesh

class Mat4x4 {
public:

	float m[16];

	Mat4x4(){ // Default: Identity
		m[0] = 1.f;  m[4] = 0.f;  m[8]  = 0.f;  m[12] = 0.f;
		m[1] = 0.f;  m[5] = 1.f;  m[9]  = 0.f;  m[13] = 0.f;
		m[2] = 0.f;  m[6] = 0.f;  m[10] = 1.f;  m[14] = 0.f;
		m[3] = 0.f;  m[7] = 0.f;  m[11] = 0.f;  m[15] = 1.f;
	}

	void make_identity(){
		m[0] = 1.f;  m[4] = 0.f;  m[8]  = 0.f;  m[12] = 0.f;
		m[1] = 0.f;  m[5] = 1.f;  m[9]  = 0.f;  m[13] = 0.f;
		m[2] = 0.f;  m[6] = 0.f;  m[10] = 1.f;  m[14] = 0.f;
		m[3] = 0.f;  m[7] = 0.f;  m[11] = 0.f;  m[15] = 1.f;
	}

	void print(){
		std::cout << m[0] << ' ' <<  m[4] << ' ' <<  m[8]  << ' ' <<  m[12] << "\n";
		std::cout << m[1] << ' ' <<   m[5] << ' ' <<  m[9]  << ' ' <<   m[13] << "\n";
		std::cout << m[2] << ' ' <<   m[6] << ' ' <<  m[10] << ' ' <<   m[14] << "\n";
		std::cout << m[3] << ' ' <<   m[7] << ' ' <<  m[11] << ' ' <<   m[15] << "\n";
	}

	void make_scale(float x, float y, float z){
		make_identity();
		m[0] = x; m[5] = y; m[10] = x;
	}
};

static inline const Vec3f operator*(const Mat4x4 &m, const Vec3f &v) {
	Vec3f r(m.m[0] * v[0] + m.m[4] * v[1] + m.m[8] * v[2],
		m.m[1] * v[0] + m.m[5] * v[1] + m.m[9] * v[2],
		m.m[2] * v[0] + m.m[6] * v[1] + m.m[10] * v[2]);
	return r;
}
static inline const Vec3f operator*(const double d, const Vec3f &v) {
	Vec3f r(v[0] * d, v[1] * d, v[2] * d);
	return r;
}
static inline const Vec3f operator*(const Vec3f &v, const double d) {
	Vec3f r(v[0] * d, v[1] * d, v[2] * d);
	return r;
}
static inline const Vec3f operator+(const Vec3f &v1, const Vec3f &v2) {
	Vec3f r(v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]);
	return r;
}

float dot(Vec3f v1, Vec3f v2) {
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}


typedef struct {
	bool active = false;
	GLdouble prev_x;
	GLdouble prev_y;
} MouseInfo;

//
//	Global state variables
//
namespace Globals {
	float win_width, win_height; // window size
	float aspect;
	int vert_dim = 2;
	GLuint position_vbo[1], colors_vbo[1], faces_ibo[1], normals_vbo[1], scene_vao;
	Vec3f lightDir = Vec3f(-1, -1, 0);

	//  Model, view and projection matrices, initialized to the identity
	Mat4x4 model;
	Mat4x4 view;
	Mat4x4 projection;

	// Scene variables
	Vec3f eye;
	float near = .1;
	float far = 1000;
	float left = -.1;
	float right = .1;
	float top = .1;
	float bottom = -.1;
	Vec3f viewDir;
	Vec3f upDir;
	Vec3f rightDir;

	// Input variables
	bool key_w; // forward movement
	bool key_s; // backward movement
	bool key_d; // right strafing
	bool key_a; // left strafing
	bool key_e; // upward movement
	bool key_q; // downward movement
	bool key_lshift; // speed up

	double theta;
	double phi;
	Mat4x4 xRot;
	Mat4x4 yRot;

	MouseInfo mouse;
	mcl::Shader currShader;
	double currTime, prevTime;
	int preset = 0;
	bool pause = false, addTimeMultiplier = false, subTimeMultiplier = false;
	double timeMultiplier = 1;
	double movementSpeed = 0.1;
	GLFWwindow *activeWindow;
}

class ShallowWater {
public:
	ShallowWater(int e = 0) {
		changeWater(e);
	}

	ShallowWater(int n, int d, double damp) {
		bool goDefault = false;
		if (n < 1) {
			std::cerr << "Error! Cannot have less than 1 grid in either dimension! Using default values." << std::endl;
			goDefault = true;
		}
		if (d < 0) {
			std::cerr << "Error! Cannot have grid size less than or equal to 0 in either dimension! Using default values." << std::endl;
			goDefault = true;
		}

		if (goDefault) {
			numCellsX = 5;
			cellSizeX = .2;
			numCellsZ = 1;
			cellSizeZ = 1;

			damping = 4;
		}
		else {
			numCellsX = n;
			cellSizeX = d;
			numCellsZ = n;
			cellSizeZ = d;

			damping = damp;
		}

		initVectors();
	}

	ShallowWater(int nx, double dx, int nz, double dz, double damp) {
		bool goDefault = false;
		if (nx < 1 || nz < 1) {
			std::cerr << "Error! Cannot have less than 1 grid in either dimension! Using default values." << std::endl;
			goDefault = true;
		}
		if (dx < 0 || dz < 0) {
			std::cerr << "Error! Cannot have grid size less than or equal to 0 in either dimension! Using default values." << std::endl;
			goDefault = true;
		}

		if (goDefault) {
			numCellsX = 5;
			cellSizeX = .2;
			numCellsZ = 1;
			cellSizeZ = 1;

			damping = 4;
		}
		else {
			numCellsX = nx;
			cellSizeX = dx;
			numCellsZ = nz;
			cellSizeZ = dz;

			damping = damp;
		}

		initVectors();
	}

	virtual ~ShallowWater() {}

	void setupWave(int e) {
		stopVelocity();

		switch(e) {
		default:
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX + 3; i++) {
				#if USE_OMP
					#pragma omp parallel for
				#endif
				for (int j = 0; j < numCellsZ + 3; j++) {
					height[i][j] = 1;
				}
			}
			break;

		case 2:
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX + 3; i++) {
				#if USE_OMP
					#pragma omp parallel for
				#endif
				for (int j = 0; j < numCellsZ + 3; j++) {
					height[i][j] = 2*double(i)/numCellsX;
					if (numCellsX >= 100) height[i][j] *= (numCellsX/100.0);
				}
			}
			break;

		case 3:
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX + 3; i++) {
				#if USE_OMP
					#pragma omp parallel for
				#endif
				for (int j = 0; j < numCellsZ + 3; j++) {
					height[i][j] = 2*double(j)/numCellsZ;
					if (numCellsX >= 100) height[i][j] *= (numCellsX/100.0);
				}
			}
			break;

		case 4:
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX + 3; i++) {
				#if USE_OMP
					#pragma omp parallel for
				#endif
				for (int j = 0; j < numCellsZ + 3; j++) {
					height[i][j] = (1.3 - abs(double(i)/numCellsX - double(j)/numCellsZ));
					if (numCellsX >= 100) height[i][j] *= (numCellsX/100.0);
				}
			}
			break;

		case 5:
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX + 3; i++) {
				#if USE_OMP
					#pragma omp parallel for
				#endif
				for (int j = 0; j < numCellsZ + 3; j++) {
					double sqr1 = .1*(i - (numCellsX+2)/2) * 2*(i - (numCellsX+2)/2)/(numCellsX+2);
					double sqr2 = .1*(j - (numCellsZ+2)/2) * 2*(j - (numCellsZ+2)/2)/(numCellsZ+2);
					height[i][j] = .8 - (1 - sqrt(sqr1 + sqr2));
				}
			}
			break;

		case 6:
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX + 3; i++) {
				#if USE_OMP
					#pragma omp parallel for
				#endif
				for (int j = 0; j < numCellsZ + 3; j++) {
					height[i][j] = (j % 25 < 12 || i % 25 < 12) ? 0.9 : 1.1;
				}
			}
			break;

		case 7:
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX + 3; i++) {
				#if USE_OMP
					#pragma omp parallel for
				#endif
				for (int j = 0; j < numCellsZ + 3; j++) {
					height[i][j] = (j % 2 == 0 || i % 2 == 0) ? 0.6 : 2 + (numCellsX/100.0);
				}
			}
			break;

		} // end switch

		#if USE_OMP
			#pragma omp parallel for
		#endif
		for (int i = 0; i < numCellsX + 3; i++) {
			height[i][0] = height[i][1];
			height[i][height[i].size()-1] = height[i][height[i].size()-2];
		}

		#if USE_OMP
			#pragma omp parallel for
		#endif
		for (int i = 0; i < numCellsZ + 3; i++) {
			height[0][i] = height[1][i];
			height[height.size()-1][i] = height[height.size()-2][i];
		}
	}

	void interact(int e) {
		int x, z;
		switch (e) {
		default: break;
		case 0: 
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX + 3; i++) {
				#if USE_OMP
					#pragma omp parallel for
				#endif
				for (int j = 0; j < numCellsZ + 3; j++) {
					if (force >= 0)
						height[i][j] += force;
					else
						height[i][j] += (height[i][j] <= MINHEIGHT) ? 0 : force;
				}
			}
			return;
		case 1: x = 0; z = 0; break;
		case 2: x = (numCellsX)/2; z = 0; break;
		case 3: x = numCellsX; z = 0; break;
		case 4: x = 0; z = (numCellsZ)/2; break;
		case 5: x = (numCellsX)/2; z = (numCellsZ)/2; break;
		case 6: x = numCellsX; z = (numCellsZ)/2; break;
		case 7: x = 0; z = numCellsZ; break;
		case 8: x = (numCellsX)/2; z = numCellsZ; break;
		case 9: x = numCellsX; z = numCellsZ;
		}
		if (force >= 0)
			height[x+1][z+1] += force;
		else
			height[x+1][z+1] += (height[x+1][z+1] <= MINHEIGHT) ? 0 : force;
	}

	void changeWater(int e) {
		switch (e) {
		case 0:
		default:
			numCellsX = 10;
			cellSizeX = .1;
			numCellsZ = 10;
			cellSizeZ = .1;

			damping = 2;
			break;

		case 1:
			numCellsX = 10;
			cellSizeX = .1;
			numCellsZ = 1;
			cellSizeZ = .5;

			damping = 4;
			break;

		case 2:
			numCellsX = 50;
			cellSizeX = .04;
			numCellsZ = 2;
			cellSizeZ = .5;

			damping = 4;
			break;

		case 3:
			numCellsX = 50;
			cellSizeX = .05;
			numCellsZ = 50;
			cellSizeZ = .05;

			damping = 2;
			break;

		case 4:
			numCellsX = 100;
			cellSizeX = .05;
			numCellsZ = 100;
			cellSizeZ = .05;

			damping = 1.5;
			break;

		case 5:
			numCellsX = 200;
			cellSizeX = .1;
			numCellsZ = 200;
			cellSizeZ = .1;

			damping = .8;
			break;
		}

		initVectors();
	}

	bool changeShading() {
		smoothShading = !smoothShading;
		changeWater(Globals::preset);
		return smoothShading;
	}

	void changeColorMode() {
		colorMode = (colorMode+1)%4;
		switch (colorMode) {
		case 0:
			std::cout << "Shallow Water - Color Mode: Regular" << std::endl;
			break;
		case 1:
			std::cout << "Shallow Water - Color Mode: Height map" << std::endl;
			break;
		case 2:
			std::cout << "Shallow Water - Color Mode: Normal map" << std::endl;
			break;
		case 3:
			std::cout << "Shallow Water - Color Mode: Velocity map" << std::endl;
			break;
		default:;
		}
	}

	// Update according to the shallow water equation
	void wave(double dt) {
		if (holdWave)
			return;

		// Half step computations
		// X direction
		#if USE_OMP
			#pragma omp parallel for
		#endif
		for (int i = 0; i < numCellsX + 2; i++) {
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int j = 0; j < numCellsZ + 2; j++) {
				heightX[i][j] = (height[i][j+1] + height[i+1][j+1])/2.0 - (dt/2.0)*(velocityX[i+1][j+1] - velocityX[i][j+1])/cellSizeX;
				heightX[i][j] = max(MINHEIGHT, heightX[i][j]);

				velocityXMidX[i][j] = (velocityX[i][j+1] + velocityX[i+1][j+1])/2.0 + (dt/2.0)*(
					((velocityX[i+1][j+1]*velocityX[i+1][j+1])/height[i+1][j+1] + GRAVITY/2.0*(height[i+1][j+1]*height[i+1][j+1])) -
					((velocityX[i][j+1]*velocityX[i][j+1])/height[i][j+1] + GRAVITY/2.0*(height[i][j+1]*height[i][j+1]))
				)/cellSizeX;

				velocityXMidZ[i][j] = (velocityZ[i][j+1] + velocityZ[i+1][j+1])/2.0 + (dt/2.0)*(
					(velocityX[i+1][j+1]*velocityZ[i+1][j+1])/height[i+1][j+1] -
					(velocityX[i][j+1]*velocityZ[i][j+1])/height[i][j+1]
				)/cellSizeX;
			}
		}

		// Z direction
		#if USE_OMP
			#pragma omp parallel for
		#endif
		for (int i = 0; i < numCellsX + 2; i++) {
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int j = 0; j < numCellsZ + 2; j++) {
				heightZ[i][j] = (height[i+1][j] + height[i+1][j+1])/2.0 - (dt/2.0)*(velocityZ[i+1][j+1] - velocityZ[i+1][j])/cellSizeZ;
				heightZ[i][j] = max(MINHEIGHT, heightZ[i][j]);

				velocityZMidX[i][j] = (velocityX[i+1][j] + velocityX[i+1][j+1])/2.0 + (dt/2.0)*(
					(velocityX[i+1][j+1]*velocityZ[i+1][j+1])/height[i+1][j+1] -
					(velocityX[i+1][j]*velocityZ[i+1][j])/height[i+1][j]
				)/cellSizeZ;

				velocityZMidZ[i][j] = (velocityZ[i+1][j] + velocityZ[i+1][j+1])/2.0 + (dt/2.0)*(
					((velocityZ[i+1][j+1]*velocityZ[i+1][j+1])/height[i+1][j+1] + GRAVITY/2.0*(height[i+1][j+1]*height[i+1][j+1])) -
					((velocityZ[i+1][j]*velocityZ[i+1][j])/height[i+1][j] + GRAVITY/2.0*(height[i+1][j]*height[i+1][j]))
				)/cellSizeZ;
			}
		}

		// Full step computations
		#if USE_OMP
			#pragma omp parallel for
		#endif
		for (int i = 0; i < numCellsX+1; i++) {
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int j = 0; j < numCellsZ+1; j++) {
				height[i+1][j+1] -= dt*(velocityXMidX[i+1][j] - velocityXMidX[i][j])/cellSizeX +
									dt*(velocityZMidZ[i][j+1] - velocityZMidZ[i][j])/cellSizeZ;
			}
		}

		#if USE_OMP
			#pragma omp parallel for
		#endif
		for (int i = 0; i < numCellsX+1; i++) {
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int j = 0; j < numCellsZ+1; j++) {
				if (height[i+1][j+1] < MINHEIGHT) {
					height[i+1][j+1] = MINHEIGHT;
					velocityX[i+1][j+1] = 0.0;
					velocityZ[i+1][j+1] = 0.0;
					continue;
				}

				velocityX[i+1][j+1] += dt*(-damping*velocityX[i+1][j+1] +
										(((velocityXMidX[i+1][j]*velocityXMidX[i+1][j])/heightX[i+1][j] + GRAVITY/2.0*(heightX[i+1][j]*heightX[i+1][j])) -
										((velocityXMidX[i][j]*velocityXMidX[i][j])/heightX[i][j] + GRAVITY/2.0*(heightX[i][j]*heightX[i][j])))/cellSizeX +
										((velocityZMidX[i][j+1]*velocityZMidZ[i][j+1])/heightZ[i][j+1] -
										(velocityZMidX[i][j]*velocityZMidZ[i][j])/heightZ[i][j])/cellSizeZ);
										
				velocityZ[i+1][j+1] += dt*(-damping*velocityZ[i+1][j+1] +
										((velocityXMidX[i+1][j]*velocityXMidZ[i+1][j])/heightX[i+1][j] -
										(velocityXMidX[i][j]*velocityXMidZ[i][j])/heightX[i][j])/cellSizeX +
										(((velocityZMidZ[i][j+1]*velocityZMidZ[i][j+1])/heightZ[i][j+1] + GRAVITY/2.0*(heightZ[i][j+1]*heightZ[i][j+1])) -
										((velocityZMidZ[i][j]*velocityZMidZ[i][j])/heightZ[i][j] + GRAVITY/2.0*(heightZ[i][j]*heightZ[i][j])))/cellSizeZ);
			}
		}

		// Update edge cases
		#if USE_OMP
			#pragma omp parallel for
		#endif
		for (int i = 0; i < numCellsZ + 3; i++) {
			height[0][i] = height[1][i];
			height[numCellsX+2][i] = height[numCellsX+1][i];

			velocityX[0][i] = -.8*velocityX[1][i];
			velocityX[numCellsX+2][i] = -.8*velocityX[numCellsX+1][i];

			velocityZ[0][i] = velocityZ[1][i];
			velocityZ[numCellsX+2][i] = velocityZ[numCellsX+1][i];
		}

		#if USE_OMP
			#pragma omp parallel for
		#endif
		for (int i = 0; i < numCellsX + 3; i++) {
			height[i][0] = height[i][1];
			height[i][numCellsZ+2] = height[i][numCellsZ+1];

			velocityX[i][0] = velocityX[i][1];
			velocityX[i][numCellsZ+2] = velocityX[i][numCellsZ+1];

			velocityZ[i][0] = -.8*velocityZ[i][1];
			velocityZ[i][numCellsZ+2] = -.8*velocityZ[i][numCellsZ+1];
		}
	}

	void updatePositions() {
		// Update the positions on a quad basis
		if (smoothShading) {
			// Top faces
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX+1; i++) {
				#if USE_OMP
					#pragma omp parallel for
				#endif
				for (int j = 0; j < numCellsZ+1; j++) {
					int ih = i + 1;
					int jh = j + 1;
					position[i*(numCellsZ+1) + j][1] = height[ih][jh];
				}
			}

			// Back faces
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX + 1; i++) {
				int ih = i + 1;
				position[(numCellsX+1)*(numCellsZ+1) + i*2][1] = height[ih][numCellsZ+1];
			}

			// Front faces
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX + 1; i++) {
				int ih = i + 1;
				position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*2 + i*2][1] = height[ih][1];
			}

			// Start faces
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsZ + 1; i++) {
				int ih = i + 1;
				position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + i*2][1] = height[1][ih];
			}

			// End faces
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsZ + 1; i++) {
				int ih = i + 1;
				position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*2 + i*2][1] = height[numCellsX+1][ih];
			}
		}
		else {
			// Top faces
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX; i++) {
				#if USE_OMP
					#pragma omp parallel for
				#endif
				for (int j = 0; j < numCellsZ; j++) {
					int ih = i + 1;
					int jh = j + 1;
					position[i*numCellsZ*4 + j*4][1] = height[ih][jh];
					position[i*numCellsZ*4 + j*4 + 1][1] = height[ih][jh+1];
					position[i*numCellsZ*4 + j*4 + 2][1] = height[ih+1][jh];
					position[i*numCellsZ*4 + j*4 + 3][1] = height[ih+1][jh+1];
				}
			}

			// Front and back faces
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX; i++) {
				int ih = i + 1;
				position[numCellsX*numCellsZ*4 + i*8 + 2][1] = height[ih][numCellsZ+1];
				position[numCellsX*numCellsZ*4 + i*8 + 3][1] = height[ih+1][numCellsZ+1];
				position[numCellsX*numCellsZ*4 + i*8 + 6][1] = height[ih+1][1];
				position[numCellsX*numCellsZ*4 + i*8 + 7][1] = height[ih][1];
			}

			// Start and end faces
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsZ; i++) {
				int ih = i + 1;
				position[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 2][1] = height[1][ih];
				position[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 3][1] = height[1][ih+1];
				position[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 6][1] = height[numCellsX+1][ih];
				position[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 7][1] = height[numCellsX+1][ih+1];
			}
		}
	}

	void updateTopNormals() {
		// Update the normals on a quad basis
		if (smoothShading) {
			// Grab the normal of every face
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX; i++) {
				#if USE_OMP
					#pragma omp parallel for
				#endif
				for (int j = 0; j < numCellsZ; j++) {
					faceNormal[i*numCellsZ + j] = (position[(i+1)*(numCellsZ+1) + j] - position[i*(numCellsZ+1) + j]).cross(position[(i+1)*(numCellsZ+1) + j+1] - position[i*(numCellsZ+1) + j]);
					faceNormal[i*numCellsZ + j].normalize();
				}
			}

			// Average the vertex normals using the faces
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 1; i < numCellsX; i++) {
				#if USE_OMP
					#pragma omp parallel for
				#endif
				for (int j = 1; j < numCellsZ; j++) {
					normal[i*(numCellsZ+1) + j] = (faceNormal[(i-1)*numCellsZ + (j-1)] +
												   faceNormal[(i-1)*numCellsZ + j] +
												   faceNormal[i*numCellsZ + (j-1)] +
												   faceNormal[i*numCellsZ + j]) * .25;
					normal[i*(numCellsZ+1) + j] = faceNormal[i*numCellsZ + j];
					// might want to normalize
				}
			}
			// Update the edge faces using the faces they can access
			for (int i = 1; i < numCellsX; i++) {
				normal[i*(numCellsZ+1)] = (faceNormal[(i-1)*numCellsZ] +
										   faceNormal[i*numCellsZ]) * .5;
			}
			// Update the first corner
			normal[0] = faceNormal[0];
		}
		else {
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX; i++) {
				#if USE_OMP
					#pragma omp parallel for
				#endif
				for (int j = 0; j < numCellsZ; j++) {
					normal[i*numCellsZ*4 + j*4] = (position[i*numCellsZ*4 + j*4 + 2] - position[i*numCellsZ*4 + j*4]).cross(position[i*numCellsZ*4 + j*4 + 3] - position[i*numCellsZ*4 + j*4]);
					normal[i*numCellsZ*4 + j*4].normalize();

					normal[i*numCellsZ*4 + j*4 + 1] = normal[i*numCellsZ*4 + j*4];
					normal[i*numCellsZ*4 + j*4 + 2] = normal[i*numCellsZ*4 + j*4];
					normal[i*numCellsZ*4 + j*4 + 3] = normal[i*numCellsZ*4 + j*4];
				}
			}
		}
	}

	void updateAllNormals() {
		// Update the normals on a quad basis
		if (smoothShading) {
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX; i++) {
				#if USE_OMP
					#pragma omp parallel for
				#endif
				for (int j = 0; j < numCellsZ; j++) {
					normal[i*numCellsZ + j] = (position[(i+1)*numCellsZ + j] - position[i*numCellsZ + j]).cross(position[(i+1)*numCellsZ + j + 1] - position[i*numCellsZ + j]);
					normal[i*numCellsZ + j].normalize();
				}
			}

			// Back faces
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX; i++) {
				normal[(numCellsX+1)*(numCellsZ+1) + i*2] = (position[(numCellsX+1)*(numCellsZ+1) + i*2 + 2] -
															 position[(numCellsX+1)*(numCellsZ+1) + i*2]).cross(position[(numCellsX+1)*(numCellsZ+1) + i*2 + 3] -
															 position[(numCellsX+1)*(numCellsZ+1) + i*2]);
				normal[(numCellsX+1)*(numCellsZ+1) + i*2].normalize();

				normal[(numCellsX+1)*(numCellsZ+1) + i*2 + 1] = (position[(numCellsX+1)*(numCellsZ+1) + i*2 + 3] -
															 position[(numCellsX+1)*(numCellsZ+1) + i*2 + 1]).cross(position[(numCellsX+1)*(numCellsZ+1) + i*2 + 4] -
															 position[(numCellsX+1)*(numCellsZ+1) + i*2 + 1]);
				normal[(numCellsX+1)*(numCellsZ+1) + i*2 + 1].normalize();
			}
			normal[(numCellsX+1)*(numCellsZ+1) + numCellsX*2] = normal[(numCellsX+1)*(numCellsZ+1) + numCellsX*2 - 2];
			normal[(numCellsX+1)*(numCellsZ+1) + numCellsX*2 + 1] = normal[(numCellsX+1)*(numCellsZ+1) + numCellsX*2 - 1];

			// Front faces
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX + 1; i++) {
				normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*2 + i*2] = (position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*2 + i*2 + 2] -
																			   position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*2 + i*2]).cross(position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*2 + i*2 + 3] -
																			   position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*2 + i*2]);
				normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*2 + i*2].normalize();

				normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*2 + i*2 + 1] = (position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*2 + i*2 + 3] -
																				   position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*2 + i*2 + 1]).cross(position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*2 + i*2 + 4] -
																				   position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*2 + i*2 + 1]);
				normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*2 + i*2 + 1].normalize();
			}
			normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*2 + numCellsX*2] = normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*2 + numCellsX*2 - 2];
			normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*2 + numCellsX*2 + 1] = normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*2 + numCellsX*2 - 1];

			// Start faces
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < (numCellsZ + 1); i++) {
				normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + i*2] = (position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + i*2 + 2] -
																	 position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + i*2]).cross(position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + i*2 + 3] -
																	 position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + i*2]);
				normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + i*2].normalize();

				normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + i*2 + 1] = (position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + i*2 + 2] -
																		 position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + i*2 + 1]).cross(position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + i*2 + 3] -
																		 position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + i*2 + 1]);
				normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + i*2 + 1].normalize();
			}
			normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + numCellsZ*2] = normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + numCellsZ*2 - 2];
			normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + numCellsZ*2 + 1] = normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + numCellsZ*2 - 1];

			// End faces
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < (numCellsZ + 1); i++) {
				normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*2 + i*2] = (position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*2 + i*2 + 2] -
																	 position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*2 + i*2]).cross(position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*2 + i*2 + 3] -
																	 position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*2 + i*2]);
				normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*2 + i*2].normalize();

				normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*2 + i*2 + 1] = (position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*2 + i*2 + 2] -
																		 position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*2 + i*2 + 1]).cross(position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*2 + i*2 + 3] -
																		 position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*2 + i*2 + 1]);
				normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*2 + i*2 + 1].normalize();
			}
			normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*2 + numCellsZ*2] = normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*2 + numCellsZ*2 - 2];
			normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*2 + numCellsZ*2 + 1] = normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*2 + numCellsZ*2 - 1];

			// Bottom face
			normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*4 + 1] = (position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*4 + 2] -
																								 position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*4 + 1]).cross(position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*4 + 3] -
																								 position[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*4 + 1]);
			normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*4 + 1].normalize();

			normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*4 + 1] = normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*4];
			normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*4 + 2] = normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*4];
			normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*4 + 3] = normal[(numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*4];
		}
		else {
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX; i++) {
				#if USE_OMP
					#pragma omp parallel for
				#endif
				for (int j = 0; j < numCellsZ; j++) {
					normal[i*numCellsZ*4 + j*4] = (position[i*numCellsZ*4 + j*4 + 2] - position[i*numCellsZ*4 + j*4]).cross(position[i*numCellsZ*4 + j*4 + 3] - position[i*numCellsZ*4 + j*4]);
					normal[i*numCellsZ*4 + j*4].normalize();

					normal[i*numCellsZ*4 + j*4 + 1] = normal[i*numCellsZ*4 + j*4];
					normal[i*numCellsZ*4 + j*4 + 2] = normal[i*numCellsZ*4 + j*4];
					normal[i*numCellsZ*4 + j*4 + 3] = normal[i*numCellsZ*4 + j*4];
				}
			}

			// Front and back faces
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX; i++) {
				normal[numCellsX*numCellsZ*4 + i*8] = (position[numCellsX*numCellsZ*4 + i*8 + 2] -
													   position[numCellsX*numCellsZ*4 + i*8]).cross(position[numCellsX*numCellsZ*4 + i*8 + 3] -
													   position[numCellsX*numCellsZ*4 + i*8]);
				normal[numCellsX*numCellsZ*4 + i*8].normalize();

				normal[numCellsX*numCellsZ*4 + i*8 + 1] = normal[numCellsX*numCellsZ*4 + i*8];
				normal[numCellsX*numCellsZ*4 + i*8 + 2] = normal[numCellsX*numCellsZ*4 + i*8];
				normal[numCellsX*numCellsZ*4 + i*8 + 3] = normal[numCellsX*numCellsZ*4 + i*8];

				normal[numCellsX*numCellsZ*4 + i*8 + 4] = (position[numCellsX*numCellsZ*4 + i*8 + 6] -
														   position[numCellsX*numCellsZ*4 + i*8 + 4]).cross(position[numCellsX*numCellsZ*4 + i*8 + 7] -
														   position[numCellsX*numCellsZ*4 + i*8 + 4]);
				normal[numCellsX*numCellsZ*4 + i*8 + 4].normalize();

				normal[numCellsX*numCellsZ*4 + i*8 + 5] = normal[numCellsX*numCellsZ*4 + i*8 + 4];
				normal[numCellsX*numCellsZ*4 + i*8 + 6] = normal[numCellsX*numCellsZ*4 + i*8 + 4];
				normal[numCellsX*numCellsZ*4 + i*8 + 7] = normal[numCellsX*numCellsZ*4 + i*8 + 4];
			}

			// Start and end faces
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsZ; i++) {
				normal[numCellsX*numCellsZ*4 + numCellsX*8 + i*8] = (position[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 2] -
																	 position[numCellsX*numCellsZ*4 + numCellsX*8 + i*8]).cross(position[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 3] -
																	 position[numCellsX*numCellsZ*4 + numCellsX*8 + i*8]);
				normal[numCellsX*numCellsZ*4 + numCellsX*8 + i*8].normalize();

				normal[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 1] = normal[numCellsX*numCellsZ*4 + numCellsX*8 + i*8];
				normal[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 2] = normal[numCellsX*numCellsZ*4 + numCellsX*8 + i*8];
				normal[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 3] = normal[numCellsX*numCellsZ*4 + numCellsX*8 + i*8];

				normal[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 4] = (position[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 6] -
																		 position[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 4]).cross(position[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 7] -
																		 position[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 4]);
				normal[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 4].normalize();

				normal[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 5] = normal[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 4];
				normal[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 6] = normal[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 4];
				normal[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 7] = normal[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 4];
			}

			// Bottom face
			normal[numCellsX*numCellsZ*4 + numCellsZ*8 + numCellsX*8] = (position[numCellsX*numCellsZ*4 + numCellsZ*8 + numCellsX*8 + 2] -
																 position[numCellsX*numCellsZ*4 + numCellsZ*8 + numCellsX*8]).cross(position[numCellsX*numCellsZ*4 + numCellsZ*8 + numCellsX*8 + 3] -
																 position[numCellsX*numCellsZ*4 + numCellsZ*8 + numCellsX*8]);
			normal[numCellsX*numCellsZ*4 + numCellsZ*8 + numCellsX*8].normalize();

			normal[numCellsX*numCellsZ*4 + numCellsZ*8 + numCellsX*8 + 1] = normal[numCellsX*numCellsZ*4 + numCellsZ*8 + numCellsX*8];
			normal[numCellsX*numCellsZ*4 + numCellsZ*8 + numCellsX*8 + 2] = normal[numCellsX*numCellsZ*4 + numCellsZ*8 + numCellsX*8];
			normal[numCellsX*numCellsZ*4 + numCellsZ*8 + numCellsX*8 + 3] = normal[numCellsX*numCellsZ*4 + numCellsZ*8 + numCellsX*8];
		}
	}

	void updateColors() {
		if (smoothShading) {
			// Top faces
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX+1; i++) {
				#if USE_OMP
					#pragma omp parallel for
				#endif
				for (int j = 0; j < numCellsZ+1; j++) {
					int ih = i + 1;
					int jh = j + 1;
					int location = i*(numCellsZ+1) + j;
					switch (colorMode) {
					case 0:
						color[location] = Vec3f(0, 0, 1);
						break;
					case 1:
						color[location] = Vec3f(
							1 - height[ih][jh]/(3/2),
							(height[ih][jh] < 3/2) ? height[ih][jh]/(3/2) : 1 - (height[ih][jh] - 3/2)/(3/2),
							(height[ih][jh] - 3/2)/(3/2));
						break;
					case 2:
						color[location] = normal[location];
						break;
					case 3:
						color[location] = Vec3f(
							velocityX[ih][jh],
							velocityZ[ih][jh],
							0);
						break;

					default:;
					}
				}
			}

			// Back faces
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX + 1; i++) {
				int ih = i + 1;
				int location = (numCellsX+1)*(numCellsZ+1) + i*2;
				switch (colorMode) {
				case 0:
					color[location] = Vec3f(0, 0, 1);
					break;
				case 1:
					color[location] = Vec3f(
						1 - height[ih][numCellsZ+1]/(3/2),
						(height[ih][numCellsZ+1] < 3/2) ? height[ih][numCellsZ+1]/(3/2) : 1 - (height[ih][numCellsZ+1] - 3/2)/(3/2),
						(height[ih][numCellsZ+1] - 3/2)/(3/2));
					break;
				case 2:
					color[location] = normal[location];
					break;
				case 3:
					color[location] = Vec3f(
						velocityX[ih][numCellsZ+1],
						velocityZ[ih][numCellsZ+1],
						0);
					break;

				default:;
				}
			}

			// Front faces
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX + 1; i++) {
				int ih = i + 1;
				int location = (numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*2 + i*2;
				switch (colorMode) {
				case 0:
					color[location] = Vec3f(0, 0, 1);
					break;
				case 1:
					color[location] = Vec3f(
						1 - height[ih][1]/(3/2),
						(height[ih][1] < 3/2) ? height[ih][1]/(3/2) : 1 - (height[ih][1] - 3/2)/(3/2),
						(height[ih][1] - 3/2)/(3/2));
					break;
				case 2:
					color[location] = normal[location];
					break;
				case 3:
					color[location] = Vec3f(
						velocityX[ih][1],
						velocityZ[ih][1],
						0);
					break;

				default:;
				}
			}

			// Start faces
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsZ + 1; i++) {
				int ih = i + 1;
				int location = (numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + i*2;
				switch (colorMode) {
				case 0:
					color[location] = Vec3f(0, 0, 1);
					break;
				case 1:
					color[location] = Vec3f(
						1 - height[1][ih]/(3/2),
						(height[1][ih] < 3/2) ? height[1][ih]/(3/2) : 1 - (height[1][ih] - 3/2)/(3/2),
						(height[1][ih] - 3/2)/(3/2));
					break;
				case 2:
					color[location] = normal[location];
					break;
				case 3:
					color[location] = Vec3f(
						velocityX[1][ih],
						velocityZ[1][ih],
						0);
					break;

				default:;
				}
			}

			// End faces
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsZ + 1; i++) {
				int ih = i + 1;
				int location = (numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*2 + i*2;
				switch (colorMode) {
				case 0:
					color[location] = Vec3f(0, 0, 1);
					break;
				case 1:
					color[location] = Vec3f(
						1 - height[numCellsX+1][ih]/(3/2),
						(height[numCellsX+1][ih] < 3/2) ? height[numCellsX+1][ih]/(3/2) : 1 - (height[numCellsX+1][ih] - 3/2)/(3/2),
						(height[numCellsX+1][ih] - 3/2)/(3/2));
					break;
				case 2:
					color[location] = normal[location];
					break;
				case 3:
					color[location] = Vec3f(
						velocityX[numCellsX+1][ih],
						velocityZ[numCellsX+1][ih],
						0);
					break;

				default:;
				}
			}
		}
		else {
			// Top faces
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX; i++) {
				#if USE_OMP
					#pragma omp parallel for
				#endif
				for (int j = 0; j < numCellsZ; j++) {
					int ih = i + 1;
					int jh = j + 1;
					int location = i*numCellsZ*4 + j*4;
					switch (colorMode) {
					case 0:
						color[location] = Vec3f(0, 0, 1);
						color[location + 1] = Vec3f(0, 0, 1);
						color[location + 2] = Vec3f(0, 0, 1);
						color[location + 3] = Vec3f(0, 0, 1);
						break;
					case 1:
						color[location] = Vec3f(
							1 - height[ih][jh]/(3/2),
							(height[ih][jh] < 3/2) ? height[ih][jh]/(3/2) : 1 - (height[ih][jh] - 3/2)/(3/2),
							(height[ih][jh] - 3/2)/(3/2));
						color[location + 1] = Vec3f(
							1 - height[ih][jh+1]/(3/2),
							(height[ih][jh+1] < 3/2) ? height[ih][jh+1]/(3/2) : 1 - (height[ih][jh+1] - 3/2)/(3/2),
							(height[ih][jh+1] - 3/2)/(3/2));
						color[location + 2] = Vec3f(
							1 - height[ih+1][jh]/(3/2),
							(height[ih+1][jh] < 3/2) ? height[ih+1][jh]/(3/2) : 1 - (height[ih+1][jh] - 3/2)/(3/2),
							(height[ih+1][jh] - 3/2)/(3/2));
						color[location + 3] = Vec3f(
							1 - height[ih+1][jh+1]/(3/2),
							(height[ih+1][jh+1] < 3/2) ? height[ih+1][jh+1]/(3/2) : 1 - (height[ih+1][jh+1] - 3/2)/(3/2),
							(height[ih+1][jh+1] - 3/2)/(3/2));
						break;
					case 2:
						color[location] = normal[location];
						color[location + 1] = normal[location + 1];
						color[location + 2] = normal[location + 2];
						color[location + 3] = normal[location + 3];
						break;
					case 3:
						color[location] = Vec3f(
							velocityX[ih][jh],
							velocityZ[ih][jh],
							0);
							
						color[location + 1] = Vec3f(
							velocityX[ih][jh+1],
							velocityZ[ih][jh+1],
							0);

						color[location + 2] = Vec3f(
							velocityX[ih+1][jh],
							velocityZ[ih+1][jh],
							0);

						color[location + 3] = Vec3f(
							velocityX[ih+1][jh+1],
							velocityZ[ih+1][jh+1],
							0);
						break;

					default:;
					}
				}
			}

			// Front and back faces
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsX; i++) {
				int ih = i + 1;
				int location = numCellsX*numCellsZ*4 + i*8;
				switch (colorMode) {
				case 0:
					color[location + 2] = Vec3f(0, 0, 1);
					color[location + 3] = Vec3f(0, 0, 1);
					color[location + 6] = Vec3f(0, 0, 1);
					color[location + 7] = Vec3f(0, 0, 1);
					break;
				case 1:
					color[location + 2] = Vec3f(
						1 - height[ih][numCellsZ+1]/(3/2),
						(height[ih][numCellsZ+1] < 3/2) ? height[ih][numCellsZ+1]/(3/2) : 1 - (height[ih][numCellsZ+1] - 3/2)/(3/2),
						(height[ih][numCellsZ+1] - 3/2)/(3/2));
					color[location + 3] = Vec3f(
						1 - height[ih+1][numCellsZ+1]/(3/2),
						(height[ih+1][numCellsZ+1] < 3/2) ? height[ih+1][numCellsZ+1]/(3/2) : 1 - (height[ih+1][numCellsZ+1] - 3/2)/(3/2),
						(height[ih+1][numCellsZ+1] - 3/2)/(3/2));
					color[location + 6] = Vec3f(
						1 - height[ih+1][1]/(3/2),
						(height[ih+1][1] < 3/2) ? height[ih+1][1]/(3/2) : 1 - (height[ih+1][1] - 3/2)/(3/2),
						(height[ih+1][1] - 3/2)/(3/2));
					color[location + 7] = Vec3f(
						1 - height[ih][1]/(3/2),
						(height[ih][1] < 3/2) ? height[ih][1]/(3/2) : 1 - (height[ih][1] - 3/2)/(3/2),
						(height[ih][1] - 3/2)/(3/2));
					break;
				case 2:
					color[location + 2] = normal[location + 2];
					color[location + 3] = normal[location + 3];
					color[location + 6] = normal[location + 6];
					color[location + 7] = normal[location + 7];
					break;
				case 3:
					color[location] = Vec3f(
						velocityX[ih][numCellsZ+1],
						velocityZ[ih][numCellsZ+1],
						0);

					color[location + 1] = Vec3f(
						velocityX[ih+1][numCellsZ+1],
						velocityZ[ih+1][numCellsZ+1],
						0);

					color[location + 2] = Vec3f(
						velocityX[ih+1][1],
						velocityZ[ih+1][1],
						0);

					color[location + 3] = Vec3f(
						velocityX[ih][1],
						velocityZ[ih][1],
						0);
					break;

				default:;
				}
			}

			// Start and end faces
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int i = 0; i < numCellsZ; i++) {
				int ih = i + 1;
				position[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 2][1] = height[1][ih];
				position[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 3][1] = height[1][ih+1];
				position[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 6][1] = height[numCellsX+1][ih];
				position[numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 7][1] = height[numCellsX+1][ih+1];
				int location = numCellsX*numCellsZ*4 + numCellsX*8 + i*8;
				switch (colorMode) {
				case 0:
					color[location + 2] = Vec3f(0, 0, 1);
					color[location + 3] = Vec3f(0, 0, 1);
					color[location + 6] = Vec3f(0, 0, 1);
					color[location + 7] = Vec3f(0, 0, 1);
					break;
				case 1:
					color[location + 2] = Vec3f(
						1 - height[numCellsX+1][ih]/(3/2),
						(height[numCellsX+1][ih] < 3/2) ? height[numCellsX+1][ih]/(3/2) : 1 - (height[numCellsX+1][ih] - 3/2)/(3/2),
						(height[numCellsX+1][ih] - 3/2)/(3/2));
					color[location + 3] = Vec3f(
						1 - height[numCellsX+1][ih+1]/(3/2),
						(height[numCellsX+1][ih+1] < 3/2) ? height[numCellsX+1][ih+1]/(3/2) : 1 - (height[numCellsX+1][ih+1] - 3/2)/(3/2),
						(height[numCellsX+1][ih+1] - 3/2)/(3/2));
					color[location + 6] = Vec3f(
						1 - height[1][ih]/(3/2),
						(height[1][ih] < 3/2) ? height[1][ih]/(3/2) : 1 - (height[1][ih] - 3/2)/(3/2),
						(height[1][ih] - 3/2)/(3/2));
					color[location + 7] = Vec3f(
						1 - height[ih][ih+1]/(3/2),
						(height[ih+1][ih+1] < 3/2) ? height[ih][ih+1]/(3/2) : 1 - (height[ih][ih+1] - 3/2)/(3/2),
						(height[ih][ih+1] - 3/2)/(3/2));
					break;
				case 2:
					color[location + 2] = normal[location + 2];
					color[location + 3] = normal[location + 3];
					color[location + 6] = normal[location + 6];
					color[location + 7] = normal[location + 7];
					break;
				case 3:
					color[location] = Vec3f(
						velocityX[numCellsX+1][ih],
						velocityZ[numCellsX+1][ih],
						0);

					color[location + 1] = Vec3f(
						velocityX[numCellsX+1][ih+1],
						velocityZ[numCellsX+1][ih+1],
						0);

					color[location + 2] = Vec3f(
						velocityX[1][ih],
						velocityZ[1][ih],
						0);

					color[location + 3] = Vec3f(
						velocityX[1][ih+1],
						velocityZ[1][ih+1],
						0);
					break;

				default:;
				}
			}
		}
	}

	void render() {
		using namespace Globals;
		glBindBuffer(GL_ARRAY_BUFFER, position_vbo[0]);
		if (!position.empty())
			glBufferData(GL_ARRAY_BUFFER, position.size() * sizeof(position[0]), &position[0][0], GL_DYNAMIC_DRAW);
		else
			glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ARRAY_BUFFER, colors_vbo[0]);
		if (!color.empty())
			glBufferData(GL_ARRAY_BUFFER, color.size() * sizeof(color[0]), &color[0][0], GL_DYNAMIC_DRAW);
		else
			glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ARRAY_BUFFER, normals_vbo[0]);
		if (!normal.empty())
			glBufferData(GL_ARRAY_BUFFER, normal.size() * sizeof(normal[0]), &normal[0][0], GL_DYNAMIC_DRAW);
		else
			glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glDrawElements(GL_TRIANGLES, faces.size(), GL_UNSIGNED_INT, (GLvoid*)0);
	}

	void updateFaces() {
		using namespace Globals;
		// Create buffer for indices
		glGenBuffers(1, faces_ibo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, faces_ibo[0]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.size() * sizeof(faces[0]), &faces[0], GL_STATIC_DRAW);
	}

	bool holdWave = false; // pauses the wave equation
	double force = 0.1; // how much height you add to the water when interacting with it
	bool zone[10]; // needed for adding in multiple spots simultaneously

private:
	void stopVelocity() {
		#if USE_OMP
			#pragma omp parallel for
		#endif
		for (int i = 0; i < numCellsX + 3; i++) {
			#if USE_OMP
				#pragma omp parallel for
			#endif
			for (int j = 0; j < numCellsZ + 3; j++) {
				velocityX[i][j] = 0;
				velocityZ[i][j] = 0;
			}
		}
	}

	// Initializes the rendering and simulation vectors
	void initVectors() {
		std::cout << "Shallow Water - # of Cells: " << numCellsX <<"x" << numCellsZ << std::endl;

		height.clear();
		velocityX.clear();
		velocityZ.clear();
		heightX.clear();
		heightZ.clear();
		velocityXMidX.clear();
		velocityXMidZ.clear();
		velocityZMidX.clear();
		velocityZMidZ.clear();

		position.clear();
		color.clear();
		normal.clear();
		faceNormal.clear();
		faces.clear();

		// Initialize all simulation vectors
		for (int i = 0; i < numCellsX + 2; i++) {
			height.push_back(vector<double>());
			velocityX.push_back(vector<double>());
			velocityZ.push_back(vector<double>());
			heightX.push_back(vector<double>());
			heightZ.push_back(vector<double>());
			velocityXMidX.push_back(vector<double>());
			velocityXMidZ.push_back(vector<double>());
			velocityZMidX.push_back(vector<double>());
			velocityZMidZ.push_back(vector<double>());

			for (int j = 0; j < numCellsZ + 2; j++) {
				height[i].push_back(0.0);
				velocityX[i].push_back(0.0);
				velocityZ[i].push_back(0.0);
				heightX[i].push_back(0.0);
				heightZ[i].push_back(0.0);
				velocityXMidX[i].push_back(0.0);
				velocityXMidZ[i].push_back(0.0);
				velocityZMidX[i].push_back(0.0);
				velocityZMidZ[i].push_back(0.0);
			}

			height[i].push_back(0.0);
			velocityX[i].push_back(0.0);
			velocityZ[i].push_back(0.0);
			height[i].push_back(0.0);
			velocityX[i].push_back(0.0);
			velocityZ[i].push_back(0.0);
		}

		height.push_back(vector<double>());
		velocityX.push_back(vector<double>());
		velocityZ.push_back(vector<double>());
		for (int i = 0; i < numCellsZ + 3; i++) {
			height.back().push_back(0.0);
			velocityX.back().push_back(0.0);
			velocityZ.back().push_back(0.0);
		}

		// Initialize all render vectors
		if (smoothShading) {
			// Top faces
			for (int i = 0; i < numCellsX + 1; i++) {
				for (int j = 0; j < numCellsZ + 1; j++) {
					// Vertex positions
					position.push_back(Vec3f(i*cellSizeX, 1.0, -j*cellSizeZ));

					// Vertex colors and normals
					color.push_back(Vec3f(0.0, 0.0, 1.0));
					normal.push_back(Vec3f(0.0, 1.0, 0.0));

					if (i < numCellsX && j < numCellsZ) {
						faceNormal.push_back(Vec3f(0.0, 1.0, 0.0));

						// Face
						faces.push_back(i*(numCellsZ+1) + j);
						faces.push_back((i+1)*(numCellsZ+1) + j);
						faces.push_back((i+1)*(numCellsZ+1) + j + 1);
						faces.push_back(i*(numCellsZ+1) + j);
						faces.push_back((i+1)*(numCellsZ+1) + j + 1);
						faces.push_back(i*(numCellsZ+1) + j + 1);
					}
				}
			}

			// Back faces
			for (int i = 0; i < numCellsX + 1; i++) {
				position.push_back(Vec3f(i*cellSizeX, 1.0, -numCellsZ*cellSizeZ));
				position.push_back(Vec3f(i*cellSizeX, 0.0, -numCellsZ*cellSizeZ));

				color.push_back(Vec3f(0.0, 0.0, 1.0));
				normal.push_back(Vec3f(0.0, 1.0, 0.0));
				color.push_back(Vec3f(0.0, 0.0, 1.0));
				normal.push_back(Vec3f(0.0, 1.0, 0.0));

				if (i < numCellsX) {
					faces.push_back((numCellsX+1)*(numCellsZ+1) + i*2);
					faces.push_back((numCellsX+1)*(numCellsZ+1) + i*2 + 2);
					faces.push_back((numCellsX+1)*(numCellsZ+1) + i*2 + 3);
					faces.push_back((numCellsX+1)*(numCellsZ+1) + i*2);
					faces.push_back((numCellsX+1)*(numCellsZ+1) + i*2 + 3);
					faces.push_back((numCellsX+1)*(numCellsZ+1) + i*2 + 1);
				}
			}

			// Front faces
			for (int i = 0; i < numCellsX + 1; i++) {
				position.push_back(Vec3f(i*cellSizeX, 1.0, 0.0));
				position.push_back(Vec3f(i*cellSizeX, 0.0, 0.0));

				color.push_back(Vec3f(0.0, 0.0, 1.0));
				normal.push_back(Vec3f(0.0, 1.0, 0.0));
				color.push_back(Vec3f(0.0, 0.0, 1.0));
				normal.push_back(Vec3f(0.0, 1.0, 0.0));

				if (i < numCellsX) {
					// Back face
					faces.push_back((numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*2 + i*2);
					faces.push_back((numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*2 + i*2 + 2);
					faces.push_back((numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*2 + i*2 + 3);
					faces.push_back((numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*2 + i*2);
					faces.push_back((numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*2 + i*2 + 3);
					faces.push_back((numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*2 + i*2 + 1);
				}
			}

			// Start faces
			for (int i = 0; i < numCellsZ + 1; i++) {
				position.push_back(Vec3f(0.0, 1.0, -i*cellSizeZ));
				position.push_back(Vec3f(0.0, 0.0, -i*cellSizeZ));

				color.push_back(Vec3f(0.0, 0.0, 1.0));
				normal.push_back(Vec3f(0.0, 1.0, 0.0));
				color.push_back(Vec3f(0.0, 0.0, 1.0));
				normal.push_back(Vec3f(0.0, 1.0, 0.0));

				if (i < numCellsZ) {
					faces.push_back((numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + i*2);
					faces.push_back((numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + i*2 + 2);
					faces.push_back((numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + i*2 + 3);
					faces.push_back((numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + i*2);
					faces.push_back((numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + i*2 + 3);
					faces.push_back((numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + i*2 + 1);
				}
			}

			// End faces
			for (int i = 0; i < numCellsZ + 1; i++) {
				position.push_back(Vec3f(cellSizeX*numCellsX, 1.0, -i*cellSizeZ));
				position.push_back(Vec3f(cellSizeX*numCellsX, 0.0, -i*cellSizeZ));

				color.push_back(Vec3f(0.0, 0.0, 1.0));
				normal.push_back(Vec3f(0.0, 1.0, 0.0));
				color.push_back(Vec3f(0.0, 0.0, 1.0));
				normal.push_back(Vec3f(0.0, 1.0, 0.0));

				if (i < numCellsZ) {
					faces.push_back((numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*2 + i*2);
					faces.push_back((numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*2 + i*2 + 2);
					faces.push_back((numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*2 + i*2 + 3);
					faces.push_back((numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*2 + i*2);
					faces.push_back((numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*2 + i*2 + 3);
					faces.push_back((numCellsX+1)*(numCellsZ+1) + (numCellsX+1)*4 + (numCellsZ+1)*2 + i*2 + 1);
				}
			}

			// Bottom face
			position.push_back(Vec3f(0.0, 0.0, 0.0));
			position.push_back(Vec3f(0.0, 0.0, -numCellsZ*cellSizeZ));
			position.push_back(Vec3f(numCellsX*cellSizeX, 0.0, 0.0));
			position.push_back(Vec3f(numCellsX*cellSizeX, 0.0, -numCellsZ*cellSizeZ));
			for (int j = 0; j < 4; j++) {
				color.push_back(Vec3f(0.0, 0.0, 1.0));
				normal.push_back(Vec3f(0.0, 1.0, 0.0));
			}
			faces.push_back((numCellsX+1)*(numCellsZ+1) + (numCellsZ+1)*4 + (numCellsX+1)*4);
			faces.push_back((numCellsX+1)*(numCellsZ+1) + (numCellsZ+1)*4 + (numCellsX+1)*4 + 2);
			faces.push_back((numCellsX+1)*(numCellsZ+1) + (numCellsZ+1)*4 + (numCellsX+1)*4 + 3);
			faces.push_back((numCellsX+1)*(numCellsZ+1) + (numCellsZ+1)*4 + (numCellsX+1)*4);
			faces.push_back((numCellsX+1)*(numCellsZ+1) + (numCellsZ+1)*4 + (numCellsX+1)*4 + 3);
			faces.push_back((numCellsX+1)*(numCellsZ+1) + (numCellsZ+1)*4 + (numCellsX+1)*4 + 1);
		}
		else {
			// Top faces
			for (int i = 0; i < numCellsX; i++) {
				for (int j = 0; j < numCellsZ; j++) {
					// Vertex positions
					position.push_back(Vec3f(i*cellSizeX, 1.0, -j*cellSizeZ));
					position.push_back(Vec3f(i*cellSizeX, 1.0, -(j+1)*cellSizeZ));
					position.push_back(Vec3f((i+1)*cellSizeX, 1.0, -j*cellSizeZ));
					position.push_back(Vec3f((i+1)*cellSizeX, 1.0, -(j+1)*cellSizeZ));

					// Vertex colors and normals
					for (int j = 0; j < 4; j++) {
						color.push_back(Vec3f(0.0, 0.0, 1.0));
						normal.push_back(Vec3f(0.0, 1.0, 0.0));
					}

					// Face
					faces.push_back(i*numCellsZ*4 + j*4);
					faces.push_back(i*numCellsZ*4 + j*4 + 2);
					faces.push_back(i*numCellsZ*4 + j*4 + 3);
					faces.push_back(i*numCellsZ*4 + j*4);
					faces.push_back(i*numCellsZ*4 + j*4 + 3);
					faces.push_back(i*numCellsZ*4 + j*4 + 1);
				}
			}

			// Back and front faces
			for (int i = 0; i < numCellsX; i++) {
				// Back vertices
				position.push_back(Vec3f(i*cellSizeX, 0.0, -numCellsZ*cellSizeZ));
				position.push_back(Vec3f((i+1)*cellSizeX, 0.0, -numCellsZ*cellSizeZ));
				position.push_back(Vec3f(i*cellSizeX, 1.0, -numCellsZ*cellSizeZ));
				position.push_back(Vec3f((i+1)*cellSizeX, 1.0, -numCellsZ*cellSizeZ));
				// Front vertices
				position.push_back(Vec3f((i+1)*cellSizeX, 0.0, 0.0));
				position.push_back(Vec3f(i*cellSizeX, 0.0, 0.0));
				position.push_back(Vec3f((i+1)*cellSizeX, 1.0, 0.0));
				position.push_back(Vec3f(i*cellSizeX, 1.0, 0.0));

				// Vertex colors and normals
				for (int j = 0; j < 8; j++) {
					color.push_back(Vec3f(0.0, 0.0, 1.0));
					normal.push_back(Vec3f(0.0, 1.0, 0.0));
				}

				// Back face
				faces.push_back(numCellsX*numCellsZ*4 + i*8);
				faces.push_back(numCellsX*numCellsZ*4 + i*8 + 2);
				faces.push_back(numCellsX*numCellsZ*4 + i*8 + 3);
				faces.push_back(numCellsX*numCellsZ*4 + i*8);
				faces.push_back(numCellsX*numCellsZ*4 + i*8 + 3);
				faces.push_back(numCellsX*numCellsZ*4 + i*8 + 1);
				// Front face
				faces.push_back(numCellsX*numCellsZ*4 + i*8 + 4);
				faces.push_back(numCellsX*numCellsZ*4 + i*8 + 6);
				faces.push_back(numCellsX*numCellsZ*4 + i*8 + 7);
				faces.push_back(numCellsX*numCellsZ*4 + i*8 + 4);
				faces.push_back(numCellsX*numCellsZ*4 + i*8 + 7);
				faces.push_back(numCellsX*numCellsZ*4 + i*8 + 5);
			}

			// Start and end faces
			for (int i = 0; i < numCellsZ; i++) {
				// Start vertices
				position.push_back(Vec3f(0.0, 0.0, -i*cellSizeZ));
				position.push_back(Vec3f(0.0, 0.0, -(i+1)*cellSizeZ));
				position.push_back(Vec3f(0.0, 1.0, -i*cellSizeZ));
				position.push_back(Vec3f(0.0, 1.0, -(i+1)*cellSizeZ));
				// End vertices
				position.push_back(Vec3f(cellSizeX*numCellsX, 0.0, -i*cellSizeZ));
				position.push_back(Vec3f(cellSizeX*numCellsX, 0.0, -(i+1)*cellSizeZ));
				position.push_back(Vec3f(cellSizeX*numCellsX, 1.0, -i*cellSizeZ));
				position.push_back(Vec3f(cellSizeX*numCellsX, 1.0, -(i+1)*cellSizeZ));

				// Vertex colors and normals
				for (int j = 0; j < 8; j++) {
					color.push_back(Vec3f(0.0, 0.0, 1.0));
					normal.push_back(Vec3f(0.0, 1.0, 0.0));
				}

				// Start face
				faces.push_back(numCellsX*numCellsZ*4 + numCellsX*8 + i*8);
				faces.push_back(numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 2);
				faces.push_back(numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 3);
				faces.push_back(numCellsX*numCellsZ*4 + numCellsX*8 + i*8);
				faces.push_back(numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 3);
				faces.push_back(numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 1);
				// End face
				faces.push_back(numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 4);
				faces.push_back(numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 6);
				faces.push_back(numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 7);
				faces.push_back(numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 4);
				faces.push_back(numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 7);
				faces.push_back(numCellsX*numCellsZ*4 + numCellsX*8 + i*8 + 5);
			}

			// Bottom face
			position.push_back(Vec3f(0.0, 0.0, 0.0));
			position.push_back(Vec3f(0.0, 0.0, -numCellsZ*cellSizeZ));
			position.push_back(Vec3f(numCellsX*cellSizeX, 0.0, 0.0));
			position.push_back(Vec3f(numCellsX*cellSizeX, 0.0, -numCellsZ*cellSizeZ));
			for (int j = 0; j < 4; j++) {
				color.push_back(Vec3f(0.0, 0.0, 1.0));
				normal.push_back(Vec3f(0.0, 1.0, 0.0));
			}
			faces.push_back(numCellsX*numCellsZ*4 + numCellsZ*8 + numCellsX*8);
			faces.push_back(numCellsX*numCellsZ*4 + numCellsZ*8 + numCellsX*8 + 2);
			faces.push_back(numCellsX*numCellsZ*4 + numCellsZ*8 + numCellsX*8 + 3);
			faces.push_back(numCellsX*numCellsZ*4 + numCellsZ*8 + numCellsX*8);
			faces.push_back(numCellsX*numCellsZ*4 + numCellsZ*8 + numCellsX*8 + 3);
			faces.push_back(numCellsX*numCellsZ*4 + numCellsZ*8 + numCellsX*8 + 1);
		}

		setupWave(0);
		updateAllNormals();
	}

	int numCellsX; // number of cells in x direction of the simulation
	int numCellsZ; // number of cells in z direction of the simulation
	double cellSizeX; // size of individual cell in x direction
	double cellSizeZ; // size of individual cell in z direction
	double damping; // damping constant for stability
	
	vector<vector<double>> height;
	vector<vector<double>> velocityX;
	vector<vector<double>> velocityZ;

	vector<vector<double>> heightX;
	vector<vector<double>> velocityXMidX;
	vector<vector<double>> velocityXMidZ;
	vector<vector<double>> heightZ;
	vector<vector<double>> velocityZMidX;
	vector<vector<double>> velocityZMidZ;

	int colorMode = 0;

	// Rendering values
	vector<Vec3f> position;
	vector<Vec3f> color;
	vector<Vec3f> normal;
	vector<Vec3f> faceNormal;
	vector<GLuint> faces;
	bool smoothShading = true;

};

namespace Globals {
	ShallowWater sw;
}

void updateViewProjection() {
	using namespace Globals;

	// Calculate the orthogonal axes based on the viewing parameters
	Vec3f n = viewDir * (-1.f / viewDir.len());
	Vec3f u = upDir.cross(n);
	u.normalize();
	Vec3f v = n.cross(u);

	// Calculate the translation based on the new axes
	float dx = -(eye.dot(u));
	float dy = -(eye.dot(v));
	float dz = -(eye.dot(n));

	// Fill in the matrix
	view.m[0] = u[0];	view.m[4] = u[1];	view.m[8] = u[2];	view.m[12] = dx;
	view.m[1] = v[0];	view.m[5] = v[1];	view.m[9] = v[2];	view.m[13] = dy;
	view.m[2] = n[0];	view.m[6] = n[1];	view.m[10] = n[2];	view.m[14] = dz;
	view.m[3] = 0;		view.m[7] = 0;		view.m[11] = 0;		view.m[15] = 1;
}

//
//	Callbacks
//
static void error_callback(int error, const char* description){ fprintf(stderr, "Error: %s\n", description); }

// function that is called whenever a mouse or trackpad button press event occurs
static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		glfwGetCursorPos(window, &Globals::mouse.prev_x, &Globals::mouse.prev_y);
		Globals::mouse.active = true;
	}
}

// Function to rotate the viewing transform about the y axis
static const Mat4x4 rotateY(float theta) {
	float t = theta*PI/180.f;
	
	Mat4x4 mat;
	mat.m[0] = cos(t);		mat.m[4] = 0.f;		mat.m[8] = sin(t);		mat.m[12] = 0.f;
	mat.m[1] = 0.f;			mat.m[5] = 1.f;		mat.m[9] = 0.f;			mat.m[13] = 0.f;
	mat.m[2] = -sin(t);		mat.m[6] = 0.f;		mat.m[10] = cos(t);		mat.m[14] = 0.f;
	mat.m[3] = 0.f;			mat.m[7] = 0.f;		mat.m[11] = 0.f;		mat.m[15] = 1.f;
	
	return mat;
}

// Function to rotate the viewing transform about the y axis
static const Mat4x4 rotateX(float phi) {
	float t = phi*PI/180.f;
	
	Mat4x4 mat;
	mat.m[0] = 1.f;		mat.m[4] = 0.f;		mat.m[8] = 0.f;			mat.m[12] = 0.f;
	mat.m[1] = 0.f;		mat.m[5] = cos(t);	mat.m[9] = -sin(t);		mat.m[13] = 0.f;
	mat.m[2] = 0.f;		mat.m[6] = sin(t);	mat.m[10] = cos(t);		mat.m[14] = 0.f;
	mat.m[3] = 0.f;		mat.m[7] = 0.f;		mat.m[11] = 0.f;		mat.m[15] = 1.f;
	
	return mat;
}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
	using namespace Globals;
	if (!mouse.active)
		return;

	viewDir = Vec3f(0, 0, -1);
	upDir = Vec3f(0, 1, 0);
	
	if (xpos != mouse.prev_x) {	
		theta -= 0.2*(xpos - mouse.prev_x);
		yRot = rotateY(theta);
		mouse.prev_x = xpos;
	}

	if (ypos != mouse.prev_y) {
		phi -= 0.2*(ypos - mouse.prev_y);
		if (phi > 89)
			phi = 89;
		else if (phi < -89)
			phi = -89;
		xRot = rotateX(phi);
		mouse.prev_y = ypos;
	}
	
	viewDir = xRot*viewDir;
	viewDir = yRot*viewDir;

	upDir = xRot*upDir;
	upDir = yRot*upDir;

	rightDir = upDir.cross(viewDir);
}


static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods){
	using namespace Globals;

	bool smooth;

	// Close on escape or Q
	if (action == GLFW_PRESS) {
		switch (key) {
			// Close on escape
		case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, GL_TRUE); break;
			// Pause the simulation
		case GLFW_KEY_SPACE: if (pause) pause = false;
							 else pause = true;
							 std::cout << ((pause) ? "Paused" : "Unpaused") << std::endl;
							 break;

							 // Movement keys trigger booleans to be processed during the graphics loop
							 // Forward movement
		case GLFW_KEY_W: key_w = true; break;

			// Backward movement
		case GLFW_KEY_S: key_s = true; break;

			// Right strafing movement
		case GLFW_KEY_D: key_d = true; break;

			// Left strafing movement
		case GLFW_KEY_A: key_a = true; break;

			// Upward movement
		case GLFW_KEY_E: key_e = true; break;

			// Downward movement
		case GLFW_KEY_Q: key_q = true; break;

			// Speed up
		case GLFW_KEY_LEFT_SHIFT: key_lshift = true; break;

			// Release mouse
		case GLFW_KEY_KP_ENTER:
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			mouse.active = false;
			break;

			// Increase/decrease game speed
		case GLFW_KEY_MINUS: subTimeMultiplier = true; std::cout << "Decreasing dt multiplier..." << std::endl; break;
		case GLFW_KEY_EQUAL: addTimeMultiplier = true; std::cout << "Increasing dt multiplier..." << std::endl; break;
		case GLFW_KEY_BACKSPACE: timeMultiplier = 1.0; std::cout << "Reseting dt multiplier..." << std::endl; break;

			// Setup waves
		case GLFW_KEY_1: sw.setupWave(1); sw.holdWave = true; break;
		case GLFW_KEY_2: sw.setupWave(2); sw.holdWave = true; break;
		case GLFW_KEY_3: sw.setupWave(3); sw.holdWave = true; break;
		case GLFW_KEY_4: sw.setupWave(4); sw.holdWave = true; break;
		case GLFW_KEY_5: sw.setupWave(5); sw.holdWave = true; break;
		case GLFW_KEY_6: sw.setupWave(6); sw.holdWave = true; break;
		case GLFW_KEY_7: sw.setupWave(7); sw.holdWave = true; break;
		case GLFW_KEY_8: sw.setupWave(8); sw.holdWave = true; break;
		case GLFW_KEY_9: sw.setupWave(9); sw.holdWave = true; break;

			// Interact with waves
		case GLFW_KEY_KP_ADD: sw.force += .1; std::cout << "Shallow Water - Interaction force = " << sw.force << std::endl; break;
		case GLFW_KEY_KP_SUBTRACT: sw.force -= .1; std::cout << "Shallow Water - Interaction force = " << sw.force << std::endl;; break;
		case GLFW_KEY_KP_0: sw.interact(0); break;
		case GLFW_KEY_KP_1: sw.interact(1); break;
		case GLFW_KEY_KP_2: sw.interact(2); break;
		case GLFW_KEY_KP_3: sw.interact(3); break;
		case GLFW_KEY_KP_4: sw.interact(4); break;
		case GLFW_KEY_KP_5: sw.interact(5); break;
		case GLFW_KEY_KP_6: sw.interact(6); break;
		case GLFW_KEY_KP_7: sw.interact(7); break;
		case GLFW_KEY_KP_8: sw.interact(8); break;
		case GLFW_KEY_KP_9: sw.interact(9); break;

			// Change the water preset
		case GLFW_KEY_TAB:
			preset = (preset+1)%NUMPRESETS;
			sw.changeWater(preset);
			sw.updateFaces();
			break;

			// Change the water lighting
		case GLFW_KEY_KP_DECIMAL:
			smooth = sw.changeShading();
			sw.updateFaces();
			std::cout << "Shallow Water - Shading: " << ((smooth) ? "Interpolated" : "Not interpolated" ) << std::endl;
			break;

			// Change the water color
		case GLFW_KEY_GRAVE_ACCENT:
			sw.changeColorMode();
			break;
		}

	}
	else if ( action == GLFW_RELEASE ) {
		switch ( key ) {
			// Movement keys trigger booleans to be processed during the graphics loop
			// Forward movement
		case GLFW_KEY_W: key_w = false; break;

			// Backward movement
		case GLFW_KEY_S: key_s = false; break;

			// Right strafing movement
		case GLFW_KEY_D: key_d = false; break;

			// Left strafing movement
		case GLFW_KEY_A: key_a = false; break;

			// Upward movement
		case GLFW_KEY_E: key_e = false; break;

			// Downward movement
		case GLFW_KEY_Q: key_q = false; break;

			// Speed up
		case GLFW_KEY_LEFT_SHIFT: key_lshift = false; break;

			// Increase/decrease game speed
		case GLFW_KEY_MINUS: subTimeMultiplier = false; break;
		case GLFW_KEY_EQUAL: addTimeMultiplier = false;

			// Setup waves
		case GLFW_KEY_1:
		case GLFW_KEY_2:
		case GLFW_KEY_3:
		case GLFW_KEY_4:
		case GLFW_KEY_5:
		case GLFW_KEY_6:
		case GLFW_KEY_7:
		case GLFW_KEY_8:
		case GLFW_KEY_9: sw.holdWave = false; break;
		}
	}
}

void updatePerspectiveProjection() {
	using namespace Globals;

	for (int i = 0; i < 15; i++) {
		projection.m[i] = 0;
	}
	left = aspect * bottom;
	right = aspect * top;
	//diagonal values done first
	projection.m[0] = 2 * near / (right - left);
	projection.m[5] = 2 * near / (top - bottom);
	projection.m[10] = -(near + far) / (far - near);
	projection.m[15] = 0;
	//other values are then calculated.
	projection.m[8] = (right + left) / (right - left);
	projection.m[9] = (top + bottom) / (top - bottom);
	projection.m[14] = -2 * far*near / (far - near);
	projection.m[11] = -1;
}

static void framebuffer_size_callback(GLFWwindow* window, int width, int height){
	Globals::win_width = float(width);
	Globals::win_height = float(height);
    Globals::aspect = Globals::win_width/Globals::win_height;
	
    glViewport(0,0,width,height);

	// ToDo: update the perspective matrix according to the new window size
	updatePerspectiveProjection();
}





// Function to set up geometry
void init_scene();

//
//	Main
//
int main(int argc, char *argv[]){

	// Set up window
	GLFWwindow* window;
	glfwSetErrorCallback(&error_callback);

	// Initialize the window
	if( !glfwInit() ){ return EXIT_FAILURE; }

	// Ask for OpenGL 3.2
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

	// Create the glfw window
	Globals::win_width = WIN_WIDTH;
	Globals::win_height = WIN_HEIGHT;
	window = glfwCreateWindow(int(Globals::win_width), int(Globals::win_height), "Shallow Water Simulation", NULL, NULL);
	if( !window ){ glfwTerminate(); return EXIT_FAILURE; }
	Globals::activeWindow = window;
	// Bind callbacks to the window
	glfwSetKeyCallback(window, &key_callback);
	glfwSetFramebufferSizeCallback(window, &framebuffer_size_callback);

	// Make current
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	// Initialize glew AFTER the context creation and before loading the shader.
	// Note we need to use experimental because we're using a modern version of opengl.
	#ifdef USE_GLEW
		glewExperimental = GL_TRUE;
		glewInit();
	#endif

	// Initialize the shader (which uses glew, so we need to init that first).
	// MY_SRC_DIR is a define that was set in CMakeLists.txt which gives
	// the full path to this project's src/ directory.
	std::stringstream ss; ss << MY_SRC_DIR << "shader.";
	Globals::currShader.init_from_files( ss.str()+"vert", ss.str()+"frag" );

	// Initialize the scene
	// IMPORTANT: Only call after gl context has been created
	init_scene();
	framebuffer_size_callback(window, int(Globals::win_width), int(Globals::win_height));

	// Enable the shader, this allows us to set uniforms and attributes
	Globals::currShader.enable();
	glBindVertexArray(Globals::scene_vao);

	// Initialize OpenGL
	glEnable(GL_DEPTH_TEST);
	glClearColor(1.0, 1.0, 1.0, 1.f);

	updatePerspectiveProjection();
	updateViewProjection();

	double timePassed = 0;
	double dt = 0;
	int frames = 0;
	double counter = 0;
	int seconds = 0;

	using namespace Globals;

	sw.updateFaces();

	int framesPassed = 0;

	// Game loop
	while (!glfwWindowShouldClose(window)) {

		framesPassed++;

		// Clear screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		prevTime = currTime;
		currTime = glfwGetTimerValue();
		timePassed = (double)(currTime - prevTime) / glfwGetTimerFrequency();
		
		// Prevents the simulation from destabilizing due to computation freeze (e.g. dragging the window)
		if (DEBUG && timePassed >= MAXDT) {
			std::cout << "Update cycle past " << MAXDT << "s, refreshing..." << std::endl;
			continue;
		}
		dt = timePassed*timeMultiplier;

		// PHYSICS UPDATE
		if (!pause) {
			int updateFrames = 10;
			double subDT = dt/updateFrames;
			for (int i = 0; i < updateFrames; i++) {
				sw.wave(subDT);
			}
			sw.updatePositions();
			sw.updateTopNormals();
		}
		sw.updateColors();

		// INPUT PROCESSING
		if (addTimeMultiplier)
			timeMultiplier = std::min(1.0, timeMultiplier + .01);
		if (subTimeMultiplier)
			timeMultiplier = std::max(0.01, timeMultiplier - .01);

		if (key_lshift)
			movementSpeed += .01;
		else
			movementSpeed = .1;

		if (key_w) // Move the camera forward
			eye += viewDir*movementSpeed;
		if (key_s) // Move the camera backward
			eye += viewDir*(-movementSpeed);
		if (key_a) // Move the camera leftward
			eye += rightDir*movementSpeed;
		if (key_d) // Move the camera rightward
			eye += rightDir*(-movementSpeed);
		if (key_e) // Move the camera upward
			eye += upDir*movementSpeed;
		if (key_q) // Move the camera downward
			eye += upDir*(-movementSpeed);
		
		// Send updated info to the GPU
		updateViewProjection();

		// FRAME RATE DISPLAY
		frames++;
		counter += timePassed;
		if (counter >= 1.0) {
			//std::cout << "S" << seconds << " - ";
			std::cout << "FPS: " << frames << std::endl;
			frames = 0;
			counter -= 1.0;
			seconds++;
		}

		// RENDERING
		//glUniformMatrix4fv( shader.uniform("model"), 1, GL_FALSE, model.m  ); // model transformation
		glUniformMatrix4fv(Globals::currShader.uniform("view"), 1, GL_FALSE, view.m); // viewing transformation
		glUniformMatrix4fv(Globals::currShader.uniform("projection"), 1, GL_FALSE, projection.m); // projection matrix
		glUniform3f(Globals::currShader.uniform("viewdir"), viewDir[0], viewDir[1], viewDir[2]);
		glUniform3f(Globals::currShader.uniform("light"), lightDir[0], lightDir[1], lightDir[2]);

		sw.render();

		// Finalize
		glfwSwapBuffers(window);
		glfwPollEvents();

		
	} // end game loop
	// Unbind
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	// Disable the shader, we're done using it
	Globals::currShader.disable();

	return EXIT_SUCCESS;
}


void init_scene(){
	using namespace Globals;

	// Define the keyboard callback function
	glfwSetKeyCallback(activeWindow, key_callback);
	// Define the mouse button callback function
	glfwSetMouseButtonCallback(activeWindow, mouse_button_callback);
	// Define the mouse motion callback function
	glfwSetCursorPosCallback(activeWindow, cursor_position_callback);
	viewDir = Vec3f(0, -1, -1);
	phi = -45;
	xRot = rotateX(phi);
	upDir = Vec3f(0, 1, 0);
	rightDir = Vec3f(-1, 0, 0);
	eye = Vec3f(.5, 2, 1);

	// Initialize shallow water
	sw = ShallowWater(preset);

	glGenVertexArrays(1, &scene_vao);
	glBindVertexArray(scene_vao);

	glPointSize(10.0);
	//glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glGenBuffers(1, position_vbo);
	// Create the buffer for colors
	glGenBuffers(1, colors_vbo);
	//create buffer for normals
	glGenBuffers(1, normals_vbo);

	//GLint glnormal = glGetAttribLocation(currShader.program_id, "normal");
	//particle position
	glEnableVertexAttribArray(currShader.attribute("in_position"));
	glBindBuffer(GL_ARRAY_BUFFER, position_vbo[0]);
	glVertexAttribPointer(currShader.attribute("in_position"), 3, GL_FLOAT, GL_FALSE, 0, 0);
	//particle color
	glEnableVertexAttribArray(currShader.attribute("in_color"));
	glBindBuffer(GL_ARRAY_BUFFER, colors_vbo[0]);
	glVertexAttribPointer(currShader.attribute("in_color"), 3, GL_FLOAT, GL_FALSE, 0, 0);
	//particle normal
	glEnableVertexAttribArray(currShader.attribute("in_normal"));
	glBindBuffer(GL_ARRAY_BUFFER, normals_vbo[0]);
	glVertexAttribPointer(currShader.attribute("in_normal"), 3, GL_FLOAT, GL_TRUE, 0, 0);

	glUniform3f(currShader.uniform("light"), lightDir[0], lightDir[1], lightDir[2]);
	glUniform3f(currShader.uniform("viewdir"), viewDir[0], viewDir[1], viewDir[2]);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//done setting data for VAO
	glBindVertexArray(0);
	currTime = glfwGetTimerValue();
}
