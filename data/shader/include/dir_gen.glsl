// u = dot(prev_dir, next_dir)
// g and a - parameters of Draine's phase function
float evalDraine(in float u, in float g, in float a)
{
	const float g2 = g * g;
	return ((1 - g2) * (1 + a * u * u)) / (4. * (1 + (a * (1 + 2 * g2)) / 3.) * PI * pow(1 + g2 - 2 * g * u, 1.5));
}


// u = dot(prev_dir, next_dir)
float ApproximateMie(in float u)
{
	return mix(evalDraine(u, gHG, 0.0), evalDraine(u, gD, draineA), wD);
}


float hg_phase_func(const float cos_theta)
{
	const float g = VOLUME_G;
	const float g2 = g * g;
	const float result = 0.5 * (1 - g2) / pow(1 + g2 - (2 * g * cos_theta), 1.5);
	return result;
}


mat4 rotationMatrix(vec3 axis, float angle)
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    
    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                0.0,                                0.0,                                0.0,                                1.0);
}

// Importance sample Henyey-Greenstein phase function
// sample: (sample deflection cosine)
float SampleHGAngle(float g)
{
	float cosTheta;
	if (abs(g) < 0.001)
	{
		cosTheta = 1 - 2 * RandFloat(1.0);
	}
	else
	{
		float sqrTerm = (1 - g * g) / (1 - g + (2 * g * RandFloat(1.0)));
		cosTheta = (1 + (g * g) - (sqrTerm * sqrTerm)) / (2 * g);
	}
	return acos(cosTheta);
}

// Importance sample Draine phase function
// sample: (sample an exact deflection cosine)
//   xi = a uniform random real in [0,1]
float SampleDraineAngle(in float g, in float a)
{
	float xi = RandFloat(1.0);

	const float g2 = g * g;
	const float g3 = g * g2;
	const float g4 = g2 * g2;
	const float g6 = g2 * g4;
	const float pgp1_2 = (1 + g2) * (1 + g2);
	const float T1 = (-1 + g2) * (4 * g2 + a * pgp1_2);
	const float T1a = -a + a * g4;
	const float T1a3 = T1a * T1a * T1a;
	const float T2 = -1296 * (-1 + g2) * (a - a * g2) * (T1a) * (4 * g2 + a * pgp1_2);
	const float T3 = 3 * g2 * (1 + g * (-1 + 2 * xi)) + a * (2 + g2 + g3 * (1 + 2 * g2) * (-1 + 2 * xi));
	const float T4a = 432 * T1a3 + T2 + 432 * (a - a * g2) * T3 * T3;
	const float T4b = -144 * a * g2 + 288 * a * g4 - 144 * a * g6;
	const float T4b3 = T4b * T4b * T4b;
	const float T4 = T4a + sqrt(-4 * T4b3 + T4a * T4a);
	const float T4p3 = pow(T4, 1.0 / 3.0);
	const float T6 = (2 * T1a + (48 * pow(2, 1.0 / 3.0) *
		(-(a * g2) + 2 * a * g4 - a * g6)) / T4p3 + T4p3 / (3. * pow(2, 1.0 / 3.0))) / (a - a * g2);
	const float T5 = 6 * (1 + g2) + T6;
	return acos((1 + g2 - pow(-0.5 * sqrt(T5) + sqrt(6 * (1 + g2) - (8 * T3) / (a * (-1 + g2) * sqrt(T5)) - T6) / 2., 2)) / (2. * g));
}

vec3 NewRayDir(vec3 oldRayDir, const bool phaseFuncSampling)
{
	// Assert: length(oldRayDir) == 1.0
	// Assert: rand's are in [0.0, 1.0]

	oldRayDir = normalize(oldRayDir);

	// Get any orthogonal vector
	//return  c<a  ? (b,-a,0) : (0,-c,b)
	vec3 orthoDir = oldRayDir.z < oldRayDir.x ? vec3(oldRayDir.y, -oldRayDir.x, 0.0) : vec3(0.0, -oldRayDir.z, oldRayDir.y);
	orthoDir = normalize(orthoDir);

	// Rotate around that orthoDir
	float angle;
	if (phaseFuncSampling)
	{
		angle = SampleHGAngle(VOLUME_G);
	}
	else
	{
		angle = RandFloat(PI);
	}
	mat4 rotMat = rotationMatrix(orthoDir, angle);
	vec3 newRayDir = (rotMat * vec4(oldRayDir, 1.0)).xyz;

	// Rotate around oldRayDir
	angle = RandFloat(2.0 * PI);
	rotMat = rotationMatrix(oldRayDir, angle);
	newRayDir = (rotMat * vec4(newRayDir, 1.0)).xyz;

	return normalize(newRayDir);
}

vec3 NewRayDirApproximateMie(vec3 oldRayDir, const bool phaseFuncSampling)
{
	// Assert: length(oldRayDir) == 1.0
	// Assert: rand's are in [0.0, 1.0]

	oldRayDir = normalize(oldRayDir);

	// Get any orthogonal vector
	//return  c<a  ? (b,-a,0) : (0,-c,b)
	vec3 orthoDir = oldRayDir.z < oldRayDir.x ? vec3(oldRayDir.y, -oldRayDir.x, 0.0) : vec3(0.0, -oldRayDir.z, oldRayDir.y);
	orthoDir = normalize(orthoDir);

	// Rotate around that orthoDir
	float angle;
	if (phaseFuncSampling)
	{
		if (RandFloat(1.0) < wD)
		{
			// Draine lobe sampling
			angle = SampleDraineAngle(gD, draineA);
		}
		else
		{
			// Henyey-Greenstein lobe sampling
			angle = SampleHGAngle(gHG);
		}
	}
	else
	{
		angle = RandFloat(PI);
	}
	mat4 rotMat = rotationMatrix(orthoDir, angle);
	vec3 newRayDir = (rotMat * vec4(oldRayDir, 1.0)).xyz;

	// Rotate around oldRayDir
	angle = RandFloat(2.0 * PI);
	rotMat = rotationMatrix(oldRayDir, angle);
	newRayDir = (rotMat * vec4(newRayDir, 1.0)).xyz;

	return normalize(newRayDir);
}