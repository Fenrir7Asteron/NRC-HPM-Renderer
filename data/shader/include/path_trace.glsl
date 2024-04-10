const int MAX_UNBIASED_POWER_ESTIMATE_COUNT = 8;

float GetTransmittance(const vec3 start, const vec3 end, const uint count)
{
	const vec3 dir = end - start;
	const float stepSize = length(dir) / float(count);

	if (stepSize == 0.0)
	{
		return 1.0;
	}

	float transmittance = 1.0;
	for (uint i = 0; i < count; i++)
	{
		const float factor = float(i) / float(count);
		const vec3 samplePoint = start + (factor * dir);
		const float density = getDensity(samplePoint);
		const float t_r = exp(-density * stepSize);
		transmittance *= t_r;
	}

	return transmittance;
}

float RatioTrack(const vec3 start, const vec3 end)
{
	const float invMaxDensity = 1.0 / VOLUME_DENSITY_FACTOR;
	//const float invMaxDensity = 1.0;

	const vec3 dir = normalize(end - start);
	const float tMax = distance(end, start);
	float transmittance = 1.0;
	float t = 0.0;
	
	for (uint i = 0; i < 128; i++)
	{
		t -= log(1.0 - RandFloat(1.0)) * invMaxDensity;
		if (t >= tMax) { break; }
		const vec3 nextSamplePoint = start + (t * dir);
		transmittance *= 1.0 - (getDensity(nextSamplePoint) * invMaxDensity);
	}
	
	return transmittance;
}

float GetNForCMF(float controlOpticalThickness)
{
	return ceil(pow((0.015 + controlOpticalThickness) * (0.65 + controlOpticalThickness) * (60.3 + controlOpticalThickness), 1.0 / 3.0));
}

float BKExpectedEvalOrder(float c, float pz)
{
	float K = floor(c);
	float t = 1.0;
	float sum = 1.0;

	for (int k = 1; k <= K; ++k)
	{
		t = t * c / float(k);
		sum += t;
	}

	float En = K + (exp(c) - sum) / t;

	// Non-zero orders are evaluated with probability 1 - pz
	return (1 - pz) * En;
}

int DetermineTupleSize(float controlOpticalThickness, float pz)
{
	const float NForCMF = GetNForCMF(controlOpticalThickness);
	const float NForBK = BKExpectedEvalOrder(2.0, pz); // 0.31945
	return max(1, int(NForCMF / (NForBK + 1.0) + 0.5));
}

float[MAX_UNBIASED_POWER_ESTIMATE_COUNT] ElementaryMeans(float X[MAX_UNBIASED_POWER_ESTIMATE_COUNT], int N, int Z)
{
	float m[MAX_UNBIASED_POWER_ESTIMATE_COUNT];
	m[0] = 1.0;
	for (int k = 1; k <= Z; ++k)
	{
		m[k] = 0.0;
	}

	for (int n = 1; n <= N; ++n)
	{
		for (int k = min(n, Z); k >= 1; --k)
		{
			m[k] = m[k] + (float(k) / float(n)) * (m[k - 1] * X[n] - m[k]);
		}
	}

	return m;
}

int AggressiveBKRoulette(float pz, out float weights[MAX_UNBIASED_POWER_ESTIMATE_COUNT])
{
	float p = 1.0 - pz;
	weights[0] = 1.0;
	float u = RandFloat(1.0);

	// Stop at the zeroth order term with probability pz
	if (p <= u)
		return 0;

	// BK with K = c = 2
	int K = 2;
	float c = 2;

	for (int i = 1; i <= K; ++i)
	{
		weights[i] = 1.0 / p;
	}

	int i = K + 1;
	while (true)
	{
		// Compute the continuation probabilities
		float ci = min(c / float(i), 1.0);
		// Update the probability of sampling at least order i
		p *= ci;
		// Russian roulette termination
		if (p <= u)
		{
			return i - 1;
		}

		// Final weight for order i
		weights[i] = 1.0 / p;

		i += 1;
	}
}

float UnbiasedRaymarchingTransmittanceEstimator(const vec3 start, const vec3 end, float controlOpticalThickness)
{
	const vec3 dir = normalize(end - start);
	float intervalLength = distance(end, start);
	const int M = min(DetermineTupleSize(controlOpticalThickness, 0.9), MAX_UNBIASED_POWER_ESTIMATE_COUNT - 2);

	float weights[MAX_UNBIASED_POWER_ESTIMATE_COUNT];
	const int N = min(AggressiveBKRoulette(0.9, weights), MAX_UNBIASED_POWER_ESTIMATE_COUNT - 2);

	const float d1 = getDensity(start);
	const float d2 = getDensity(start + dir * intervalLength);
	const float sampleLength = (intervalLength / float(M));
	float X[MAX_UNBIASED_POWER_ESTIMATE_COUNT];

	for (int i = 1; i <= N + 1; ++i)
	{
		const float u = RandFloat(1.0);

		X[i] = 0.0;

		for (int j = 0; j <= M - 1; j++)
		{
			X[i] += getDensity(start + dir * (sampleLength * (u + float(j))));
		}

		X[i] = -sampleLength * X[i];
		//X[i] = X[i] - sampleLength * (0.5 - u) * (d2 - d1); // endpoint matching, optional
	}

	float T = 0;
	for (int j = 1; j <= N + 1; ++j)
	{
		float X_remaining[MAX_UNBIASED_POWER_ESTIMATE_COUNT];
		float pivot = X[j];
		for (int k = 1; k < j; ++k)
		{
			X_remaining[k] = X[k] - pivot;
		}
		
		// Remove j-th element from X with offsetting array elements one step left.
		for (int k = j; k <= N; ++k)
		{
			X_remaining[k] = X[k + 1] - pivot;
		}

		float m[MAX_UNBIASED_POWER_ESTIMATE_COUNT] = ElementaryMeans(X_remaining, N, N);
		float temp = 0.0;
		float kFactorial = 1;
		for (int k = 0; k <= N; ++k)
		{
			kFactorial *= float(max(1, k));
			temp += m[k] / (kFactorial * weights[k]);
		}
		T = T + (1.0 / float(N + 1)) * exp(pivot) * temp;
	}

	return T;
}


float BiasedRaymarchingTransmittanceEstimator(const vec3 start, const vec3 end, float controlOpticalThickness)
{
	const vec3 dir = normalize(end - start);
	float intervalLength = distance(end, start);
	const float M = GetNForCMF(controlOpticalThickness);
	const float u = RandFloat(1.0);
	const float d1 = getDensity(start);
	const float d2 = getDensity(start + dir * intervalLength);
	const float sampleLength = (intervalLength / M);

	float X = 0.0;

	for (uint i = 0; i < M; i++)
	{
		X += getDensity(start + dir * (sampleLength * (u + i)));
	}

	X = -sampleLength * X;
	//X = X - sampleLength * (0.5 - u) * (d2 - d1); // endpoint matching, optional
	return exp(X);
}

vec3 TraceDirLight(const vec3 pos, const vec3 dir)
{
	if (dir_light.strength == 0.0)
	{
		return vec3(0.0);
	}

	const float transmittance = BiasedRaymarchingTransmittanceEstimator(pos, find_entry_exit(pos, -normalize(dir_light.dir))[1], VOLUME_DENSITY_FACTOR);
	//const float phase = hg_phase_func(dot(dir_light.dir, -dir));
	const float phase = ApproximateMie(dot(dir_light.dir, -dir));
	const vec3 dirLighting = vec3(1.0f) * transmittance * dir_light.strength * phase;
	return dirLighting;
}

vec3 TracePointLight(const vec3 pos, const vec3 dir)
{
	if (pointLight.strength == 0.0)
	{
		return vec3(0.0);
	}

	const float transmittance = BiasedRaymarchingTransmittanceEstimator(pointLight.pos, pos, VOLUME_DENSITY_FACTOR);
	//const float phase = hg_phase_func(dot(normalize(pointLight.pos - pos), -dir));
	const float phase = ApproximateMie(dot(normalize(pointLight.pos - pos), -dir));
	const vec3 pointLighting = pointLight.color * pointLight.strength * transmittance * phase;
	return pointLighting;
}

vec3 SampleHdrEnvMap(const vec2 dir)
{
	const vec2 invAtan = vec2(0.1591, 0.3183);

	vec2 uv = dir;
    uv *= invAtan;
    uv += 0.5;

	return texture(hdrEnvMap, uv).xyz * HDR_ENV_MAP_STRENGTH;
}

vec3 SampleHdrEnvMap(const vec3 dir)
{
	vec2 phiTheta = vec2(atan(dir.z, dir.x), asin(dir.y));
	return SampleHdrEnvMap(phiTheta);
}

vec3 SampleHdrEnvMap(const vec3 pos, const vec3 dir, uint sampleCount)
{
	if (HDR_ENV_MAP_STRENGTH == 0.0)
	{
		return vec3(0.0);
	}

	vec3 light = vec3(0.0);

	for (uint i = 0; i < sampleCount; i++)
	{
		const vec3 randomDir = NewRayDirApproximateMie(dir, false);
		//const float phase = hg_phase_func(dot(randomDir, -dir));
		const float phase = ApproximateMie(dot(randomDir, -dir));
		const vec3 exit = find_entry_exit(pos, randomDir)[1];
		//const float transmittance = GetTransmittance(pos, exit, 16);
		const float transmittance = BiasedRaymarchingTransmittanceEstimator(pos, exit, VOLUME_DENSITY_FACTOR);
		const vec3 sampleLight = SampleHdrEnvMap(randomDir) * phase * transmittance;

		light += sampleLight;
	}

	// Half env map importance sampled
//	for (uint i = 0; i < sampleCount - halfSampleCount; i++)
//	{
//		const float thetaNorm = texture(hdrEnvMapInvCdfY, RandFloat(1.0)).x;
//		const float phiNorm = texture(hdrEnvMapInvCdfX, vec2(RandFloat(1.0), thetaNorm)).x;
//
//		//const float thetaNorm = 0.458;
//		//const float phiNorm = 0.477;
//
//		const vec3 randomDir = sin(thetaNorm * PI) * vec3(cos(phiNorm * 2.0 * PI), 1.0, sin(phiNorm * 2.0 * PI));
//
//		const float phase = hg_phase_func(dot(randomDir, -dir));
//		const vec3 exit = find_entry_exit(pos, randomDir)[1];
//		const float transmittance = GetTransmittance(pos, exit, 16);
//		const vec3 sampleLight = texture(hdrEnvMap, vec2(phiNorm, thetaNorm)).xyz * hdrEnvMapData.hpmStrength * phase * transmittance;
//
//		light += sampleLight;
//	}

	light /= float(sampleCount);

	return light;
}

vec3 TraceScene(const vec3 pos, const vec3 dir)
{
	const vec3 totalLight = TraceDirLight(pos, dir) + TracePointLight(pos, dir) + SampleHdrEnvMap(pos, dir, 1);
	return totalLight;
}

vec3 TraceScene(const vec3 pos, const vec3 dir, const vec3 hdrEnvMapUniformDir)
{
	const vec3 exit = find_entry_exit(pos, hdrEnvMapUniformDir)[1];
	const float hdrEnvMapTransmittance = GetTransmittance(pos, exit, 16);
	//const float hdrEnvMapPhase = hg_phase_func(dot(-dir, hdrEnvMapUniformDir));
	const float hdrEnvMapPhase = ApproximateMie(dot(-dir, hdrEnvMapUniformDir));
	const vec3 hdrEnvMapLight = SampleHdrEnvMap(hdrEnvMapUniformDir) * hdrEnvMapTransmittance * hdrEnvMapPhase;

	const vec3 totalLight = TraceDirLight(pos, dir) + TracePointLight(pos, dir) + hdrEnvMapLight;
	return totalLight;
}

vec3 DeltaTrack(const vec3 rayOrigin, const vec3 rayDir, out bool volumeExit)
{
	volumeExit = false;

	const float invMaxDensity = 1.0 / VOLUME_DENSITY_FACTOR;
	//const float invMaxDensity = 1.0;

	const vec3 exit = find_entry_exit(rayOrigin, rayDir)[1];
	const float tMax = distance(exit, rayOrigin);
	float t = 0.0;

	for (uint i = 0; i < 128; i++)
	{
		t -= log(1.0 - RandFloat(1.0)) * invMaxDensity;
		if (t >= tMax)
		{
			volumeExit = true;
			break; 
		}
		const vec3 nextSamplePoint = rayOrigin + (t * rayDir);
		if (getDensity(nextSamplePoint) * invMaxDensity > RandFloat(1.0)) { return nextSamplePoint; }
	}

	return rayOrigin + (RandFloat(tMax) * rayDir);
}
