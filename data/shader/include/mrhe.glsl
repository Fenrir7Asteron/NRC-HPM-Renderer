float GetMrheFeature(const uint level, const uint entryIndex, const uint featureIndex)
{
	const uint linearIndex = (POS_HASH_TABLE_SIZE * POS_FEATURE_COUNT * level) + (entryIndex * POS_FEATURE_COUNT) + featureIndex;
	const float feature = mrhe[linearIndex];
	return feature;
}

uint HashFunc(const uvec3 pos)
{
	const uvec3 primes = uvec3(1, 19349663, 83492791);
	uint hash = (pos.x * primes.x) + (pos.y * primes.y) + (pos.z * primes.z);
	hash %= POS_HASH_TABLE_SIZE;
	return hash;
}

void EncodePosMrhe(const vec3 pos)
{
	const vec3 normPos = (pos / skySize) + vec3(0.5);

	for (uint level = 0; level < POS_LEVEL_COUNT; level++)
	{
		// Get level resolution
		const uint res = mrheResolutions[level];
		const vec3 resPos = normPos * float(res);

		// Get all 8 neighbours
		const vec3 floorPos = floor(resPos);

		vec3 neighbours[8]; // 2^3
		for (uint x = 0; x < 2; x++)
		{
			for (uint y = 0; y < 2; y++)
			{
				for (uint z = 0; z < 2; z++)
				{
					uint linearIndex = (x * 4) + (y * 2) + z;
					neighbours[linearIndex] = floorPos + vec3(uvec3(x, y, z));
				}
			}
		}

		// Get neighbour indices and store
		uint neighbourIndices[8];
		for (uint neigh = 0; neigh < 8; neigh++)
		{
			const uint index = HashFunc(uvec3(neighbours[neigh]));
			neighbourIndices[neigh] = index;
			
			//const uint linearIndex = (level * 8) + neigh;
			//allNeighbourIndices[linearIndex] = index;
		}

		// Extract neighbour features
		vec2 neighbourFeatures[8];
		for (uint neigh = 0; neigh < 8; neigh++)
		{
			const uint entryIndex = neighbourIndices[neigh];
			neighbourFeatures[neigh] = vec2(GetMrheFeature(level, entryIndex, 0), GetMrheFeature(level, entryIndex, 1));
		}

		// Linearly interpolate neightbour features
		vec3 lerpFactors = pos - neighbours[0];
		//allLerpFactors[level] = lerpFactors;

		vec2 zLerpFeatures[4];
		for (uint i = 0; i < 4; i++)
		{
			zLerpFeatures[i] = 
				(neighbourFeatures[i] * (1.0 - lerpFactors.z)) + 
				(neighbourFeatures[4 + i] * lerpFactors.z);
		}

		vec2 yLerpFeatures[2];
		for (uint i = 0; i < 2; i++)
		{
			yLerpFeatures[i] =
				(zLerpFeatures[i] * (1.0 - lerpFactors.y)) +
				(zLerpFeatures[2 + i] * lerpFactors.y);
		}

		vec2 xLerpFeatures =
			(yLerpFeatures[0] * (1.0 - lerpFactors.x)) +
			(yLerpFeatures[1] * lerpFactors.x);

		// Store in feature array
		neurons[(level * POS_FEATURE_COUNT) + 0] = xLerpFeatures.x;
		neurons[(level * POS_FEATURE_COUNT) + 1] = xLerpFeatures.y;
	}
}

/*void BackpropMrhe()
{
	for (uint level = 0; level < 16; level++)
	{
		const vec3 lerpFactors = allLerpFactors[level];

		uint neighbourIndices[8];
		for (uint neigh = 0; neigh < 8; neigh++)
		{
			const uint linearIndex = (level * 8) + neigh;
			neighbourIndices[neigh] = allNeighbourIndices[linearIndex];
		}

		const vec2 error = vec2(nr0[(level * 2) + 0], nr0[(level * 2) + 0]);

		for (uint x = 0; x < 2; x++)
		{
			for (uint y = 0; y < 2; y++)
			{
				for (uint z = 0; z < 2; z++)
				{
					const uint linearIndex = (x * 4) + (y * 2) + z;
					const uint tableEntryIndex = neighbourIndices[linearIndex];

					const float xFactor = x == 1 ? lerpFactors.x : (1.0 - lerpFactors.x);
					const float yFactor = y == 1 ? lerpFactors.y : (1.0 - lerpFactors.y);
					const float zFactor = z == 1 ? lerpFactors.z : (1.0 - lerpFactors.z);
					const float errorWeight = xFactor * yFactor * zFactor;
					const vec2 delta = -error * errorWeight * ONE_OVER_PIXEL_COUNT;

					atomicAdd(mrDeltaHashTable[(2 * tableEntryIndex) + 0], delta.x);
					atomicAdd(mrDeltaHashTable[(2 * tableEntryIndex) + 1], delta.y);
				}
			}
		}
	}
}*/
