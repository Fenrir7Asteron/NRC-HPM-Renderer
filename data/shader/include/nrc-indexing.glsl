uint GetLinearPixelIndex(const uint x, const uint y, const uint renderWidth, const uint renderHeight,
	const uint batchSizeHorizontal, const uint batchSizeVertical)
{
	const uint inferBatchCountVertical = uint(ceil(float(renderHeight) / batchSizeVertical));
	const uint inferBatchCountHorizontal = uint(ceil(float(renderWidth) / batchSizeHorizontal));
	const uint inferBatchSizeTotal = batchSizeHorizontal * batchSizeVertical;
	const uint batchIndexVertical = y / batchSizeVertical;
	const uint batchIndexHorizontal = x / batchSizeHorizontal;

	const uint lastBatchSizeVertical = renderHeight - ((inferBatchCountVertical - 1) * batchSizeVertical);
	const uint lastBatchSizeHorizontal = renderWidth - ((inferBatchCountHorizontal - 1) * batchSizeHorizontal);

	const uint isLastBatchVertical = int(step(inferBatchCountVertical - 1, batchIndexVertical));
	const uint isLastBatchHorizontal = int(step(inferBatchCountHorizontal - 1, batchIndexHorizontal));
	const uint isNotLastBatchVertical = 1 - isLastBatchVertical;
	const uint isNotLastBatchHorizontal = 1 - isLastBatchHorizontal;
	const uint batchOffsetY = y - batchIndexVertical * batchSizeVertical;
	const uint batchOffsetX = x - batchIndexHorizontal * batchSizeHorizontal;

	//	const uint linearPixelIndex = (y * renderWidth) + x;

	return batchIndexVertical * inferBatchSizeTotal * (inferBatchCountHorizontal - 1)
		+ batchIndexVertical * lastBatchSizeHorizontal * batchSizeVertical
		+ batchIndexHorizontal * inferBatchSizeTotal * isNotLastBatchVertical
		+ batchIndexHorizontal * lastBatchSizeVertical * batchSizeHorizontal * isLastBatchVertical
		+ batchOffsetY * batchSizeHorizontal * isNotLastBatchHorizontal
		+ batchOffsetY * lastBatchSizeHorizontal * isLastBatchHorizontal
		+ batchOffsetX
		;
}