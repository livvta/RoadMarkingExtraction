# Design

## Point-cloud projection

The extraction pipeline separates ground and non-ground points, rasterizes their intensity, elevation and density, and uses the resulting masks to retrieve candidate point clouds. Each projection pixel retains a relationship with the original points so that the 2D operations can be mapped back to 3D before classification and vectorization.

Point mappings use a compressed row layout: one contiguous 32-bit offset array per selected cloud and one contiguous array of point indices. A pixel's points lie between two adjacent offsets. A counting pass and stable insertion build the mapping without a second cursor array. Storage is four bytes per pixel plus four bytes per indexed point for each mapping, including empty pixels without per-pixel heap allocations. Grids and clouds exceeding the 32-bit index capacity are rejected.

Intensity and elevation projections compute one average per pixel directly from the index range; they do not copy each point's scalar values. Density is the difference between adjacent offsets. The historical normalization and per-pixel point order are preserved, with explicit handling for empty or constant images and expected density counts below one. The unused original-cloud mapping is omitted in the ground/non-ground path. Non-ground mappings are released after projection, and ground mappings after point retrieval.

Dense OpenCV images still scale with the rectangular XY area divided by the square of the resolution. The compact point mapping reduces overhead but does not make arbitrarily large grids memory independent. Ground segmentation, image filtering, classification and vectorization remain separate processing stages.
