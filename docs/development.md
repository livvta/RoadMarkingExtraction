# Development

## Ubuntu 22.04 dependency compatibility

The build uses C++14 as required by PCL 1.12 and searches the optional project-local `.deps` prefix for libLAS. Source compatibility fixes use PCL's point-cloud pointer type, current OpenCV constants, explicit standard and Boost headers, and standard thread sleeping. These changes retain the existing extraction and LAS I/O algorithms.

libLAS remains required for LAS input and output. Replacing it with a maintained LAS library would require a separate I/O migration and validation of coordinates, intensity, color and output files; PCL and OpenCV do not provide a drop-in replacement. The local libLAS build described in the README does not include LAZ or coordinate reference system transformation support.

## Zero-intensity candidates

Candidate clouds can contain only zero intensities. Such candidates provide no intensity contrast for Otsu thresholding, so they are discarded before histogram normalization rather than dividing by zero. Empty candidates and thresholds with an empty histogram class are also skipped.

## Memory for fine projection grids

Projection previously allocated three arrays of per-pixel vectors, then copied point intensities and elevations into further per-pixel vectors. Empty pixels incurred allocation overhead, making large, fine grids expensive. The point-to-pixel mapping now uses contiguous offsets and point indices, and projection statistics are accumulated directly from these ranges. Stable point ordering and the existing intensity/elevation normalization are retained. The main pipeline omits the unused original-cloud index and releases other indices once their last consumer has finished. Seed filling marks pixels when queued to avoid duplicate stack entries.

The density normalization cap is clamped to at least one point. Previously, expected counts below one were truncated to zero and made empty pixels white, which invalidated the road mask at fine resolutions. This is the only intended change to normal projection values. Projection/back-projection tests cover reference values, stable point retrieval, empty cells and the sub-one density case. On the campus map, all image, LAS and DXF files at the existing 10 cm and 15 cm settings remained identical; the complete 5 cm run used about 1.12 GiB peak resident memory.
