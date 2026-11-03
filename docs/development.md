# Development

## Ubuntu 22.04 dependency compatibility

The build uses C++14 as required by PCL 1.12 and searches the optional project-local `.deps` prefix for libLAS. Source compatibility fixes use PCL's point-cloud pointer type, current OpenCV constants, explicit standard and Boost headers, and standard thread sleeping. These changes retain the existing extraction and LAS I/O algorithms.

libLAS remains required for LAS input and output. Replacing it with a maintained LAS library would require a separate I/O migration and validation of coordinates, intensity, color and output files; PCL and OpenCV do not provide a drop-in replacement. The local libLAS build described in the README does not include LAZ or coordinate reference system transformation support.
