#!/bin/bash

# 1. FIND NUMPY HEADERS (The Fix for your error)
# We use the python inside the environment to ask for the include path
export CPATH=$CPATH:$(python -c "import numpy; print(numpy.get_include())")

# 2. ZED SDK & System Libs (Your previous config)
# Prepend /usr/lib/x86_64-linux-gnu to ensure system libraries (like CUDA) are found
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH:/usr/local/zed/lib
export CPATH=$CPATH:/usr/local/zed/include

if [ -f "install/setup.bash" ]; then
    source install/setup.bash
fi

export LIBGL_ALWAYS_SOFTWARE=1
