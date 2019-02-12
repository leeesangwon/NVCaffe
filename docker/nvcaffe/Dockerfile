FROM nvcr.io/nvidia/caffe:19.01-py2

# Bring in changes from outside container to /tmp
# (assume caffe-refinedet-modifications.patch is in same directory as Dockerfile)
COPY nvcaffe-refinedet-modifications.patch /tmp

# Change working directory to NVCaffe source path
WORKDIR /opt/caffe

# Apply modifications
RUN patch -p1 < /tmp/nvcaffe-refinedet-modifications.patch

# Rebuild NVCaffe
RUN mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr/local -DUSE_MPI=ON -DUSE_NCCL=ON -DUSE_CUDNN=ON \
          -DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN="52 60 61 70 75" -DCUDA_ARCH_PTX="75" \
          -Dpython_version=$PYVER -DUSE_TRT=ON .. && \
    make -j"$(nproc)" install && \
    make clean && \
    cd .. && rm -rf build && \
    ldconfig

# Reset default working directory
WORKDIR /workspace
