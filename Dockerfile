FROM nvidia/cuda:12.1-devel-ubuntu22.04

LABEL maintainer="ai4s-isotope"
LABEL description="AI4S Cross-scale Prediction Platform for Anomalous Isotope Effects"
LABEL version="1.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1

# ---- System Dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc g++ gfortran \
    cmake \
    git \
    wget \
    ca-certificates \
    libopenblas-dev \
    libfftw3-dev \
    liblapack-dev \
    libscalapack-mpi-dev \
    libboost-all-dev \
    libgomp1 \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Use Python 3.10 explicitly
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# ---- Python Dependencies ----
RUN python -m pip install --no-cache-dir \
    numpy==1.24.3 \
    scipy==1.10.1 \
    matplotlib==3.7.2 \
    h5py==3.9.0 \
    ase==3.22.1 \
    pyyaml==6.0 \
    tqdm==4.65.0 \
    pytest==7.4.0 \
    pytest-cov==4.1.0

# ---- ABACUS (from source) ----
# Note: production uses pre-built binary or conda package.
# This build is for environment validation.
RUN git clone --depth 1 --branch v3.7.0 \
    https://github.com/deepmodeling/abacus-develop.git /opt/abacus && \
    cd /opt/abacus && mkdir build && cd build && \
    cmake .. \
    -DCMAKE_INSTALL_PREFIX=/opt/abacus/install \
    -DENABLE_DEEPKS=OFF \
    -DENABLE_LIBXC=ON \
    -DUSE_OPENMP=ON \
    && make -j4 && make install
ENV PATH="/opt/abacus/install/bin:${PATH}"

# ---- DeePMD-kit ----
RUN python -m pip install --no-cache-dir deepmd-kit==2.2.7

# ---- DP-GEN ----
RUN python -m pip install --no-cache-dir dpgen==0.5.2

# ---- i-PI (Path Integral engine) ----
RUN git clone --depth 1 --branch v2.6.1 \
    https://github.com/i-pi/i-pi.git /opt/i-pi && \
    cd /opt/i-pi && python -m pip install --no-cache-dir .
ENV PATH="/opt/i-pi/bin:${PATH}"

# ---- LAMMPS with DeePMD-kit plugin ----
RUN git clone --depth 1 --branch stable_2Aug2023 \
    https://github.com/lammps/lammps.git /opt/lammps && \
    cd /opt/lammps && mkdir build && cd build && \
    cmake ../cmake \
    -D PKG_USER-DEEPMD=ON \
    -D CMAKE_INSTALL_PREFIX=/opt/lammps/install \
    && make -j4 && make install
ENV PATH="/opt/lammps/install/bin:${PATH}"

# ---- Workspace Setup ----
WORKDIR /workspace/ai4s-isotope
COPY . .

# ---- Environment Verification ----
RUN python -c "\
import numpy; import scipy; import deepmd; \
print('NumPy:', numpy.__version__); \
print('SciPy:', scipy.__version__); \
print('DeePMD-kit:', deepmd.__version__); \
print('Environment OK')"

# i-PI socket port
EXPOSE 31415

CMD ["/bin/bash"]
