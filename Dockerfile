FROM mambaorg/micromamba:latest

# Install system dependencies
USER root
RUN apt update && apt install make cmake g++ git -y
RUN apt install uuid-dev -y
USER 1001

WORKDIR /app

#COPY . /app
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yml /tmp/env.yaml

## Create environment from environment.yml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1

USER root

# Clone and install qparselib and dependencies
ARG QPARSE_PATH="/app/qparselib"
RUN make_jobs=8 &&\
    git clone https://gitlab.inria.fr/qparse/qparselib.git "$QPARSE_PATH" &&\
    cd qparselib && git checkout gmd-score && git pull &&\
    # Remove automatically pulled submodule to install it manually (avoid some conflicts)
    rm -r extern/midifile extern/libmei extern/libmusicxml  &&\
    # Manually clone and build submodules
    git clone https://github.com/craigsapp/midifile.git --branch master extern/midifile &&\
    git clone https://github.com/DDMAL/libmei.git --branch master extern/libmei &&\
    git clone https://github.com/grame-cncm/libmusicxml.git --branch dev extern/libmusicxml &&\
    # Build and install midifile
    cd extern/midifile && make -j$make_jobs library && cd "$QPARSE_PATH" &&\
    # Build and install libmei (need to remove -Werror flag from CMakelists.txt)
    cd extern/libmei && sed -i 's/^ *add_definitions( *-Werror *)/#&/' CMakeLists.txt &&\
    mkdir build && cd build && cmake .. && make -j$make_jobs && make -j$make_jobs install && cd "$QPARSE_PATH" &&\
    # Build and install libmusicxml
    cd extern/libmusicxml/build && cmake . && make -j$make_jobs && cd "$QPARSE_PATH" &&\
    # Build and install drumparse and pyqparse from qparselib
    mkdir build && cd build && cmake .. -DPython3_FIND_STRATEGY=LOCATION &&\
    make -j$make_jobs drumparse && make -j$make_jobs pyqparse

# Make qparselib available everywhere
ENV PATH="${PATH}:${QPARSE_PATH}/build"
ENV PYTHONPATH="${PYTHONPATH}:${QPARSE_PATH}/build"

# Install ipython
RUN pip install ipython
