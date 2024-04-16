FROM mambaorg/micromamba:latest

# Replace mamba user with current one (https://micromamba-docker.readthedocs.io/en/latest/advanced_usage.html#changing-the-user-id-or-name)
ARG NEW_MAMBA_USER
ARG NEW_MAMBA_USER_ID
ARG NEW_MAMBA_USER_GID
USER root
RUN if grep -q '^ID=alpine$' /etc/os-release; then \
    # alpine does not have usermod/groupmod
    apk add --no-cache --virtual temp-packages shadow; \
    fi && \
    usermod "--login=${NEW_MAMBA_USER}" "--home=/home/${NEW_MAMBA_USER}" \
        --move-home "-u ${NEW_MAMBA_USER_ID}" "${MAMBA_USER}" && \
    groupmod "--new-name=${NEW_MAMBA_USER}" \
        "-g ${NEW_MAMBA_USER_GID}" "${MAMBA_USER}" && \
    if grep -q '^ID=alpine$' /etc/os-release; then \
    # remove the packages that were only needed for usermod/groupmod
    apk del temp-packages; \
    fi && \
    # Update the expected value of MAMBA_USER for the
    # _entrypoint.sh consistency check.
    echo "${NEW_MAMBA_USER}" > "/etc/arg_mamba_user" && \
    :
ENV MAMBA_USER=$NEW_MAMBA_USER
USER $MAMBA_USER

# Install system dependencies
USER root
RUN apt update && apt install make cmake g++ git -y
RUN apt install uuid-dev -y
USER $MAMBA_USER

WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER env.yml /tmp/env.yaml

## Create environment from environment.yml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1

USER root
# Clone and install qparselib and dependencies
ARG QPARSE_PATH="/app/qparselib"
ARG PARALLEL_MAKE_JOBS=4
RUN git clone https://gitlab.inria.fr/qparse/qparselib.git "$QPARSE_PATH" &&\
    cd qparselib && git checkout gmd-score && git pull &&\
    # Remove automatically pulled submodule to install it manually (avoid some conflicts)
    rm -r extern/midifile extern/libmei extern/libmusicxml  &&\
    # Manually clone and build submodules
    git clone https://github.com/craigsapp/midifile.git --branch master extern/midifile &&\
    git clone https://github.com/DDMAL/libmei.git --branch master extern/libmei &&\
    git clone https://github.com/grame-cncm/libmusicxml.git --branch dev extern/libmusicxml &&\
    # Build and install midifile
    cd extern/midifile && make -j$PARALLEL_MAKE_JOBS library && cd "$QPARSE_PATH" &&\
    # Build and install libmei (need to remove -Werror flag from CMakelists.txt)
    cd extern/libmei && sed -i 's/^ *add_definitions( *-Werror *)/#&/' CMakeLists.txt &&\
    mkdir build && cd build && cmake .. && make -j$PARALLEL_MAKE_JOBS && make -j$PARALLEL_MAKE_JOBS install && cd "$QPARSE_PATH" &&\
    # Build and install libmusicxml
    cd extern/libmusicxml/build && cmake . && make -j$PARALLEL_MAKE_JOBS

# Build and install drumparse and pyqparse from qparselib
RUN cd "$QPARSE_PATH" && if [ ! -d build ]; then mkdir build; fi && cd build &&\
    cmake .. -DPython3_FIND_STRATEGY=LOCATION &&\
    make -j$PARALLEL_MAKE_JOBS drumparse &&\
    make -j$PARALLEL_MAKE_JOBS pyqparse

# Allow mambauser to access GPUs
ARG VIDEO_GROUP_ID
RUN groupmod -g $VIDEO_GROUP_ID video && usermod -a -G video $MAMBA_USER
USER $MAMBA_USER

# Make qparselib available everywhere
ENV PATH="${PATH}:${QPARSE_PATH}/build"
ENV PYTHONPATH="${PYTHONPATH}:${QPARSE_PATH}/build"