#!bin/bash

SCRIPT_DIR="$(readlink -f "$(dirname "${BASH_SOURCE[0]}")")"

export UPMEM_HOME=${PWD}/upmem-2023.2.0-Linux-x86_64
echo "Setting UPMEM_HOME to ${UPMEM_HOME} and updating PATH/LD_LIBRARY_PATH/PYTHONPATH"
export LD_LIBRARY_PATH="${UPMEM_HOME}/lib${LD_LIBRARY_PATH+:$LD_LIBRARY_PATH}:${PWD}/lib"
export PATH="${UPMEM_HOME}/bin:${PATH}"

PYTHON_RELATIVE_PATH=$(/usr/bin/env python3 -c "import distutils.sysconfig; print(distutils.sysconfig.get_python_lib(True, False, ''))")

if [[ -z "${PYTHON_RELATIVE_PATH}" ]];
then
    echo "Could not set PYTHONPATH!"
else
    export PYTHONPATH="${UPMEM_HOME}/${PYTHON_RELATIVE_PATH}${PYTHONPATH+:$PYTHONPATH}"
fi

_DEFAULT_BACKEND=$1

if [[ -n "${_DEFAULT_BACKEND}" ]];
then
    echo "Setting default backend to ${_DEFAULT_BACKEND} in UPMEM_PROFILE_BASE"
    export UPMEM_PROFILE_BASE=backend=${_DEFAULT_BACKEND}
fi
