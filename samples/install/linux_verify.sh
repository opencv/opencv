#!/bin/bash
# This script verifies that all shell snippets in the
# Linux installation tutorial work (in Ubuntu 18 container)
set -e
set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

docker pull ubuntu:18.04

for f in $(cd "${SCRIPT_DIR}" && ls -1 linux_*install*.sh) ; do
    echo "Checking $f..."
    docker run -it \
        --volume "${SCRIPT_DIR}":/install:ro \
        ubuntu:18.04 \
        /bin/bash -ex /install/$f --check

done
