#!/bin/bash
# This script verifies that all shell snippets in the
# Linux installation tutorial work (in Ubuntu 20 container)
set -e
set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

containers=(ubuntu:20.04 debian:10)
scripts=$(cd "${SCRIPT_DIR}" && ls -1 linux_*install*.sh)

docker pull debian:10

for cnt in $containers ; do
    docker pull ${cnt}
    for f in $scripts ; do
        echo "Checking ${f} @ ${cnt}..."
        docker run -it \
            -e DEBIAN_FRONTEND=noninteractive \
            --volume "${SCRIPT_DIR}":/install:ro \
            ${cnt} \
            /bin/bash -ex /install/${f} --check
    done
done
