#!/usr/bin/env bash

set -e

MTHEME_VER=2fa6084b9d34fec9d2d5470eb9a17d0bf712b6c8
MTHEME_DIR=mtheme.sty

function make_sty {
    if [ -d "$MTHEME_DIR" ]; then rm -rf "$MTHEME_DIR"; fi
    mkdir "$MTHEME_DIR"

    # Download template from Github
    tmp_dir=$(mktemp -d)
    wget -P "$tmp_dir" -c https://github.com/matze/mtheme/archive/${MTHEME_VER}.tar.gz
    pushd "$tmp_dir"
    tar -xzvf "$MTHEME_VER.tar.gz"
    popd
    make -C "$tmp_dir"/mtheme-"$MTHEME_VER"
    cp   -v "$tmp_dir"/mtheme-"$MTHEME_VER"/*.sty "$MTHEME_DIR"
    rm -r "$tmp_dir"
    # Put our own .gitignore to ignore this directory completely
    echo "*" > "$MTHEME_DIR/.gitignore"
}

make_sty
