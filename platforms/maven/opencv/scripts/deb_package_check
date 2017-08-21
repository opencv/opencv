#!/bin/bash
##################################################################################################
#
# 		This script checks for the required Debian packages are installed
# 		to build OpenCV.
# Commandline parameters:
# $@                These are the names of the packages to check with 'dpkg'. Multiple values may
#         be specified per package by using pipe as a delimiter, e.g. libpng-dev|libpng12-dev.
#         Multiple values are evaluated left-to-right and the first found prevents checking of
#         the remaining package options.
#
# -o <package_name> Specifying this switch with a package name marks it as optional
#                   i.e. it is not required to be installed.
#
# Returns:
#   0 - All packages installed (success)
#   1 - One or more packages missing (failure)
#
#   Kerry Billingham <contact (at) avionicengineers (d0t) com>
#   20 April 2016
#
##################################################################################################
red=$'\e[1;31m'
green=$'\e[1;32m'
yellow=$'\e[1;33m'
end=$'\e[0m'
check_message="Checking for "
declare -i packageMissing=0
declare -i installed=1

#########################
# Function declarations.
#########################
function check_package() {
     check_message="Checking for package "
     dpkg -s $1 &>/dev/null
     is_installed=$?
     if [ ${is_installed} -ne 0 ]; then
          printf "%-80s%s\n" "$2${check_message}${red}$1" " MISSING.${end}"
          packageMissing=1
     else
          printf "%-80s%s\n" "$2${check_message}${green}$1" " INSTALLED.${end}"
          packageMissing=0
     fi
     return $is_installed
}

# Main part of script.
ORIGINAL_IFS=$IFS

dpkg -? &>/dev/null
if [ $? -ne 0 ]; then
    printf "%-80s%s\n" "${check_message} ${red}'dpkg'" " MISSING.${end}"
    exit 1
else
    printf "%-80s%s\n" "${check_message} ${green}'dpkg'" " INSTALLED.${end}"
fi

while getopts o: option; do
    case $option in
        o)
            IFS="|"
            packageChoices=( ${OPTARG} )
            if [ ${#packageChoices[@]} -gt 1 ]; then
                echo "Optional package. One of ${yellow}${packageChoices[@]}${end} can be installed."
                for choice in ${packageChoices[@]}; do
                    check_package ${choice} "    "
                    if [ $? -eq 0 ]; then
                        break
                    fi
                done
            else
                echo "Optional package ${yellow}${packageChoices}${end}"
                check_package ${OPTARG} "    "
            fi
            IFS=$ORIGINAL_IFS
            ;;
        \?)
            echo "No option found"
            ;;
    esac
done

shift $((OPTIND-1))
packageArray=( $@ )
for package in ${packageArray[@]}; do
    IFS="|"
    packageChoices=( ${package} )
    if [ ${#packageChoices[@]} -gt 1 ]; then
        echo "Multiple options. One of ${yellow}${packageChoices[@]}${end} must be installed."
        for choice in ${packageChoices[@]}; do
            check_package ${choice} "    "
            if [ $? -eq 0 ]; then
                break
            fi
        done
    else
        check_package ${package} ""
    fi
done

exit $packageMissing
