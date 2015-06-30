#
# The MIT License (MIT)
#
# Copyright (c) 2015 Vladislav Vinogradov
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

#
# GIT repository control commands
#

#
# clone_git_repo(<git url> <destination> [BRANCH <branch>])
#
#   Clones git repository from <git url> to <destination> directory.
#
#   Optionally <branch> name can be specified.
#
#   `CTEST_GIT_COMMAND` variable must be defined and must point to git command.
#

function(clone_git_repo GIT_URL GIT_DEST_DIR)
    set(options "")
    set(oneValueArgs "BRANCH")
    set(multiValueArgs "")
    cmake_parse_arguments(GIT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    check_vars_exist(CTEST_GIT_COMMAND)

    if(GIT_BRANCH)
        ctest_ext_info("Clone git repository ${GIT_URL} (branch ${GIT_BRANCH}) to ${GIT_DEST_DIR}")
        execute_process(COMMAND "${CTEST_GIT_COMMAND}" clone -b ${GIT_BRANCH} -- ${GIT_URL} ${GIT_DEST_DIR})
    else()
        ctest_ext_info("Clone git repository ${GIT_URL} to ${GIT_DEST_DIR}")
        execute_process(COMMAND "${CTEST_GIT_COMMAND}" clone ${GIT_URL} ${GIT_DEST_DIR})
    endif()
endfunction()

#
# update_git_repo(<directory> [REMOTE <remote>] [BRANCH <branch>] [UPDATE_COUNT_OUTPUT <output variable>])
#
#   Updates local git repository in <directory> to latest state from remote repository.
#
#   <remote> specifies remote repository name, `origin` by default.
#
#   <branch> specifies remote branch name, `master` by default.
#
#   <output variable> specifies optional output variable to store update count.
#   If it is zero, local repository already was in latest state.
#
#   `CTEST_GIT_COMMAND` variable must be defined and must point to git command.
#

function(update_git_repo GIT_REPO_DIR)
    set(options "")
    set(oneValueArgs "REMOTE" "BRANCH" "UPDATE_COUNT_OUTPUT")
    set(multiValueArgs "")
    cmake_parse_arguments(GIT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # TODO : use FETCH_HEAD
    set_ifndef(GIT_REMOTE "origin")
    set_ifndef(GIT_BRANCH "master")

    check_vars_exist(CTEST_GIT_COMMAND GIT_REPO_DIR)

    ctest_ext_info("Fetch git remote repository ${GIT_REMOTE} in ${GIT_REPO_DIR}")
    execute_process(COMMAND "${CTEST_GIT_COMMAND}" fetch
        WORKING_DIRECTORY "${GIT_REPO_DIR}")

    if(GIT_UPDATE_COUNT_OUTPUT)
        ctest_ext_info("Compare git local repository with ${GIT_REMOTE}/${GIT_BRANCH} state in ${GIT_REPO_DIR}")
        execute_process(COMMAND "${CTEST_GIT_COMMAND}" diff HEAD "${GIT_REMOTE}/${GIT_BRANCH}"
            WORKING_DIRECTORY "${GIT_REPO_DIR}"
            OUTPUT_VARIABLE diff_output)

        string(LENGTH "${diff_output}" update_count)
        set(${GIT_UPDATE_COUNT_OUTPUT} "${update_count}" PARENT_SCOPE)
    endif()

    ctest_ext_info("Reset git local repository to ${GIT_REMOTE}/${GIT_BRANCH} state in ${GIT_REPO_DIR}")
    execute_process(COMMAND "${CTEST_GIT_COMMAND}" reset --hard "${GIT_REMOTE}/${GIT_BRANCH}"
        WORKING_DIRECTORY "${GIT_REPO_DIR}")
endfunction()

#
# get_git_repo_info(<repository> <branch output variable> <revision output variable>)
#
#   Gets information about local git repository (branch name and revision).
#

function(get_git_repo_info GIT_REPO_DIR BRANCH_OUT_VAR REVISION_OUT_VAR)
    check_vars_exist(CTEST_GIT_COMMAND)

    execute_process(COMMAND "${CTEST_GIT_COMMAND}" rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY "${GIT_REPO_DIR}"
        OUTPUT_VARIABLE branch
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    execute_process(COMMAND "${CTEST_GIT_COMMAND}" rev-parse HEAD
        WORKING_DIRECTORY "${GIT_REPO_DIR}"
        OUTPUT_VARIABLE revision
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(${BRANCH_OUT_VAR} ${branch} PARENT_SCOPE)
    set(${REVISION_OUT_VAR} ${revision} PARENT_SCOPE)
endfunction()
