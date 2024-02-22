# Building the NuGet package for OpenCV

This repository helps generate the OpenCV Nuget package. The process is detailed here and can be replicated by forking this repository to your own account - since Github actions are free and you only need to be able to run actions on the repository.

## Workflow file
The workflow is defined in the `nuget-package-creation.yml` file. You can simply fork the opencv repository and click on the Actions tab on Github and create a new workflow with the same name. Once you have created this new workflow, you can use the provided `nuget-package-creation.yml` file for it's (actions) code.

## Running Github actions
- You need to first create the workflow on your fork of opencv repository
- Under Actions tab, you can see All Workflows, under which you should now be able to see `nuget build (nupkg) creation`
- Click on "Run Workflow", choose a branch, input the value for MSVC Compiler version (e.g. "vc14" or "vc15" or "vc16")
- Click on the green "Run Workflow" button

## Build generates and serves output nupkg
- Once the workflow action is succesful, all build files are generated and the Nuget package (nupkg) is now compiled into one final file, which can be downloaded from the "Action Artifacts"