<#
Copyright (c) Microsoft Open Technologies, Inc.
All rights reserved.

(3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification, are permitted provided that
the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
#>

[CmdletBinding()]
Param(
    [parameter(Mandatory=$False)]
    [switch]
    $HELP,

    [parameter(Mandatory=$False)]
    [switch]
    $BUILD,

    [parameter(Mandatory=$False)]
    [Array]
    [ValidateNotNull()]
    $PLATFORMS_IN = "WP",

    [parameter(Mandatory=$False)]
    [Array]
    [ValidateNotNull()]
    $VERSIONS_IN = "8.1",

    [parameter(Mandatory=$False)]
    [Array]
    [ValidateNotNull()]
    $ARCHITECTURES_IN = "x86",

    [parameter(Mandatory=$False)]
    [String]
    $TESTS = "None",

    [parameter(Mandatory=$False)]
    [String]
    [ValidateNotNull()]
    [ValidateSet("Visual Studio 15 2017","Visual Studio 14 2015","Visual Studio 12 2013","Visual Studio 11 2012")]
    $GENERATOR = "Visual Studio 15 2017",

    [parameter(Mandatory=$False)]
    [String]
    $INSTALL
)


Function L() {
    Param(
        [parameter(Mandatory=$true)]
        [String]
        [ValidateNotNull()]
        $str
    )

    Write-Host "INFO> $str"
}

Function D() {
    Param(
        [parameter(Mandatory=$true)]
        [String]
        [ValidateNotNull()]
        $str
    )

    # Use this trigger to toggle debug output
    [bool]$debug = $true

    if ($debug) {
        Write-Host "DEBUG> $str"
    }
}

function Get-Batchfile ($file) {
    $cmd = "`"$file`" & set"
    cmd /c $cmd | Foreach-Object {
        $p, $v = $_.split('=')
        Set-Item -path env:$p -value $v
    }
}

# Enables access to Visual Studio variables via "vsvars32.bat"
function Set-VS12()
{
    Try {
        $vs12comntools = (Get-ChildItem env:VS120COMNTOOLS).Value
        $batchFile = [System.IO.Path]::Combine($vs12comntools, "vsvars32.bat")
        Get-Batchfile $BatchFile
        [System.Console]::Title = "Visual Studio 2010 Windows PowerShell"
     } Catch {
        $ErrorMessage = $_.Exception.Message
        L "Error: $ErrorMessage"
        return $false
     }
     return $true
}

# Executes msbuild to build or install projects
# Throws Exception on error
function Call-MSBuild($path, $config)
{
    $command = "msbuild $path /p:Configuration='$config' /m"
    L "Executing: $($command)"
    msbuild $path /p:Configuration="$config" /m

    if(-Not $?) {
        Throw "Failure executing command: $($command)"
    }

    return $true
}

function RunAccuracyTests($path) {
    md "$path\bin\Release\accuracy"
    python "$PSScriptRoot\..\..\modules\ts\misc\run.py" -w "$path\bin\Release\accuracy" -a "$path\bin\Release"
}

function RunPerfTests($path) {
    md "$path\bin\Release\perf"
    python "$PSScriptRoot\..\..\modules\ts\misc\run.py" -w "$path\bin\Release\perf" "$path\bin\Release"
}

Function Execute() {
    If ($HELP.IsPresent) {
        ShowHelp
    }

    # Validating arguments.
    # This type of validation (rather than using ValidateSet()) is required to make .bat wrapper work

    D "Input Platforms: $PLATFORMS_IN"
    $platforms = New-Object System.Collections.ArrayList
    $PLATFORMS_IN.Split("," ,[System.StringSplitOptions]::RemoveEmptyEntries) | ForEach {
        $_ = $_.Trim()
        if ("WP","WS" -Contains $_) {
            [void]$platforms.Add($_)
            D "$_ is valid"
        } else {
            Throw "$($_) is not valid! Please use WP, WS"
        }
    }
    D "Processed Platforms: $platforms"

    D "Input Versions: $VERSIONS_IN"
    $versions = New-Object System.Collections.ArrayList
    $VERSIONS_IN.Split("," ,[System.StringSplitOptions]::RemoveEmptyEntries) | ForEach {
        $_ = $_.Trim()
        if ("8.0","8.1","10.0" -Contains $_) {
            [void]$versions.Add($_)
            D "$_ is valid"
        } else {
            Throw "$($_) is not valid! Please use 8.0, 8.1, 10.0"
        }
    }
    D "Processed Versions: $versions"

    D "Input Architectures: $ARCHITECTURES_IN"
    $architectures = New-Object System.Collections.ArrayList
    $ARCHITECTURES_IN.Split("," ,[System.StringSplitOptions]::RemoveEmptyEntries) | ForEach {
        $_ = $_.Trim()
        if ("x86","x64","ARM" -Contains $_) {
            $architectures.Add($_) > $null
            D "$_ is valid"
        } else {
            Throw "$($_) is not valid! Please use x86, x64, ARM"
        }
    }

    D "Processed Architectures: $architectures"

    # Assuming we are in '<ocv-sources>/platforms/winrt' we should move up to sources root directory
    Push-Location ../../

    $SRC = Get-Location

    $def_architectures = @{
        "x86" = "";
        "x64" = " Win64"
        "arm" = " ARM"
    }

    # Setting up Visual Studio variables to enable build
    $shouldBuid = $false
    If ($BUILD.IsPresent) {
        $shouldBuild = Set-VS12
    }

    foreach($plat in $platforms) {
        # Set proper platform name.
        $platName = ""
        Switch ($plat) {
            "WP" { $platName = "WindowsPhone" }
            "WS" { $platName = "WindowsStore" }
        }

        foreach($vers in $versions) {

            foreach($arch in $architectures) {

                # Set proper architecture. For MSVS this is done by selecting proper generator
                $genName = $GENERATOR
                Switch ($arch) {
                    "ARM" { $genName = $GENERATOR + $def_architectures['arm'] }
                    "x64" { $genName = $GENERATOR + $def_architectures['x64'] }
                }

                # Constructing path to the install binaries
                # Creating these binaries will be done by building CMake-generated INSTALL project from Visual Studio
                $installPath = "$SRC\bin\install\$plat\$vers\$arch"
                if ($INSTALL) {
                    # Do not add architrecture to the path since it will be added by OCV CMake logic
                    $installPath = "$SRC\$INSTALL\$plat\$vers"
                }

                $path = "$SRC\bin\$plat\$vers\$arch"

                L "-----------------------------------------------"
                L "Target:"
                L "    Directory: $path"
                L "    Platform: $platName"
                L "    Version: $vers"
                L "    Architecture: $arch"
                L "    Generator: $genName"
                L "    Install Directory: $installPath"

                # Delete target directory if exists to ensure that CMake cache is cleared out.
                If (Test-Path $path) {
                    Remove-Item -Recurse -Force $path
                }

                # Validate if required directory exists, create if it doesn't
                New-Item -ItemType Directory -Force -Path $path

                # Change location to the respective subdirectory
                Push-Location -Path $path

                L "Generating project:"
                L "cmake -G $genName -DCMAKE_SYSTEM_NAME:String=$platName -DCMAKE_SYSTEM_VERSION:String=$vers -DCMAKE_VS_EFFECTIVE_PLATFORMS:String=$arch -DCMAKE_INSTALL_PREFIX:PATH=$installPath $SRC"
                cmake -G $genName -DCMAKE_SYSTEM_NAME:String=$platName -DCMAKE_SYSTEM_VERSION:String=$vers -DCMAKE_VS_EFFECTIVE_PLATFORMS:String=$arch -DCMAKE_INSTALL_PREFIX:PATH=$installPath $SRC
                L "-----------------------------------------------"

                # REFERENCE:
                # Executed from '$SRC/bin' folder.
                # Targeting x86 WindowsPhone 8.1.
                # cmake -G "Visual Studio 12 2013" -DCMAKE_SYSTEM_NAME:String=WindowsPhone -DCMAKE_SYSTEM_VERSION:String=8.1 ..


                # Building and installing project
                Try {
                    If ($shouldBuild) {
                        L "Building and installing project:"

                        Call-MSBuild "OpenCV.sln" "Debug"
                        Call-MSBuild "INSTALL.vcxproj" "Debug"

                        Call-MSBuild "OpenCV.sln" "Release"
                        Call-MSBuild "INSTALL.vcxproj" "Release"

                        Try {
                            # Running tests for release versions:
                            If ($TESTS -eq "ALL") {
                                RunAccuracyTests "$path"
                                RunPerfTests "$path"
                            } else {
                                If($TESTS -eq "ACC") {
                                    RunAccuracyTests "$path"
                                }
                                If($TESTS -eq "PERF") {
                                    RunPerfTests "$path"
                                }
                            }
                        } Catch {
                            $ErrorMessage = $_.Exception.Message
                            L "Error: $ErrorMessage"
                            exit
                        }
                    }
                } Catch {
                    $ErrorMessage = $_.Exception.Message
                    L "Error: $ErrorMessage"

                    # Exiting at this point will leave command line pointing at the erroneous configuration directory
                    exit
                }

                # Return back to Sources folder
                Pop-Location
            }
        }
    }

    # Return back to Script folder
    Pop-Location
}

Function ShowHelp() {
    Write-Host "Configures OpenCV and generates projects for specified verion of Visual Studio/platforms/architectures."
    Write-Host "Must be executed from the sources folder containing main CMakeLists configuration."
    Write-Host "Parameter keys can be shortened down to a single symbol (e.g. '-a') and are not case sensitive."
    Write-Host "Proper parameter sequencing is required when omitting keys."
    Write-Host "Generates the following folder structure, depending on the supplied parameters: "
    Write-Host "     bin/ "
    Write-Host "      | "
    Write-Host "      |-WP "
    Write-Host "      |  ... "
    Write-Host "      |-WinRT "
    Write-Host "      |  |-8.0 "
    Write-Host "      |  |-8.1 "
    Write-Host "      |  |  |-x86 "
    Write-Host "      |  |  |-x64 "
    Write-Host "      |  |  |-ARM "
    Write-Host " "
    Write-Host " USAGE: "
    Write-Host "   Calling:"
    Write-Host "     PS> setup_winrt.ps1 [params]"
    Write-Host "     cmd> setup_winrt.bat [params]"
    Write-Host "     cmd> PowerShell.exe -ExecutionPolicy Unrestricted -File setup_winrt.ps1 [params]"
    Write-Host "   Parameters:"
    Write-Host "     setup_winrt [options] [platform] [version] [architecture] [tests] [generator] [install-path]"
    Write-Host "     setup_winrt -b 'WP' 'x86,ARM' "
    Write-Host "     setup_winrt -b 'WP' 'x86,ARM' ALL"
    Write-Host "     setup_winrt -b 'WP' 'x86,ARM' -test PERF "
    Write-Host "     setup_winrt -architecture x86 -platform WP "
    Write-Host "     setup_winrt -arc x86 -plat 'WP,WS' "
    Write-Host "     setup_winrt -a x86 -g 'Visual Studio 15 2017' -pl WP "
    Write-Host " WHERE: "
    Write-Host "     options -  Options to call "
    Write-Host "                 -h: diplays command line help "
    Write-Host "                 -b: builds BUILD_ALL and INSTALL projects for each generated configuration in both Debug and Release modes."
    Write-Host "     platform -  Array of target platforms. "
    Write-Host "                 Default: WP "
    Write-Host "                 Example: 'WS,WP' "
    Write-Host "                 Options: WP, WS ('WindowsPhone', 'WindowsStore'). "
    Write-Host "                 Note that you'll need to use quotes to specify more than one platform. "
    Write-Host "     version - Array of platform versions. "
    Write-Host "                 Default: 8.1 "
    Write-Host "                 Example: '8.0,8.1' "
    Write-Host "                 Options: 8.0, 8.1, 10.0. Available options may be limited depending on your local setup (e.g. SDK availability). "
    Write-Host "                 Note that you'll need to use quotes to specify more than one version. "
    Write-Host "     architecture - Array of target architectures to build for. "
    Write-Host "                 Default: x86 "
    Write-Host "                 Example: 'ARM,x64' "
    Write-Host "                 Options: x86, ARM, x64. Available options may be limited depending on your local setup. "
    Write-Host "                 Note that you'll need to use quotes to specify more than one architecture. "
    Write-Host "     tests - Test sets to run. Requires -b option otherwise ignored. "
    Write-Host "                 Default: None. "
    Write-Host "                 Example: 'ALL' "
    Write-Host "                 Options: ACC, PERF, ALL. "
    Write-Host "     generator - Visual Studio instance used to generate the projects. "
    Write-Host "                 Default: Visual Studio 12 2013 "
    Write-Host "                 Example: 'Visual Studio 11 2012' "
    Write-Host "                 Use 'cmake --help' to find all available option on your machine. "
    Write-Host "     install-path - Path to install binaries (relative to the sources directory). "
    Write-Host "                 Default: <src-dir>\bin\install\<platform>\<version>\<architecture> "
    Write-Host "                 Example: '../install' "

    Exit
}

Execute