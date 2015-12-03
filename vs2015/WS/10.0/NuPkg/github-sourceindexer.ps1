#requires -version 2

<#
.DESCRIPTION
  Github source indexer will index PDB files with HTTP source file references to a Github repository. 
   - http://hamishgraham.net/post/GitHub-Source-Symbol-Indexer.aspx
   - Adapted from http://sourcepack.codeplex.com 

.PARAMETER symbolsFolder
  The path of the directory to recursively search for pdb files to index. 
.PARAMETER userId
  The github user ID.
.PARAMETER repository
  The github repository name containing the matching source files.
.PARAMETER branch
  The github branch name, the version of the files in the branch must match the source versions the pdb file was created with.
.PARAMETER sourcesRoot
  The root of the source folder - i.e the beginning of the original file paths to be stripped out (obtained using "srctool -r Library.pdb"). 
  Will default to the longest common file path if not provided. The remainder will be appended to the appropriate Github url for source retrieval. 
.PARAMETER dbgToolsPath
  Path to the Debugging Tools for Windows (the srcsrv subfolder) - if not specifed the script tries to find it. 
  If you don't have the Debugging Tools for Windows in PATH variable you need to provide this argument.
.PARAMETER gitHubUrl
  Path to the Github server (defaults to "http://github.com") - override for in-house enterprise github installations.
.PARAMETER ignore
  Ignore a source path that contains any of the strings in this array, e.g. -ignore somedir, "some other dir"
.PARAMETER ignoreUnknown
  By default this script terminates when it encounters source from a path other than the source root.
  Pass this switch to instead ignore all paths other than the source root.
.PARAMETER serverIsRaw
  If the server serves raw the /raw directory name should not be concatenated to the source urls.
  Pass this switch to omit the /raw directory, e.g. -gitHubUrl https://raw.github.com -serverIsRaw
.PARAMETER verifyLocalRepo
  This switch verifies the local repository from the detected or passed in 'sourcesRoot' by using 
  git to get the filenames from the tree associated with 'branch' (which is either a branch or 
  commit). Any filename from the PDB that is found in the tree list, and that is not excluded by 
  other options, will have its source server information added in the same case that it is seen in 
  the tree list. Other filenames from the PDB that are not found in the tree list will be ignored. 
  This is an important switch and is recommended because PDBs don't often store case sensitivity 
  for files while github servers expect case sensitivity for the files that are requested. Use of 
  this switch implies switch ignoreUnknown.

.EXAMPLE 
  .\github-sourceindexer.ps1 -symbolsFolder "C:\git\DirectoryContainingPdbFilesToIndex" -userId "GithubUsername" -repository "GithubRepositoryName" -branch "master" -sourcesRoot "c:\git\OriginalCompiledProjectPath" -verbose
  
  Description
  -----------
  This command will index all pdb files located in the C:\git\DirectoryContainingPdbFilesToIndex directory and subdirectories, 
  adding to the source stream a reference to the master branch of the GithubRepositoryName repository for the github user GithubUsername.
  
  For example source indexes added like http://github.com/GithubUsername/GithubRepositoryName/raw/master/ExampleLibrary/LibraryClass.cs 
  where /ExampleLibrary/LibraryClass.cs is the remainder after removing c:\git\OriginalCompiledProjectPath\ from the beginning of the original compile path. 
#>

param(
       ## Folder that will be recursively searched for PDB files.
       [Parameter(Mandatory = $true)]
       [Alias("symbols")]
       [string] $symbolsFolder,
       
       ## github user ID
       [Parameter(Mandatory = $true)]
       [string] $userId,
       
       ## github repository name
       [Parameter(Mandatory = $true)]
       [string] $repository,
       
       ## github branch name
       [Parameter(Mandatory = $true)]
       [string] $branch,
       
       ## A root path for the source files
       [string] $sourcesRoot,
       
       ## Debugging Tools for Windows installation path
       [string] $dbgToolsPath,
       
       ## Github URL
       [string] $gitHubUrl,
       
       ## Ignore a source path that contains any of the strings in this array
       [string[]] $ignore,
       
       ## Ignore paths other than the source root
       [switch] $ignoreUnknown,
       
       ## Server serves raw: don't concatenate /raw in the path
       [switch] $serverIsRaw,
       
       ## Verify the filenames in the tree in the local repository
       [switch] $verifyLocalRepo
       )
       

function CorrectPathBackslash {
  param([string] $path)
  
  if (![String]::IsNullOrEmpty($path)) {
    if (!$path.EndsWith("\")) {
      $path += "\"
    }
  }
  return $path
}

###############################################################

function FindLongestCommonPath {
  param([string] $path1,
        [string] $path2)
  
  $path1Parts = $path1 -split "\\"
  $path2Parts = $path2 -split "\\"
  
  $result = @()
  for ($i = 0; ($i -lt $path1Parts.Length) -and ($i -lt $path2Parts.Length); $i++) {
    if ($path1Parts[$i] -eq $path2Parts[$i]) {
      $result += $path1Parts[$i]
    }
  }
  return [String]::Join("\", $result)
}

###############################################################

function CheckDebuggingToolsPath {
  param([string] $dbgToolsPath)

  $dbgToolsPath = CorrectPathBackslash $dbgToolsPath

  # check whether the dbgToolsPath variable is set
  # it links to srctool.exe application
  if (![String]::IsNullOrEmpty($dbgToolsPath)) {
    if (![System.IO.File]::Exists($dbgToolsPath + "srctool.exe")) {
      Write-Debug "Debugging Tools not found at the given location - trying \srcsrv subdirectory..."
      # Let's try maybe also srcsrv
      $dbgToolsPath += "srcsrv\"
      if (![System.IO.File]::Exists($dbgToolsPath + "srctool.exe")) {
          throw "The Debugging Tools for Windows could not be found at the provided location."
      }
    }
    # OK, we are fine - the srctool exists
    Write-Verbose "Debugging Tools for Windows found at $dbgToolsPath."
  } else {
    Write-Verbose "Debugging Tools path not provided - trying to guess it..."
    # Let's try to execute the srctool and check the error
    if ($(Get-Command "srctool.exe" 2>$null) -eq $null) {
      # srctool.exe can't be found - let's try cdb
      $cdbg = Get-Command "cdb.exe" 2>$null
      if ($cdbg -eq $null) {
        $errormsg = "The Debugging Tools for Windows could not be found. Please make sure " + `
                    "that they are installed and reference them using -dbgToolsPath switch."
        throw $errormsg        
      }
      # cdbg found srctool.exe should be then in the srcsrv subdirectory
      $dbgToolsPath = $([System.IO.Path]::GetDirectoryName($dbg.Defintion)) + "\srcsrv\"
      if (![System.IO.File]::Exists($dbgToolsPath + "srctool.exe")) {
        $errormsg = "The Debugging Tools for Windows could not be found. Please make sure " + `
                    "that they are installed and reference them using -dbgToolsPath switch."
        throw $errormsg
      }
      # OK, we are fine - the srctool exists
      Write-Verbose "The Debugging Tools For Windows found at $dbgToolsPath."
    }
  }
  return $dbgToolsPath
}

###############################################################

function FindGitExe {
    $suffix = "\git\bin\git.exe"
    
    $gitexe = ${env:ProgramFiles} + $suffix
    if (Test-Path $gitexe) {
        return $gitexe
    }
    
    if( [IntPtr]::size -eq 4 ) {
        return $null
    }
    
    $gitexe = ${env:ProgramFiles(x86)} + $suffix
    if (Test-Path $gitexe) {
        return $gitexe
    }
    
    return $null
}

###############################################################

function WriteStreamHeader {
  param ([string] $streamPath)
  
  Write-Verbose "Preparing stream header section..."

  Add-Content -value "SRCSRV: ini ------------------------------------------------" -path $streamPath
  Add-Content -value "VERSION=1" -path $streamPath
  Add-Content -value "INDEXVERSION=2" -path $streamPath
  Add-Content -value "VERCTL=Archive" -path $streamPath
  Add-Content -value ("DATETIME=" + ([System.DateTime]::Now)) -path $streamPath
}

###############################################################

function WriteStreamVariables {
  param([string] $streamPath)
  
  Write-Verbose "Preparing stream variables section..."

  Add-Content -value "SRCSRV: variables ------------------------------------------" -path $streamPath
  Add-Content -value "SRCSRVVERCTRL=http" -path $streamPath
  Add-Content -value "HTTP_ALIAS=$gitHubUrl" -path $streamPath
  Add-Content -value "HTTP_EXTRACT_TARGET=%HTTP_ALIAS%/%var2%/%var3%$raw/%var4%/%var5%" -path $streamPath
  Add-Content -value "SRCSRVTRG=%http_extract_target%" -path $streamPath
  Add-Content -value "SRCSRVCMD=" -path $streamPath
}

###############################################################

function WriteStreamSources {
  param([string] $streamPath,
        [string] $pdbPath)
        
  Write-Verbose "Preparing stream source files section..."

  $sources = & ($dbgToolsPath + 'srctool.exe') -r $pdbPath 2>$null
  if ($sources -eq $null) {
    write-warning "No steppable code in pdb file $pdbPath, skipping";
    "failed";
    return;
  }

  Add-Content -value "SRCSRV: source files ---------------------------------------" -path $streamPath
  
  if ([String]::IsNullOrEmpty($sourcesRoot)) {
    # That's a little bit hard - we need to guess the source root.
    # By default we compare all source paths stored in the PDB file
    # and extract the least common path, eg. for paths:
    # C:\test\test1\test2\src\Program.cs
    # C:\test\test1\test2\src\Test.Domain\Domain.cs
    # we will assume that the source code archive was created from
    # the C:\test\test1\test2\src\ path - so be careful here!
      
    $sourcesRoot = $null
    foreach ($src in $sources) {
      if ($sourcesRoot -eq $null) {
        $sourcesRoot = [System.IO.Path]::GetDirectoryName($src)
        continue
      }
      $sourcesRoot = FindLongestCommonPath $src $sourcesRoot
    }  
    $warning = "Sources root not provided, assuming: '$sourcesRoot'. If it's not correct please run the script " + `
               "with correct value set for -sourcesRoot parameter."
    Write-Warning $warning
  }
  $sourcesRoot = CorrectPathBackslash $sourcesRoot
  $outputFileName = [System.IO.Path]::GetFileNameWithoutExtension($sourceArchivePath)
  
  #if we're verifying the local repo then get the tree list from the branch/commit
  $lstree = ""
  if ($verifyLocalRepo) {
    $gitexe = FindGitExe
    if (!$gitexe) {
      throw "Script error. git.exe not found";
    }
    
    $gitrepo = $sourcesRoot + ".git"
    if (!(Test-Path $gitrepo)) {
      throw "Script error. git repo not found: $gitrepo";
    }
    
    $lstree = & "$gitexe" "--git-dir=$gitrepo" ls-tree --name-only --full-tree -r "$branch"
    if ($LASTEXITCODE) {
      throw "Script error. git could not list the files from commit/branch: $branch";
    }
  }

  
  #other source files
  foreach ($src in $sources) {
    
    #if the source path $src contains a string in the $ignore array, skip it
    [bool] $skip = $false;
    foreach ($istr in $ignore) {
      $skip = ( ($istr) -and ($src.IndexOf($istr, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) );
      if ($skip) {
        break;
      }
    }
    if ($skip) {
      continue;
    }
    
    if (!$src.StartsWith($sourcesRoot, [System.StringComparison]::CurrentCultureIgnoreCase)) {
      if ($ignoreUnknown) {
        continue;
      } else {
        throw "Script error. The source path ($src) was invalid";
      }
    }
    $srcStrip = $src.Remove(0, $sourcesRoot.Length).Replace("\", "/")
    
    if ($verifyLocalRepo) {
      #get the filepath from the tree list
      if ($lstree -ceq $srcStrip) {
        $filepath = $srcStrip
      } else {
        $matches = $lstree -ieq $srcStrip
        if (!$matches.count) {
          $warning = "File path couldn't be verified, skipping: " + $srcStrip
          Write-Host "$warning" -foregroundcolor red -backgroundcolor black
          continue
        }
        #Write-Host $matches;
        if ($matches.count -ne 1) {
          throw "Script error. Multiple matches in tree found for $srcStrip : $matches";
        }
        $filepath = $matches[0]
      }
    } else {
      $filepath = $srcStrip
    }
    
    #Add-Content -value "HTTP_ALIAS=http://github.com/%var2%/%var3%$raw/%var4%/%var5%" -path $streamPath
    Add-Content -value "$src*$userId*$repository*$branch*$filepath" -path $streamPath
    Write-Verbose "Indexing source to $gitHubUrl/$userId/$repository$raw/$branch/$filepath"
  }
}

###############################################################
# START
###############################################################
if ($verifyLocalRepo) {
  $ignoreUnknown = $TRUE
}

if ([String]::IsNullOrEmpty($gitHubUrl)) {
    $gitHubUrl = "http://github.com";
}

# If the server serves raw then /raw does not need to be concatenated
if ($serverIsRaw) {
  $raw = "";
} else {
  $raw = "/raw";
}

# Check the debugging tools path
$dbgToolsPath = CheckDebuggingToolsPath $dbgToolsPath

$pdbs = Get-ChildItem $symbolsFolder -Filter *.pdb -Recurse
foreach ($pdb in $pdbs) {
  Write-Verbose "Indexing $($pdb.FullName) ..."

  $streamContent = [System.IO.Path]::GetTempFileName()

  try {
    # fill the PDB stream file
    WriteStreamHeader $streamContent
    WriteStreamVariables $streamContent
    $success = WriteStreamSources $streamContent $pdb.FullName
    if($success -eq "failed") {
        continue
    }
    
    Add-Content -value "SRCSRV: end ------------------------------------------------" -path $streamContent
    
    # Save stream to the pdb file
    $pdbstrPath = "{0}pdbstr.exe" -f $dbgToolsPath
    $pdbFullName = $pdb.FullName
    # write stream info to the pdb file
      
    Write-Verbose "Saving the generated stream into the PDB file..."
    . $pdbstrPath -w -s:srcsrv "-p:$pdbFullName" "-i:$streamContent"
    
    
    Write-Verbose "Done."
  } finally {
    Remove-Item $streamContent
  }
}
