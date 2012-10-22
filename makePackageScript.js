//collect include
var fs = new ActiveXObject("Scripting.FileSystemObject"),
shell = new ActiveXObject("WScript.Shell"),
args = WScript.Arguments,
scriptFullPath = WScript.ScriptFullName,
modulesDir = args[0],
includeDir = args[1],
binariesDir = args[3],
incFoldersNames=["opencv2"];

//console or gui?
var enginefile=WScript.FullName;
var stdout=null;
if(enginefile.toLowerCase().indexOf("cscript.exe")>-1){
	stdout=WScript.StdOut;
}
function getDir(a) {
	return a.substring(0, a.lastIndexOf("\\") || a.lastIndexOf("/"))
}

//checking input

if (!modulesDir || !includeDir) {
	var scriptPath = getDir(WScript.ScriptFullName.toString()),
	modulesDir = modulesDir || (scriptPath + "/modules");
	includeDir = includeDir || (scriptPath + "/include");
	//shell.Popup(modulesDir+"\n"+includeDir, 0, "debug", 64);
} else {
	try {
		fs.CreateFolder(includeDir)
	} catch (err) {}
}

//copying include
if(stdout)stdout.WriteLine("Composing include...");
var infldr = fs.GetFolder(modulesDir);
var res,moduleDir,countDirs=0;
for(var moduleDirI = new Enumerator(infldr.SubFolders); !moduleDirI.atEnd(); moduleDirI.moveNext())
	if (moduleDir = moduleDirI.item()) {
		var modulename = moduleDir.Name;
		if(stdout)stdout.Write("Processing dir "+modulename+"...");
		var ced=0;
		for(var i=0;i<incFoldersNames.length;i++){
			try{
				fs.CopyFolder((moduleDir.ShortPath+"/include/"+incFoldersNames[i]),includeDir+"/"+incFoldersNames[i]);
				ced++;
			}
			catch(e){
				
			}	
		}
		countDirs+=ced;
		if(stdout)stdout.Write((ced?"done\n":"skipped\n"));
	}
// copiing binaries, pdb, etc

//displaying message
var msg="Copying include was FINISHED!\nModules : " + countDirs;
if(stdout){
	stdout.WriteLine(msg);
}else{
	shell.Popup(msg, 1, "OpenCV WSH installer", 64);
}
