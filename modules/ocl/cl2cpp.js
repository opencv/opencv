var fso = new ActiveXObject("Scripting.FileSystemObject"),
shell = new ActiveXObject("WScript.Shell"),
args = WScript.Arguments,
enginefile = WScript.FullName,
scriptFullPath = WScript.ScriptFullName,
indir = args[0],
outname = args[1],
outDir,
stdout = null;
-1 < enginefile.toLowerCase().indexOf("cscript.exe") && (stdout = WScript.StdOut);
function getDir(a) {
	return a.substring(0, a.lastIndexOf("\\") || a.lastIndexOf("/"))
}
if (!indir || !outname) {
	var scriptPath = getDir(WScript.ScriptFullName.toString()),
	indir = indir || scriptPath + "/src/kernels";
	outname || (outname = scriptPath + "/kernels.cpp", outDir = scriptPath)
} else {
	outDir = getDir(outname);
	try {
		fso.CreateFolder(outDir)
	} catch (err) {}
	
}
var infldr = fso.GetFolder(indir),
clrx = /([\w-]+)\.cl$/i, stripBeginningRx = /^(\s)+/i, stripSinglelineMstyle = /\/\*.*?\*\//ig,
outStream = fso.OpenTextFile(outname, 2, !0, -2);

outStream.write("// This file is auto-generated. Do not edit!\n\nnamespace cv{\n\tnamespace ocl{\n");
for (var res, cl_file, l, state, countFiles = 0, codeRows = 0, removedRows = 0, filei = new Enumerator(infldr.Files); !filei.atEnd(); filei.moveNext())
	if (cl_file = filei.item(), res = cl_file.Name.match(clrx)) {
		var cl_filename = res[1],
		inStream = cl_file.OpenAsTextStream(1);
		stdout && stdout.Write("Processing file " + cl_filename + ".cl...");
		outStream.writeLine("\t\tconst char* " + cl_filename + "=");
		state = 0;
		for (countFiles++; !inStream.AtEndOfStream; ) {
			l = inStream.readLine();
			stripSinglelineMstyle.lastIndex = 0;
			l = l.replace(stripSinglelineMstyle,
					"");
			var mline = l.indexOf("/*");
			0 <= mline ? (l = l.substr(0, mline), state = 1) : (mline = l.indexOf("*/"), 0 <= mline && (l = l.substr(mline + 2), state = 0));
			var slineBegin = l.indexOf("//");
			0 <= slineBegin && (l = l.substr(0, slineBegin));
			1 == state || !l ? removedRows++ : (l = l.replace(stripBeginningRx, "$1"), l = l.replace("\\", "\\\\"), l = l.replace("\r", ""), l = l.replace('"', '\\"'), l = l.replace("\t", "  "), codeRows++, outStream.writeLine('\t\t\t"' + l + '\\n"'))
		}
		outStream.writeLine("\t\t;");
		inStream.close();
		stdout && stdout.Write("done\n")
	}
outStream.writeLine("\t\t}\n\t}");
outStream.close();
var msg = "Merging OpenCL Kernels into cpp file has been FINISHED!\nFiles : " + countFiles + "\nCode rows : " + codeRows + "\nRemoved rows : " + removedRows;
stdout ? stdout.WriteLine(msg) : shell.Popup(msg, 1, "OpenCL Kernels to cpp file", 64);
