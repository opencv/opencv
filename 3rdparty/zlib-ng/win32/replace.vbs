strInputFileName = Wscript.Arguments(0)
strOutputFileName = Wscript.Arguments(1)
strOldText = Wscript.Arguments(2)
strNewText = Wscript.Arguments(3)

Set objFSO = CreateObject("Scripting.FileSystemObject")
Set objFile = objFSO.OpenTextFile(strInputFileName, 1)

strText = objFile.ReadAll
objFile.Close
strNewText = Replace(strText, strOldText, strNewText)

Set objFile = objFSO.OpenTextFile(strOutputFileName, 2, True)
objFile.Write strNewText
objFile.Close
