import os
import re
import io
import sys
import time
import json
import math
import shutil
import inspect
import warnings
import importlib
from lxml import etree # type: ignore
from pyflakes.api import checkPath
from pyflakes.reporter import Reporter

TxmlDirPath="."
cv2ModulePath="cv2"
scriptDIR=os.path.dirname(os.path.abspath(__file__))
inheritRecordFilePath=f"{time.time()}_{os.getpid()}_inherit.json"
knowTypes={} # Cache for the getType function

Krettypes={} # Record the parameters (these parameters are passed as arguments in C++, but are returned as outputs in Python) corresponding types. When checking unknown return types, these parameters will be defined as the correct types``
indexxmlRoot=None

def getFinallyObj(name):
    # Parse a string into an object, for example "cv.ORB" returns cv.ORB
    rootModule=cv2ModulePath
    paths = name.split(".")
    obj = importlib.import_module(rootModule)
    for p in paths:
        obj = getattr(obj, p)
    return obj

def isModule(name):
    try:
        obj=getFinallyObj(name)
        return inspect.ismodule(obj)
    except:
        return False

def isClass(name):
    obj = getFinallyObj(name)
    return inspect.isclass(obj)


def isFunc(name):
    obj = getFinallyObj(name)
    return callable(obj)


def getOtherType(name):
    obj = getFinallyObj(name)
    return type(obj)

def getType(name):
    # If it is 'module', 'class', or 'func', return their string; for example, 'cv2' will return 'module'. For other types, directly call type() to return; for example, 'cv2.SORT_EVERY_ROW' will return <class 'int'>
    global knowTypes
    ntype=None
    if name in knowTypes:
        return knowTypes[name]
    elif isModule(name):
        ntype = "module"
    elif isClass(name):
        ntype = "class"
    elif isFunc(name):
        ntype = "func"
    else:
        ntype = getOtherType(name)
        if ntype == type(str.__init__):
            ntype="func"
    knowTypes[name] = ntype
    return ntype


def isExist(name):
    # Check if this attribute exists in cv2
    try:
        getFinallyObj(name)
        return True
    except:
        return False

def finddocfilefromxml(cppname):
    # Return all matching refids based on the passed fully qualified name, i.e., [refid, ....]
    # This function is only used to find the corresponding function id in index.xml
    root=indexxmlRoot
    nameIndex=cppname.rfind("::")
    classname=cppname[:nameIndex]
    l1=root.xpath(f"compound/name[text()='{classname}']") # type: ignore
    if len(l1)==1:
        # Under normal circumstances, there should only be one corresponding field
        compoundTag=l1[0].getparent()
        targetIDs=[]
        targetName=cppname[nameIndex+2:].replace(" ","")
        # Get all IDs of this function
        memberNames = compoundTag.xpath(f"member/name[text()='{targetName}']")
        # Functions may be overloaded, so there may be multiple IDs
        for memberName in memberNames:
            targetIDs.append(memberName.getparent().get("refid"))

        return targetIDs
    elif len(l1)==0:
        return None
    else:
        print("error: match multiple doc!!!!!!!!!")
        return None

def gettypefromCXXtypes(CXXtypesTopylist,CXXtype):
    # CXXtypesTopylist is a manually written JSON that determines which C++ type should be converted to the corresponding Python type
    for key in CXXtypesTopylist:
        if CXXtype in CXXtypesTopylist[key]:
            return key
    return None

def cvtCXXToPYtype(CXXtpyestr0,tpyeisAnyListfile=os.path.join(scriptDIR,"CXXtypelist.txt"),CXXtypesFile=os.path.join(scriptDIR,"CXXtypes.json")):
    # Convert C++ types to Python types, with the default value being "typing.Any"

    # Remove optional prefixes and suffixes in the correspondence between C++ types and Python types. For example: const String and String both actually correspond to str in Python, so const is optional and should be removed
    CXXtypestr=CXXtpyestr0.removeprefix("const").rstrip("*").rstrip("&").strip()
    
    with open(tpyeisAnyListfile) as f:
        lines=f.readlines()
        tpyeisAnyList=[i.strip() for i in lines]
    with open(CXXtypesFile) as f:
        CXXtypesTopylist=json.loads(f.read())

    t=gettypefromCXXtypes(CXXtypesTopylist,CXXtypestr)
    if t!=None:
        # If a definite conversion relationship already exists, directly return the converted Python type
        return t
    elif CXXtpyestr0 in tpyeisAnyList:
        # tpyeisAnyList is a list containing some C++ types, which, due to their complexity, have temporarily been converted to the Any type in Python
        return "typing.Any"
    else:
        print(f"warning: noknown type: {CXXtypestr}")
        return "typing.Any"

def getFuncInfos(cppname,xmlDirPath):
    """
    If there are no related documents, return []
    ret=[
            {
            "static":False,
            "retType":"...",
            "overload":False,
            "argInfo":{
                "argName":{"type":type, "doc":""},
                ....
                },
            "doc":""
            },
            {...}
        ]
    """
    # Get the id of a function
    refids=finddocfilefromxml(cppname)
    retOverLoad=False

    if refids==None:
        # There is no related documentation for this function
        return []
    if len(refids)>1:
        retOverLoad=True

    rets=[] # The function may be overloaded, so it will return multiple documents
    for refid in refids:
        ret={
                "overload":retOverLoad,
                "static":False,
                "argInfo":{}
                }
        # refid contains the path of the file where the function documentation is located, so it can be concatenated directly
        targetXmlFilePath=os.path.join(xmlDirPath,refid[:refid.rfind("_")])+".xml"
        try:
            tree=etree.parse(targetXmlFilePath)
        except:
            print(f"File {targetXmlFilePath} parsing error, please check whether you are using doxygen 1.16.1 or a newer version, and delete the generated sutbs.")
        root=tree.getroot()

        memberdefs=root.xpath(f"compounddef/sectiondef/memberdef[@id='{refid}' and @prot='public']")
        if memberdefs==[]:
            continue
        memberdef=memberdefs[0] # refid is unique, so the length of memberdefs should be 1, you can directly use [0]

        # Get the return type of a function
        retType=''.join(memberdef.xpath("type")[0].itertext())
        ret["retType"]=retType.strip()
        
        # Get whether the function is a static function
        if memberdef.get("static")=="yes":
            ret["static"]=True

        # Documentation and types for obtaining parameters
        for param in memberdef.xpath("param"):
            count=0
            # Default parameter names: Some C++ function signatures only have types without parameter names. In the Python part of the documentation, they will be written in the form of arg{count}.
            paramName=f"arg{count}"
            while paramName in ret["argInfo"]:
                count+=1
                paramName=f"arg{count}"
            
            # Try to get the corresponding parameter name
            decnames=param.xpath("declname")
            if decnames!=[]:
                paramName=''.join(decnames[0].itertext())
            # Get parameter type
            paramType=''.join(param.xpath("type")[0].itertext())
            parameternameTags = memberdef.xpath(f"detaileddescription/para/parameterlist/parameteritem/parameternamelist/parametername[text()='{paramName}']")
            
            # Get Parameter Documentation
            paramDoc=""
            if parameternameTags!=[]:
                paramDocTags=parameternameTags[0].getparent().getparent().xpath("parameterdescription/para")
                if paramDocTags!=[]:
                    paramDoc=''.join(paramDocTags[0].itertext())

            ret["argInfo"][paramName]={"type":paramType,"doc":paramDoc}

        # Get the documentation of a function
        doc=""
        briefparaTags=memberdef.xpath("briefdescription/para")
        if briefparaTags!=[]:
            doc=''.join(briefparaTags[0].itertext())
        ret["doc"]=doc

        rets.append(ret)

    return rets


def getPySignList(rootPath):
    # Get all classes, functions, and constants of cv2 existing in the current Python environment
    # Will readjust the layout of the json
    with open(os.path.join(rootPath, "modules/python_bindings_generator/pyopencv_signatures.json")) as f:
        j = json.loads(f.read())
    d = []
    for i in j:
        for ii in j[i]:
            newii = ii
            newii["name"] = ii["name"][3:]
            newii["cppname"] = i
            d.append(newii)
    newd = []
    for i in d:
        if isExist(i["name"]):
            newd.append(i)
    return newd

def filterNotExist(newd):
    return [i for i in newd if isExist(i["name"])]

def cvtFuncJsonToPy(jSignDict):
    # Return the corresponding valid Python function definition statement
    ret = "def "
    ret += jSignDict["name"].split('.')[-1]
    strArg = jSignDict["arg"].replace(']', "")
    indexs = [m.start() for m in re.finditer(r"\[, ", strArg)]
    index=strArg.find("[, ")
    for _ in range(len(indexs)):
        index += 3
        while index < len(strArg) and re.match(r"[a-zA-Z_0-9]", strArg[index]):
            index += 1
        if index == len(strArg):
            strArg += "=..."
        else:
            strArg = strArg[:index]+"=..."+strArg[index:]
        index+=3
        index=strArg.find("[, ",index)
    strArg = strArg.replace('[', "")
    if strArg.startswith(", "):
        strArg=strArg[2:]
    strFuncRet=jSignDict["ret"]
    
    if len(strFuncRet.split(','))>1:
        strFuncRet="tuple["+strFuncRet+']'

    ret += f"({strArg}) -> {strFuncRet}: ..."
    return ret


def cvtClassJsonToPy(jSignDict):
    # Return the corresponding valid Python class definition statement
    return f"class {jSignDict["name"].split('.')[-1]}"

def cvtConstJsonTopy(jSignDict):
    # Return the corresponding valid Python constant definition statement
    jtype = getOtherType(jSignDict["name"])
    ret = f"{jSignDict["name"]}:{jtype.__name__}=..."
    return ret


def cvtJsonToPy(jSignDict):
    # Return the corresponding Python definition statement
    if "ret" in jSignDict:
        return cvtFuncJsonToPy(jSignDict)
    elif "value" in jSignDict:
        return cvtConstJsonTopy(jSignDict)
    else:
        return cvtClassJsonToPy(jSignDict)

def TryCreateFile(filePath):
    # If the file does not exist, the directories and file on the path will be created; if it exists, it will not have any effect on the file
    # For example: a/b/c.txt If a/b/c.txt does not exist, and if a/b does not exist, create the a/b directory. If a/b/c.txt does not exist, create the file and write the contents of base.pyi into it.
    # Call this function before writing to the file to ensure the file exists
    if os.path.exists(filePath):
        return
    dirname=os.path.dirname(filePath)
    basePyiPath=os.path.join(scriptDIR,"base.pyi")
    if dirname!="":
        os.makedirs(dirname,exist_ok=True)
    f=open(filePath,"a+")
    f.seek(0)
    txt=f.read()
    if txt=="":
        with open(basePyiPath) as basef:
            f.write(basef.read())
    f.close()

def writeclass(child,filePath):
    # Write class definition to the specified file
    TryCreateFile(filePath)
    l=child["name"].split(".")
    classl=child["classl"]
    insertIndex=0
    strTAB=""
    with open(filePath) as f:
        content=f.read()
    if classl!=[]:
        # classl indicates which classes contain the definition of this class, that is, which classes are nested within it (not base classes)
        strTAB=' '*4*len(classl)
        for tclassName in classl:
            insertIndex=content.find(f"class {tclassName}",insertIndex)
        insertIndex=insertIndex+1+content[insertIndex+1:].find('\n')
        elipIndex=content.rfind(" ...",insertIndex-5,insertIndex)
        if elipIndex!=-1:
            removeFileStr(filePath,elipIndex,elipIndex+3)
            insertIndex-=4
    else:
        insertIndex=len(content)

    writeClassStr=strTAB

    # Get the class name
    if l!=[]:
        writeClassStr+=f"class {l[-1]}"
    else:
        writeClassStr+=f"class {child['name']}"

    # Get base class
    finobjs=getFinallyObj(child["name"]).__bases__
    finobjNames=child["baseClassl"]
    if finobjNames!=[]:
        for i in finobjs:
            recordInherit(filePath,i.__module__,i.__name__)
        writeClassStr+=f"({','.join(finobjNames)})"
    
    # Use ... as a placeholder, temporarily set as an empty class
    writeClassStr+=": ...\n"
    insertText(filePath,insertIndex+1,writeClassStr)

def recordInherit(filePath,inherit,classname):
    # Record the base classes of these classes
    # Generate a temporary JSON in the current directory for recording
    with open(inheritRecordFilePath,"r+") as f:
        content=f.read()
    j={}
    if content!="":
        j=json.loads(content)
    
    filePath=os.path.normpath(filePath)
    if filePath not in j:
        j[filePath]={}

    if inherit not in j[filePath]:
        j[filePath][inherit]=[]

    if classname not in j[filePath][inherit]:
        j[filePath][inherit].append(classname)

    jstr=json.dumps(j)
    with open(inheritRecordFilePath,'w') as f:
        f.write(jstr)

def getInheritFilePath(name,outPath):
    # Get the path of the file where the base class is located; if it is __init__.pyi, then it is the path of the containing directory
    name=name.removeprefix("cv2.")
    name=name.replace(".","/")
    return os.path.normpath(os.path.join(outPath,name))

def cvtPathtoPyimport(key,classname):
    # Convert to py import statement
    if key==".":
        return f"from . import {classname}"
    elif set(list(key))==set(['/','.']) or key=="..":
        count=key.count("..")+1
        return f"from {'.'*count} import {classname}"
    else:
        print(f"warning: need add new cvtPathtoPyimport rules bacuse:  noknown key:{key} classname:{classname}")

def insertText(filePath,index,text):
    # Insert text at the specified location in the file
    content=""
    TryCreateFile(filePath)
    with open(filePath) as f:
        content=f.read()
    with open(filePath,"w") as f:
        if index<len(content):
            f.write(content[:index]+text+content[index:])
        else:
            f.write(content+text)

def writeInherit(outPath):
    # Write the import statements of the base classes of these classes at the beginning of the file
    with open(inheritRecordFilePath) as f:
        j=json.loads(f.read())

    for filePath in j:
        with open(filePath) as f:
            index=f.read().find('\nT0=typing.TypeVar("T0")\n')+1
        
        for i in j[filePath]:
            for ii in j[filePath][i]:
                targetPath=""
                if i == "cv2":
                    targetPath=outPath
                else:
                    targetPath=getInheritFilePath(i,outPath)
                    if os.path.basename(targetPath)!="numpy" and os.path.samefile(targetPath+".pyi",filePath):
                        continue
                if os.path.basename(targetPath)!="numpy":
                    filePath=os.path.normpath(filePath)
                    fdir=os.path.dirname(filePath)
                    relp=os.path.relpath(targetPath,fdir)
                    if relp=="." and filePath==os.path.join(outPath,"__init__.pyi"):
                        continue
                    Pyimport=cvtPathtoPyimport(relp,ii)+"\n" # type: ignore
                else:
                    Pyimport="from numpy import ndarray as numpyndarray\n"
                PyimportLen=len(Pyimport)
                insertText(filePath,index,Pyimport)
                index+=PyimportLen

        insertText(filePath,index,'\n')

def getMostSimilar(child,params,infos):
    # When obtaining function documentation, there may be multiple entries; this function allows you to get the documentation that best matches the function in Python
    paramsAndret=params+[i.strip() for i in child["ret"].split(",")]
    maxSimilarNum=0
    minNoSimilarNum=math.inf
    maxSimilarinfol=[]
    minNoSimilarinfol=[]
    isvoidFuncs=[]
    for info in infos:
        similarNum=0
        nosimilarNum=0
        for CXXparam in info["argInfo"]:
            # Calculate similarity based on the names and number of parameters
            if CXXparam in paramsAndret:
                similarNum+=1
            else:
                nosimilarNum+=1

        # First Screening
        # Update the most similar to maxSimilarinfol
        if similarNum>maxSimilarNum:
            maxSimilarNum=similarNum
            maxSimilarinfol=[info]
        elif similarNum==maxSimilarNum:
            maxSimilarinfol.append(info)
        
        # Second screening, update the most similar one in the dissimilar list
        if nosimilarNum<minNoSimilarNum:
            minNoSimilarNum=nosimilarNum
            minNoSimilarinfol=[info]
        elif nosimilarNum==minNoSimilarNum:
            minNoSimilarinfol.append(info)
        
        # Third screening, if this Python function returns None, it should correspond to void in C++
        if info["retType"]=="void":
            isvoidFuncs.append(info)

    if len(maxSimilarinfol)==1:
        # During the first screening, the corresponding document was already found.
        return maxSimilarinfol[0]
    if child["ret"]=="None" and len(isvoidFuncs)==1:
        # Third Screening
        return isvoidFuncs[0]

    if len(minNoSimilarinfol)==1:
        return minNoSimilarinfol[0]
    
    # Find the overlapping parts from the list obtained through the first and second screenings
    # At this point, if multiple most relevant documents are still found, even manually searching based on the input cannot distinguish them (manually verified in February 2026)
    overlap= [i for i in maxSimilarinfol if i in minNoSimilarinfol]
    if len(overlap)!=0:
        return overlap[0]
    return maxSimilarinfol[0]

def getFuncInfo(child,params,xmlDirPath):
    # Get function documentation
    infos=getFuncInfos(child["cppname"],xmlDirPath)
    params=[i.rstrip("=...") for i in params]
    if len(infos)==1:
        return infos[0]
    if len(infos)==0:
        return {}
    return getMostSimilar(child,params,infos)

def getIndexFromlist(l,item):
    # Get the index of the first matching value from a list
    try:
        return l.index(item)
    except ValueError:
        return -1

def getHasHint(child,strTAB,xmlDirPath):
    # When a function has documentation, return a valid Python function definition statement that includes the documentation and type hints
    # If there is no documentation, return a valid Python function definition statement without documentation (if there are generics, it will include generic type hints)

    global Krettypes
    pysign=cvtJsonToPy(child)
    params=pysign[pysign.find("(")+1:pysign.rfind(")")].split(",")
    params=[i.strip() for i in params]
    returnTypes=child["ret"].split(",")
    returnTypes=[i.strip() for i in returnTypes]
    pyFuncName=pysign[pysign.find(' ')+1:pysign.rfind('(')]
    info=getFuncInfo(child,params,xmlDirPath)
    oneTAB=' '*4
    Tcount=0

    if info=={}:
        # No Documentation
        finallyParams=[]
        # Add generic return type hint
        for param in params:
            paramName=param
            index=param.find('=')
            hint=""
            if index!=-1:
                paramName=param[:index]
            if paramName in returnTypes:
                hint=f":T{Tcount}"
                paramIndex=getIndexFromlist(returnTypes,paramName)
                returnTypes[paramIndex]=f"T{Tcount}"
                Tcount+=1
            if index!=-1:
                hint+=param[index:]
            finallyParams.append(paramName+hint)
        returnHint=','.join(returnTypes)
        if len(returnTypes)>1:
            returnHint="tuple["+returnHint+"]"

        return strTAB+f"def {pyFuncName}({','.join(finallyParams)}) -> {returnHint}: ..."

    # Get parameter type hints
    finallyParams=[]

    # Some parameters are returned in py
    params2=[i.removesuffix("=...") for i in params]
    for aarg in info["argInfo"]:
        if (aarg in returnTypes) and (aarg not in params2):
            counta=1
            aarg2=aarg
            while (aarg2 in Krettypes) and (Krettypes[aarg2]!=info["argInfo"][aarg]["type"]):
                aarg2=f"{aarg}{counta}"
                counta+=1
            Krettypes[aarg2]=info["argInfo"][aarg]["type"]
            returnTypes[getIndexFromlist(returnTypes,aarg)]=aarg2

    # In py signature, the parameters part
    for param in params:
        paramName=param
        index=param.find('=')
        hint=""
        paramType="typing.Any"
        if index!=-1:
            paramName=param[:index]

        if paramName in info["argInfo"]:
            paramType=info["argInfo"][paramName]["type"]
            paramType=cvtCXXToPYtype(paramType)

        if paramType=="typing.Any":
            if paramName in returnTypes:
                # Add generic return type hint
                hint=f":T{Tcount}"
                paramIndex=getIndexFromlist(returnTypes,paramName)
                returnTypes[paramIndex]=f"T{Tcount}"
        else:
            hint=":"+paramType

        if index!=-1:
            hint+=param[index:]
        
        finallyParams.append(f"{paramName}{hint}")

    # Get function documentation
    # Format Document
    strTAB2=strTAB+oneTAB
    infodoc=info["doc"]
    if infodoc!="":
        infodoc="```\n"+infodoc+"\n```\n"

    FuncDoc=strTAB2+infodoc
    paramHasDoc=False
    for paramName in info["argInfo"]:
        if info['argInfo'][paramName]['doc']!="":
            paramHasDoc=True
            FuncDoc+="---\n```\nParameters:\n"
            break

    alignLen=28
    if paramHasDoc:
        paramLens=[len(i) for i in info["argInfo"]]
        maxlen=max(paramLens)
        alignLen=max(maxlen,28)
    for paramName in info["argInfo"]:
        if info['argInfo'][paramName]['doc']=="":
            continue
        paramName2=paramName+':'
        FuncDoc+=f"{paramName2:<{alignLen}}{info['argInfo'][paramName]['doc']}\n"

    if paramHasDoc:
        FuncDoc+="```\n"

    FuncDoc=FuncDoc.replace('\n','\n'+strTAB2)
    FuncDoc=strTAB2+'"""\n'+FuncDoc+'"""'

    if FuncDoc.replace("\n","").replace(" ","").replace('`',"").strip() == '""""""':
        FuncDoc=strTAB2+'""""""'
    
    # Return type hint
    returnHint=','.join(returnTypes)
    if len(returnTypes)>1:
        returnHint="tuple["+returnHint+"]"
    
    finallyFuncSign=f"def {pyFuncName}({','.join(finallyParams)}) -> {returnHint}:"

    # Add markers for static functions and overloaded functions
    if info["static"]:
        finallyFuncSign="@staticmethod\n"+strTAB+finallyFuncSign
    if child["overload"]:
        finallyFuncSign="@overload\n"+strTAB+finallyFuncSign

    retstr=f"{strTAB}{finallyFuncSign}\n{FuncDoc}"
    return retstr

def removeFileStr(filePath,startIndex,endIndex):
    """
    Will delete the characters at startIndex and endIndex positions, as well as all characters in between
    """
    with open(filePath) as f:
        content=f.read()
    content=content[:startIndex]+content[endIndex+1:]
    with open(filePath,"w") as f:
        f.write(content)

def writeFunc(child,filePath,classl):
    strTAB=""
    if classl!=[]:
        # If this function is inside a certain class
        strTAB=' '*4*len(classl)
        insertIndex=0
        
        with open(filePath) as f:
            content=f.read()
            # Finding the Right Insertion Point
            insertIndex=0
            for tclassName in classl:
                insertIndex=content.find(f"class {tclassName}",insertIndex)
            insertIndex=insertIndex+1+content[insertIndex+1:].find('\n')
            elipIndex=content.rfind(" ...",insertIndex-5,insertIndex)
        if elipIndex!=-1:
            # The class where this function resides is no longer empty due to the insertion of this function, so delete the ... of this class
            removeFileStr(filePath,elipIndex,elipIndex+3)
            insertIndex-=4
        # Construct the signature of the inserted function
        text=getHasHint(child,strTAB,TxmlDirPath)
        index=text.find("(")+1
        if "@staticmethod" not in text and index!=text.find(")"):
            text=text[:index]+"self,"+text[index:]
        elif "@staticmethod" not in text:
            text=text[:index]+"self"+text[index:]
        # Insert
        insertText(filePath,insertIndex+1,f"{text}\n\n")
    else:
        TryCreateFile(filePath)
        with open(filePath,"a") as f:
            f.write(f"{getHasHint(child,strTAB,TxmlDirPath)}\n\n")

def writeClassOrFunc(outPath,child,classl):
    if child["filePath"]=="..pyi":
        child["filePath"]="root.pyi"
    filePath=os.path.join(outPath,child["filePath"])
    TryCreateFile(filePath)

    if child["type"]=="class":
        writeclass(child,filePath)
    else:
        writeFunc(child,filePath,classl)

def getConstType(name):
    obj=getFinallyObj(name)
    if type(obj)==type(1):
        return "int"
    else:
        print("warning: noknown "+name)

def writeConst(outPath,child,key):
    if child["filePath"]=="..pyi":
        child["filePath"]="root.pyi"

    filePath=os.path.join(outPath,child["filePath"])
    TryCreateFile(filePath)
    with open(filePath,"a") as f:
        f.write(f"{key}:{getConstType(child['name'])} = ...\n")

def handleLeaf(leaf,outPath,key):
    if leaf["type"] in ["func","class"]:
        writeClassOrFunc(outPath,leaf,leaf["classl"])
    elif leaf["type"] in [type(1)]:
        writeConst(outPath,leaf,key)
    elif leaf["type"] != "module":
        print(f"noknow {key}:\n {leaf} ")

def getFilePathAndClasss(node):
    # Which file should the node retrieval be written in and which class should include it
    # Because classes may be nested, classl is a list
    classl=[]
    filePath="."
    nodeli=node["name"].split('.')
    fname=""
    cppnames=node["cppname"].split("::")
    if len(cppnames)>2 and cppnames[-1]==cppnames[-2] and getType(node["name"])=="class":
        nodeli.append("__init__")

    for i in nodeli[:-1]:
        fname+="."+i
        itype=getType(fname[1:])
        if itype=="module":
            filePath=os.path.join(filePath,i)
        elif itype == "class":
            classl.append(i)
        elif itype not in [type(1)]:
            print(f"noknown type: {i}:{itype} {node}")
    if filePath==".":
        filePath="__init__"
    filePath+=".pyi"
    return filePath,classl

def organise_pyi(targetPath):
    # Organizing pyi files
    # If there is a file name (excluding the extension) that is the same as a directory name in the same directory, move it to the directory with the same name and rename it to __init__.pyi
    # Write import statements for all modules in the same directory in __init__.pyi
    for root,_,files in os.walk(targetPath):
        now_dir_name=os.path.basename(root)
        if "__init__.pyi" not in files:
            srcPath =os.path.join(root,f"../{now_dir_name}.pyi")
            if not os.path.exists(srcPath):
                continue
            destPath=os.path.join(root,"__init__.pyi")
            shutil.move(srcPath,destPath)

        initFilePath=os.path.join(root,"__init__.pyi")
        with open(initFilePath,"a") as f:
            for name in os.listdir(root):
                if "__init__.pyi"==name:
                    continue
                if name.endswith(".pyi"):
                    name=name[:-4]

                f.write(f"\nfrom . import {name}")
 

def handlen(node,outPath):
    handleLeaf(node,outPath,node["name"].split('.')[-1])

def swapn(newdclass,n1,n2):
    # Swap the positions of two values in a list
    newdclass2=newdclass.copy()
    newdclass2[n1], newdclass2[n2]=newdclass[n2], newdclass[n1]
    return newdclass2

def findclassIndex(newdclass,name):
    # Return the index based on the name
    name2='.'+name
    for i,item in enumerate(newdclass):
        itemname=item["name"]
        if itemname==name or itemname.endswith(name2):
            return i

def sortclass(newdclass):
    # Sort by class write order
    newdclass2=newdclass
    neednext=True
    while neednext:
        neednext=False
        for n,item in enumerate(newdclass2):
            l={"classl":-1,"baseClassl":-1}
            if item["classl"]!=[]:
                l["classl"]=n
            if newdclass2[n]["baseClassl"]!=[]:
                l["baseClassl"]=n
            
            for key in l:
                if l[key]==-1:
                    continue
                for classname in newdclass2[l[key]][key]:
                    index=findclassIndex(newdclass2,classname)
                    if index==None:
                        if newdclass2[l[key]][key] == ['ndarray']:
                            newdclass2[l[key]][key]=["numpyndarray"]
                        continue
                    if index<n:
                        continue
                    newdclass2=swapn(newdclass2,index,n)
                    neednext=True
                    break

    return newdclass2

def sortnewd(newd):
    # Sorting
    # Constants are written first, then classes, and finally functions
    newdconst=[]
    newdclass=[]
    newdfunc =[]
    for i in newd:
        if "value" in i:
            newdconst.append(i)
        elif "ret" in i:
            newdfunc.append(i)
        else:
            newdclass.append(i)

    newdclass=sortclass(newdclass)
    return newdconst + newdclass + newdfunc

def removeDup(newd):
    # Delete duplicates in a list
    newd2=[]
    for i in newd:
        if i not in newd2:
            newd2.append(i)
    return newd2

def getretType2(CXXtype,CXXtypesFile=os.path.join(scriptDIR,"CXXtypes.json")):
    # Convert the return value of the corresponding C++ function to a Python type
    CXXtypestr=CXXtype.removeprefix("const").rstrip("*").rstrip("&").strip()
    with open(CXXtypesFile) as f:
        CXXtypesTopylist=json.loads(f.read())
    for key in CXXtypesTopylist:
        for v in CXXtypesTopylist[key]:
            index=CXXtypestr.find(v)
            if index!=-1 and (not (((index!=0 and re.match("[a-zA-Z_0-9]",CXXtypestr[index-1])) or (index+len(v)<len(CXXtypestr) and re.match("[a-zA-Z_0-9]",CXXtypestr[index+len(v)]))))):
                CXXtypestr=CXXtypestr.replace(v,key)

    if "std::vector" in CXXtypestr and "<" in CXXtypestr and ">":
        CXXtypestr=CXXtypestr.replace("std::vector","Sequence").replace('<','[').replace('>',']')
        return CXXtypestr
    try:
        obj=getFinallyObj("typing."+CXXtypestr)
        return str(obj)
    except:
        return "typing.Any"

def isclassFromxml(name):
    # Determine whether it is a class through an XML document
    root=indexxmlRoot
    l1=[i for i in root.xpath(f"compound/name[contains(text(),'::{name}')]") if i.text and i.text.endswith("::"+name)] # type: ignore
    if len(l1)>0:
        return True
    return False
 
def getretHasClass(NretType):
    lns=["KeyPoint","cv::RotatedRect","DMatch"]
    for i in lns:
        if i in NretType:
            return i
    return None


def addNoknownType(outPath):
    # Define types that are undefined in the return type hints of the final function
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    for root,_,files in os.walk(outPath):
        for file in files:
            # Use the inspector to get undefined types in the file
            output=io.StringIO()
            reporter=Reporter(output,output)
            if not file.endswith(".pyi"):
                continue

            checkPath(os.path.join(root,file),reporter)
            noknownTypel=[]
            for line in output.getvalue().split('\n'):
                if not ": undefined name '" in line:
                    continue
                line=line[line.find(" name "):]
                noknownTypel.append(re.findall(r"'([a-zA-Z_0-9]+)'",line)[0])
            
            noknownTypel=list(set(noknownTypel))
            f=open(os.path.join(root,file),'a')
            f.write("\n")
            # Handle each undefined type
            for i in noknownTypel:
                if i=="Mat":
                    relp=os.path.relpath(outPath,root)
                    text=cvtPathtoPyimport(relp,i)
                elif i=="matches_info":
                    text=f"{i}=MatchesInfo"
                elif i in Krettypes:
                    t=cvtCXXToPYtype(Krettypes[i])
                    if t=="typing.Any":
                        t=getretType2(Krettypes[i])
                    if t=="typing.Any" and isclassFromxml(i[0].upper()+i[1:]):
                        t=i[0].upper()+i[1:]

                    text=f"{i}={t}"
                else:
                    text=f"{i}=typing.Any"
                
                retHasClass=getretHasClass(text)
                if retHasClass!=None:
                    relp=os.path.relpath(outPath,root)
                    if not (relp=="." and os.path.samefile(os.path.join(root,file),os.path.join(outPath,"__init__.pyi"))):
                        text=text.replace("cv::","") # type: ignore
                        Pyimport=cvtPathtoPyimport(relp,retHasClass.replace("cv::",""))+"\n"
                        text=Pyimport+text
                if "Pose3DPtr" in text:
                    text=text.replace("Pose3DPtr","Pose3D")
                if "numpy.ndarray" in text:
                    text=text.replace("<class 'numpy.ndarray'>","numpyndarray")
                    text="from numpy import ndarray as numpyndarray\n"+text
                text=text.replace("cv::Rect","Rect")
                f.write(f"\n{text}")

def findchilds(name,newd):
    # Return the corresponding childs based on the name
    childs=[]
    for i in newd:
        if i["name"]==name:
            childs.append(i)
    return childs

def getBaseClasss(classname):
    # Get the base class of a function
    finobjs=getFinallyObj(classname).__bases__

    if (len(finobjs)==1 and finobjs[0]!=object) or (len(finobjs)>1):
        finobjNames=[i.__name__ for i in finobjs]
        return finobjNames
    return []

def addMoreInfoTonewd(newd):
    # Add more attributes to each item in the list
    newd2=newd.copy()
    for n,node in enumerate(newd):
        filePath,classl=getFilePathAndClasss(node)
        ntype=getType(node["name"])
        node["type"]=ntype
        node["filePath"]=filePath
        node["classl"]=classl
        cppnames=node["cppname"].split("::")
        if len(cppnames)>2 and cppnames[-1]==cppnames[-2]:
            node["name"]+=".__init__"
            node["type"] ="func"
            node["ret"]  ="None"

        if len(findchilds(node["name"],newd))>1:
            node["overload"]=True
        else:
            node["overload"]=False
        if node["type"]=="class":
            baseClassl=getBaseClasss(node["name"])
            node["baseClassl"]=baseClassl
        newd2[n]=node

    return newd2

def applyPatch(newd):
    # Apply Patch
    newd2=[]
    patchPath=os.path.join(scriptDIR,"patch.json")
    with open(patchPath) as f:
        j=json.loads(f.read())
    for i in newd:
        childs=findchilds(i["name"],j)
        retChild=i
        if childs!=[]:
            retChild=retChild|childs[0]
        newd2.append(retChild)
    patchPath2=os.path.join(scriptDIR,"patch2.json")
    with open(patchPath2) as f:
        j=json.loads(f.read())

    return newd2+j

def main():
    global TxmlDirPath,indexxmlRoot,cv2ModulePath
    rootPath = sys.argv[1]
    outPath = sys.argv[2]
    if len(sys.argv)>3:
        sys.path.insert(0,sys.argv[3])
    cv2_stubsPath=os.path.join(outPath,"cv2")
    TxmlDirPath=os.path.join(rootPath,"doc/doxygen/xml")
    indexXmlFilePath=os.path.join(TxmlDirPath,"index.xml")
    if (not os.path.exists(indexXmlFilePath)):
        print(f"The file {indexXmlFilePath} does not exist! Please check if the XML document has been generated!")
    try:
        tree=etree.parse(indexXmlFilePath)
    except:
        print(f"File {indexXmlFilePath} parsing error, please check whether you are using doxygen 1.16.1 or a newer version, and delete the generated sutbs.")
    indexxmlRoot=tree.getroot()
 
    open(inheritRecordFilePath,"w").close()
    print("Organising input ...")
    newd = getPySignList(rootPath)
    newd = applyPatch(newd)
    newd = filterNotExist(newd)
    newd = removeDup(newd)
    newd = addMoreInfoTonewd(newd)
    newd = sortnewd(newd)
    
    print("All sorted!\nstart write file...\nThis part of the time may be a bit long, please be patient...")
    for i in newd:
        handlen(i,cv2_stubsPath)
    writeInherit(cv2_stubsPath)
    organise_pyi(cv2_stubsPath)
    addNoknownType(cv2_stubsPath)
    open(os.path.join(cv2_stubsPath,"py.typed"),"w").close()
    os.remove(inheritRecordFilePath)
    print("All stubs have been generated!")

if __name__ == "__main__":
    main()
