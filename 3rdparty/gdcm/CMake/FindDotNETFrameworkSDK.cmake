# - Find .NET Software Development Kit
# This module finds an installed .NET Software Development Kit.  It sets the following variables:
#
#  Copyright (c) 2006-2011 Mathieu Malaterre <mathieu.malaterre@gmail.com>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

# Note:
# .NET Framework SDK Version 1.1
# http://www.microsoft.com/downloads/details.aspx?FamilyID=9b3a2ca6-3647-4070-9f41-a333c6b9181d&displaylang=en
# .NET Framework 2.0 Software Development Kit (SDK) (x86)
# http://www.microsoft.com/downloads/details.aspx?FamilyID=fe6f2099-b7b4-4f47-a244-c96d69c35dec&displaylang=en
# Microsoft .NET Framework 3.5
# http://www.microsoft.com/downloads/details.aspx?familyid=333325FD-AE52-4E35-B531-508D977D32A6&displaylang=en

# Comparison Between C++ and C#
# http://msdn.microsoft.com/en-us/library/yyaad03b(VS.71).aspx

# http://www.akadia.com/services/dotnet_assemblies.html

# Visual C# Language Concepts
# Building from the Command Line
# http://msdn.microsoft.com/en-us/library/1700bbwd(VS.71).aspx

# FIXME, path are hardcoded.
# http://www.walkernews.net/2007/07/30/how-to-verify-dot-net-framework-version/

find_program(CSC_v1_EXECUTABLE csc
 $ENV{windir}/Microsoft.NET/Framework/v1.1.4322/
)
find_program(CSC_v2_EXECUTABLE csc
 $ENV{windir}/Microsoft.NET/Framework/v2.0.50727/
)
find_program(CSC_v3_EXECUTABLE csc
 $ENV{windir}/Microsoft.NET/Framework/v3.5/
)
find_program(CSC_v4_EXECUTABLE csc
 $ENV{windir}/Microsoft.NET/Framework/v4.0.30319/
)

get_filename_component(current_list_path ${CMAKE_CURRENT_LIST_FILE} PATH)
set(DotNETFrameworkSDK_USE_FILE ${current_list_path}/UseDotNETFrameworkSDK.cmake)

mark_as_advanced(
  CSC_v1_EXECUTABLE
  CSC_v2_EXECUTABLE
  CSC_v3_EXECUTABLE
  CSC_v4_EXECUTABLE
)
