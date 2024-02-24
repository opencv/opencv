<!-- SPDX-License-Identifier: BSD-3-Clause -->
<!-- Copyright (c) Contributors to the OpenEXR Project -->

[![License](https://img.shields.io/github/license/AcademySoftwareFoundation/openexr)](LICENSE.md)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2799/badge)](https://bestpractices.coreinfrastructure.org/projects/2799)
[![Build Status](https://github.com/AcademySoftwareFoundation/openexr/workflows/CI/badge.svg)](https://github.com/AcademySoftwareFoundation/openexr/actions?query=workflow%3ACI)
[![Analysis Status](https://github.com/AcademySoftwareFoundation/openexr/workflows/Analysis/badge.svg)](https://github.com/AcademySoftwareFoundation/openexr/actions?query=workflow%3AAnalysis)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=AcademySoftwareFoundation_openexr&metric=alert_status)](https://sonarcloud.io/dashboard?id=AcademySoftwareFoundation_openexr)

# OpenEXR

<img align="right" src="docs/technical/images/windowExample1.png">

OpenEXR provides the specification and reference implementation of the
EXR file format, the professional-grade image storage format of the
motion picture industry.

The purpose of EXR format is to accurately and efficiently represent
high-dynamic-range scene-linear image data and associated metadata,
with strong support for multi-part, multi-channel use cases.

OpenEXR is widely used in host application software where accuracy is
critical, such as photorealistic rendering, texture access, image
compositing, deep compositing, and DI.

## OpenEXR Project Mission

The goal of the OpenEXR project is to keep the EXR format reliable and
modern and to maintain its place as the preferred image format for
entertainment content creation. 

Major revisions are infrequent, and new features will be carefully
weighed against increased complexity.  The principal priorities of the
project are:

* Robustness, reliability, security
* Backwards compatibility, data longevity
* Performance - read/write/compression/decompression time
* Simplicity, ease of use, maintainability
* Wide adoption, multi-platform support - Linux, Windows, macOS, and others

OpenEXR is intended solely for 2D data. It is not appropriate for
storage of volumetric data, cached or lit 3D scenes, or more complex
3D data such as light fields.

The goals of the Imath project are simplicity, ease of use,
correctness and verifiability, and breadth of adoption. Imath is not
intended to be a comprehensive linear algebra or numerical analysis
package.

## Project Governance

OpenEXR is a project of the [Academy Software
Foundation](https://www.aswf.io). See the project's [governance
policies](GOVERNANCE.md), [contribution guidelines](CONTRIBUTING.md), and [code of conduct](CODE_OF_CONDUCT)
for more information.

# Quick Start

See the [technical documentation](https://openexr.readthedocs.io) for
complete details, but to get started, the "hello, world" `.exr` writer program is:

    #include <ImfRgbaFile.h>
    #include <ImfArray.h>
    #include <iostream>
    
    int
    main()
    {
        try {
            int width =  10;
            int height = 10;
            
            Imf::Array2D<Imf::Rgba> pixels(width, height);
            for (int y=0; y<height; y++)
                for (int x=0; x<width; x++)
                    pixels[y][x] = Imf::Rgba(0, x / (width-1.0f), y / (height-1.0f));
        
            Imf::RgbaOutputFile file ("hello.exr", width, height, Imf::WRITE_RGBA);
            file.setFrameBuffer (&pixels[0][0], 1, width);
            file.writePixels (height);
        } catch (const std::exception &e) {
            std::cerr << "Unable to read image file hello.exr:" << e.what() << std::endl;
            return 1;
        }
        return 0;
    }

The `CMakeLists.txt` to build:

    cmake_minimum_required(VERSION 3.10)
    project(exrwriter)
    find_package(OpenEXR REQUIRED)
    
    add_executable(${PROJECT_NAME} writer.cpp)
    target_link_libraries(${PROJECT_NAME} OpenEXR::OpenEXR)

To build:

    $ mkdir _build
    $ cmake -S . -B _build
    $ make --build _build

For more details, see [The OpenEXR
API](https://openexr.readthedocs.io/en/latest/API.html#the-openexr-api).

# Community

* **Ask a question:**

  - Email: openexr-dev@lists.aswf.io

  - Slack: [academysoftwarefdn#openexr](https://academysoftwarefdn.slack.com/archives/CMLRW4N73)

* **Attend a meeting:**

  - Technical Steering Committee meetings are open to the
    public, fortnightly on Thursdays, 1:30pm Pacific Time.

  - Calendar: https://lists.aswf.io/g/openexr-dev/calendar

* **Report a bug:**

  - Submit an Issue: https://github.com/AcademySoftwareFoundation/openexr/issues

* **Report a security vulnerability:**

  - Email to security@openexr.com

* **Contribute a Fix, Feature, or Improvement:**

  - Read the [Contribution Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md)

  - Sign the [Contributor License
    Agreement](https://contributor.easycla.lfx.linuxfoundation.org/#/cla/project/2e8710cb-e379-4116-a9ba-964f83618cc5/user/564e571e-12d7-4857-abd4-898939accdd7)
  
  - Submit a Pull Request: https://github.com/AcademySoftwareFoundation/openexr/pulls

# Resources

- Website: http://www.openexr.com
- Technical documentation: https://openexr.readthedocs.io
- Porting help: [OpenEXR/Imath Version 2.x to 3.x Porting Guide](https://openexr.readthedocs.io/en/latest/PortingGuide.html)
- Reference images: https://github.com/AcademySoftwareFoundation/openexr-images
- Security policy: [SECURITY.md](SECURITY.md)
- Release notes: [CHANGES.md](CHANGES.md)
- Contributors: [CONTRIBUTORS.md](CONTRIBUTORS.md)  

# License

OpenEXR is licensed under the [BSD-3-Clause license](LICENSE.md).


---

![aswf](/ASWF/images/aswf.png)
