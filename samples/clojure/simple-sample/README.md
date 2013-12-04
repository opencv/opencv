# Introduction to Clojure Development

As of OpenCV 2.4.4, OpenCV supports desktop Java development using
nearly the same interface as for Android development. [Clojure][1] is
a recent LISP dialect hosted on the Java Virtual Machine and it offers
an high level of interoperability with the underlying hosting JVM.

This guide will help you to create your first Clojure (CLJ)
application using OpenCV.

For detailed instruction on installing OpenCV with desktop Java
support refer to the [corresponding tutorial][2].

# Install Leiningen

Once you installed OpenCV with desktop java support the only other
requirement is to install [Leiningeng][3] which allows you to manage
the entire life cycle of your CLJ projects.

The available [installation guide][4] is very easy to be followed:

1. [Download the script][5]
2. Place it on your `$PATH` (cf. `~/bin` is a good choice if it is on
   your `path`.)
3. Set the script to be executable. (i.e. `chmod 755 ~/bin/lein`).

If you work on Windows, follow [this instruction][6]

# Create a Leiningen project

The first step of using OpenCV from CLJ is to create a new CLJ project
by using the `lein new` command from the terminal.

```bash
lein new simple-sample
Generating a project called simple-sample based on the 'default' template.
To see other templates (app, lein plugin, etc), try `lein help new`.
```

The above command (or task in `lein` parlance) creates the following
`simple-sample` directory layout:

```bash
tree simple-sample/
simple-sample/
├── LICENSE
├── README.md
├── doc
│   └── intro.md
├── project.clj
├── resources
├── src
│   └── simple_sample
│       └── core.clj
└── test
    └── simple_sample
        └── core_test.clj

6 directories, 6 files
```

## Locally install the OpenCV libs

To be able to interact in CLJ with OpenCV through its java interface,
you first need to locally install the java interface lib and the
corresponding native shared lib created by the OpenCV installation
with the desktop java support.

Assuming you built OpenCV in the `build` directory (see OpenCV
installation guide), you find the `opencv-247.jar` lib under the
`build/bin` directory and the corresponding native shared
`libopencv_java247.dylib` lib under the `build/lib` directory.

> NOTE 1: For this tutorial I'm using the Mac OS X Operating System
> which use `.dylib` as the file extension for shared libs. If you use
> a GNU/Linux distribution the shared lib file extension is `.so`.

### Copy the libs

Start by copying both the above libs in a new directory of your choice
(e.g. `~/opt/opencv`).

```bash
cd
mkdir -p opt/opencv
cd opt/opencv
cp ~/Developer/opencv/build/bin/opencv-247.jar .
```

The shared `libopencv_java247.dylib` lib has to be copied in a
directories structure that reflects your Operating System
(e.g. `macosx`, `linux`, `windows`, etc.) and the corresponding
architecture (e.g. `x86_64`, `x86`, `arm`, `sparc`, etc.).

```bash
mkdir -p native/macosx/x86_64
cp ~/Developer/opencv/build/lib/libopencv_java247.dylib native/macosx/x86_64/
```

You now need to package the shared native lib in a `jar` file as follows:

```bash
jar -cMf opencv-native.jar native
```

Your final directories layout should be the following

```bash
tree
.
├── native
│   └── macosx
│       └── x86_64
│           └── libopencv_java247.dylib
├── opencv-247.jar
└── opencv-native.jar

3 directories, 3 files
```

Start by locally install the `libopencv_java247.dylib` shared
lib. Before to be able to install it in such a way that it's visible
to the CLJ project, we first need to copy it somewhere and package it
in a `jar` file.

```bash
cd simple-sample
mkdir -p opencv/native/macosx/x86_64
cd opencv
cp ~/Developer/opencv/build/lib/libopencv_java247.dylib native/macosx/x86_64/
jar -cMf opencv-native.jar native
tree
.
├── native
│   └── macosx
│       └── x86_64
│           └── libopencv_java247.dylib
└── opencv-native.jar

3 directories, 2 files
```




## Dry test

Now `cd` in the `simple-sample` directory and issue the following
`lein` task:

```bash
cd simple-sample
lein repl
...
...
nREPL server started on port 50907 on host 127.0.0.1
REPL-y 0.3.0
Clojure 1.5.1
    Docs: (doc function-name-here)
          (find-doc "part-of-name-here")
  Source: (source function-name-here)
 Javadoc: (javadoc java-object-or-class-here)
    Exit: Control+D or (exit) or (quit)
 Results: Stored in vars *1, *2, *3, an exception in *e

user=>
```

The very first time you run a `lein` task it will take sometimes to
download all the required dependencies before activating the
interactive REPL (Read Eval Print Loop). You can immediately interact
with the REPL by issuing any CLJ expression to be evaluated.

```clj
user=> (+ 41 1)
42
user=> (println "Hello, Clojure!")
Hello, Clojure!
nil
user=> (defn foo [] (str "bar"))
#'user/foo
user=> (foo)
"bar"
user=>
user=> (exit)
Bye for now!
```

## Add OpenCV dependencies to the project

Now open the `project.clj` file with your preferred editor to see its
content.

```clj
(defproject simple-sample "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.5.1"]])
```

As you see the only current dependency of the `simple-sample` project
is the `"1.5.1"` release of the Clojure Programming Language itself.

To be able to use the OpenCV Java interface and the corresponding
dynamic lib created when you install OpenCV with desktop java support,
you need to add them to the `:dependencies` section of the project.

```clj
(defproject simple-sample "0.1.0-SNAPSHOT"
  ...
  :dependencies [[org.clojure/clojure "1.5.1"]
                 [local/opencv "2.4.7"] ; new line
                 [local/opencv-native "2.4.7"]]) ; new line
```

The dependencies section of the project is represented as a vector of
vector where each vector element is generally composed of two values:

* the groupID/artifactId of the lib: e.g. `local/opencv`
* the lib version release: e.g. `"2.4.7"

We used `local` as `groupId` of both libs because we need to install
them locally on your machine and are not going to be downloaded from
the net.

## Package the OpenCV libs

Even if we already added the above OpenCV dependencies to the project,

interoperate with OpenCV through CLJ we need
Each dependencies is represented as a CLJ vector (e.g. `[local/opencv "2.4.7"]`). The first element of the vector is



a `lein` project is a kind of key/value map. Each key is a
`keyword` (e.g. `:description`, `:url`, etc) and each value could be
any CLJ expression: a string (e.g. `"FIXME: write description"`, a map
`{:name "Eclipse Public License" :url
"http://www.eclipse.org/legal/epl-v10.html"}` and even a vector of
vector (e.g. `[[org.clojure/clojure "1.5.1"]]`.


To exit the REPL just First let's run The generated
`project.clj` is used to configure the project and its
dependencies. By default it only add the `clojure` dependency to the
project.


The most important asset in the `simple-directory` is the
`project.clj` file used to configure the project dependencies. If you `cd


## Usage

FIXME

## License

Copyright © 2013 FIXME

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
