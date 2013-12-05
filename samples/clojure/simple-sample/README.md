# Introduction to OpenCV Development with Clojure 

As of OpenCV 2.4.4, OpenCV supports desktop Java development using
nearly the same interface as for Android development.

[Clojure][1] is a contemporary LISP dialect hosted by the Java Virtual
Machine and it offers a complete interoperability with the underlying
JVM. This means that we should be able to even use the Clojure REPL
(Read Eval Print Loop) as and interactive programmable interface to
the underlying OpenCV engine.

This guide will help in setting up a very basic environment for
starting to learn and experiment with OpenCV in a rich fully
programmable REPL.

## Preamble

For detailed instruction on installing OpenCV with desktop Java
support refer to the [corresponding tutorial][2].

If you are in hurry, here is a minimum quick start to install OpenCV
on Mac OS X:

```bash
cd ~/
mkdir opt
git clone https://github.com/Itseez/opencv.git
cd opencv
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS=OFF ..
...
...
make -j8
# optional
# make install
```

## Install Leiningen

Once you installed OpenCV with desktop java support the only other
requirement is to install [Leiningeng][3] which allows you to manage
the entire life cycle of your CLJ projects.

The available [installation guide][4] is very easy to be followed:

1. [Download the script][5]
2. Place it on your `$PATH` (cf. `~/bin` is a good choice if it is on
   your `path`.)
3. Set the script to be executable. (i.e. `chmod 755 ~/bin/lein`).

If you work on Windows, follow [this instruction][6]

You now have both the OpenCV library and a fully installed basic
Clojure environment. What is now needed is to configure the Clojure
environment to interact with the OpenCV library.

## Install the local-repo Leiningen plugin

The set of commands (tasks in Leiningen parlance) natively supported
by Leiningen can be very easily extended by various plugins. One of
them is the [lein-localrepo][7] plugin which allows to install any jar
lib as an artifact in the local maven repository of your machine
(typically in the `~/.m2/repository` directory of your username).

We're going to use this `lein` plugin to add to the local maven
repository the opencv components needed by Java and Clojure to use the
opencv lib.

Generally speaking, if you want to use a plugin on project base only,
it can be added directly to a CLJ project created by `lein`.

Instead, when you want a plugin to be available to any CLJ project in
your username space, you can add it to the `profiles.clj` in the
`~/.lein/` directory.

I think the `lein-localrepo` plugin will be useful to me in other CLJ
projects where I need to call native libs wrapped by a Java
interface. So I decide to make it available to any CLJ project:

```bash
mkdir ~/.lein
```

Create a file named `profiles.clj` in the `~/.lein` directory and
copy into it the following content:

```clj
{:user {:plugins [[lein-localrepo "0.5.2"]]}}
```

Here we're saying that the version release `"0.5.2"` of the
`lein-localrepo` plugin will be available to the `:user` profile for
any CLJ project created by `lein`.

You do not need to do anything else to install the plugin because it
will be automatically downloaded from a remote repository the very
first time you issue any `lein` task.

## Install the java specific libs as local repository

If you followed the standard documentation for installing OpenCV on
your computer, you should find the following two libs under the
directory where you built OpenCV:

* the `build/bin/opencv-247.jar` java lib
* the `build/lib/libopencv_java247.dylib` native lib (or `.so` in you
  built OpenCV a GNU/Linux OS)

They are the only opencv libs needed by the JVM to interact with
OpenCV.

### Take apart the needed opencv libs 

Create a new directory to store in the above two libs. Start by
copying into it the `opencv-247.jar` lib.

```bash
cd ~/opt
mkdir clj-opencv
cd clj-opencv
cp ~/opt/opencv/build/bin/opencv-247.jar .
```

First lib done.

Now, to be able to add the `libopencv_java247.dylib` shared native lib
to the local maven repository, we first need to package it as a jar
file.

The native lib has to be copied into a directories layout which mimics
the names of your operating system and architecture. I'm using a Mac
OS X with a X86 64 bit architecture. So my layout will be the
following:

```bash
mkdir -p native/macosx/x86_64
```

Copy into the `x86_64` directory the `libopencv_java247.dylib` lib.

```bash
cp ~/opt/opencv/build/lib/libopencv_java247.dylib native/macosx/x86_64/
```

> NOTE 1: On a GNU/Linux
> 
> ```bash
> mkdir -p native/linux/x86_64
> cp ~/opt/opencv/build/lib/libopencv_java247.so native/linux/x86_64/
> ```

#### Package the native lib as a jar

Next you need to package the native lib in a jar file by using the
`jar` command to create a new jar file from a directory.

```bash
jar -cMf opencv-native-247.jar native
```

> NOTE 2: the `M` option instructs the `jar` command to not create a
> MANIFEST file for the artifact.

Your directories layout should look like the following:

```bash
tree
.
├── native
│   └── macosx
│       └── x86_64
│           └── libopencv_java247.dylib
├── opencv-247.jar
└── opencv-native-247.jar

3 directories, 3 files
```

### Locally install the jars

We are now ready to add the two jars as artifacts to the local maven
repository with the help of the `lein-localrepo` plugin.

```bash
lein localrepo install opencv-247.jar opencv/opencv 2.4.7
```

Here the `localrepo install` task creates the `2.4.7.` release of the
`opencv/opencv` maven artifact from the `opencv-247.jar` lib and then
installs it into the local maven repository. The `opencv/opencv`
artifact will then be available to any maven compliant project
(Leiningen is internally based on maven).

Do the same thing with the native lib previously wrapped in a new jar
file.

```bash
lein localrepo install opencv-native-247.jar opencv/opencv-native 2.4.7
```

Note that the groupId, `opencv`, of the two artifacts is the same. We
are now ready to create a new CLJ project to start interacting with
OpenCV.

## Create a project

Create a new CLJ project by using the `lein new` task from the
terminal.

```bash
# cd in the directory where you work with your development projects (e.g. ~/devel)
lein new simple-sample
Generating a project called simple-sample based on the 'default' template.
To see other templates (app, lein plugin, etc), try `lein help new`.
```

The above task creates the following `simple-sample` directories layout:

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

We need do add the two `opencv` artifacts as dependencies of the newly
created project. Open the `project.clj` and modify its dependencies
section as follows:

```bash
(defproject simple-sample "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.5.1"]
                 [opencv/opencv "2.4.7"] ; added line
                 [opencv/opencv-native "2.4.7"]]) ;added line
```

> NOTE 3: The Clojure Programming Language is an artifact too. This is
> why Clojure is called an hosted language.

To verify that everything went right issue the `lein deps` task. The
very first time you run a `lein` task it will take sometimes to
download all the required dependencies before executing the task
itself.


```bash
cd simple-sample
lein deps
```

The `deps` task reads from the `project.clj` and the
`~/.lein/profiles.clj` files all the dependencies of the
`simple-sample` project and verifies if they have already been cached
in the local maven repository. If the task returns without messages
about not being able to retrieve the two new artifacts, your
installation is correct, otherwise go back and check that you did
everything right.

## REPLing with OpenCV

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

You can immediately interact with the REPL by issuing any CLJ
expression to be evaluated.

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
```

When ran from the a the home directory of a lein based project, the
`lein repl` task automatically loads all the project dependencies. So
we should be able to interact with the OpenCV libs by just importing
the corresponding java packages (namespaces in Clojure parlance).

```clj
user=> (org.opencv.core.Point. 0 0)
#<Point {0.0, 0.0}>
```

Here we created a two dimensions opencv `Point` instance. Even if all
the java packages included within the java interface to OpenCV are
immediately available as CLJ namespeces from the CLJ REPL, it's very
annoying to prefix the `Point.` instance constructors with the fully
qualified namespace.

Fortunately CLJ offer a very easy way to overcome this annoyance by
just importing the java package in which the `Point` class lives.

```clj
user=> (import 'org.opencv.core.Point)
org.opencv.core.Point
user=> (def p1 (Point. 0 0))
#'user/p1
user=> p1
#<Point {0.0, 0.0}>
user=> (def p2 (Point. 100 100))
#'user/p2
```

We can even inspect the class of an instance and verify if the value
of a symbol is an instance of a `Point` java class.

```clj
user=> (class p1)
org.opencv.core.Point
user=> (instance? org.opencv.core.Point p1)
true
```

If we now want to use the opencv `Rect` class to create a rectangle,
we ha to fully qualify it's constructor even if it leaves in the same
`org.opencv.core` namespace.

```clj
user=> (org.opencv.core.Rect. p1 p2)
#<Rect {0, 0, 100x100}>
```

Again, the CLJ importing facilities is very handy. You can decide to
import in your REPL session from a java package all the symbols you're
going to use more frequently.

```clj
user=> (import '[org.opencv.core Point Rect Size])
org.opencv.core.Size
user=> (def r1 (Rect. p1 p2))
#'user/r1
user=> r1
#<Rect {0, 0, 100x100}>
user=> (class r1)
org.opencv.core.Rect
user=> (instance? org.opencv.core.Rect r1)
true
user=> (Size. 100 100)
#<Size 100x100>
user=> (def sq-100 (Size. 100 100))
#'user/sq-100
user=> (class sq-100)
org.opencv.core.Size
user=> (instance? org.opencv.core.Size sq-100)
true
```

Obviously you can call methods on instances as well.

```clj
user=> (.area r1)
10000.0
user=> (.area sq-100)
10000.0
```

Or modify the value of a  member field.

```clj
user=> (set! (.x p1) 10)
10
user=> p1
#<Point {10.0, 0.0}>
user=> (set! (.width sq-100) 10)
10
user=> (set! (.height sq-100) 10)
10
user=> (.area sq-100)
100.0
```

To exit the REPL type `(exit)`, `ctr-D` or `(quit)` at the REPL
prompt.

```clj
user=> (exit)
Bye for now!
```

## Next Steps

I recommend any newbie with Clojure to read the
[Clojure Java Interop chapter][8] to get all you need to know to
interoperate with any plain java lib that has not been wrapped in
Clojure to make it usable in a more idiomatic and functional way
within Clojure (e.g. [Vision][9]).

### License ###

Copyright © 2013 Giacomo (Mimmo) Cosenza aka Magomimmo

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.

[1]: http://clojure.org/
[2]: http://docs.opencv.org/2.4.4-beta/doc/tutorials/introduction/desktop_java/java_dev_intro.html
[3]: https://github.com/technomancy/leiningen
[4]: https://github.com/technomancy/leiningen#installation
[5]: https://raw.github.com/technomancy/leiningen/stable/bin/lein
[6]: https://github.com/technomancy/leiningen#windows
[7]: https://github.com/kumarshantanu/lein-localrepo
[8]: http://clojure.org/java_interop
[9]: http://nakkaya.com/vision.html

