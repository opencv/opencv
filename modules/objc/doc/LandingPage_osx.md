
This is the documentation for the macOS OpenCV framework.

## Quick Start Tutorial

Follow the steps below to get a simple "Hello World" app running on macOS

* Open Xcode and create a new project from the **File** > **New** > **Project...** menu

![New Project menu](common-create-project.png)

* From the template selection dialog select the **macOS** tab and then select **App**

![Template selection dialog](macos-create-app.png)

* Enter a name for the project and set **Interface** to **Storyboard** and **Language** to **Swift**

![App settings dialog](macos-create-app-settings.png)

* Choose a location to save the project

![Choose project location dialog](common-create-app-location.png)

* Create a new folder `Framework`

![Create new folder menu](macos-create-framework-folder.png)

* In a Finder window copy the framework file to the newly created folder. The framework should appear in the project navigator as below. (Note: in this case the framework has extension `.framework` but depending on how you built it or from where you obtained it your framework may have extension `.xcframework`)

![Add framework](macos-add-framework.png)

* Select the project root in the project navigator and on the **General** tab locate the **Frameworks, Libraries and Embedded Content** section

![Frameworks, Libraries and Embedded Content](macos-add-dependencies-1.png)

* In the **Choose frameworks and libraries to add** dialog add the following
    * OpenCL.framework
    * liblapack.tbd
    * libblas.tbd

![Choose frameworks and libraries to add](macos-add-dependencies-2.png)

* The **Frameworks, Libraries and Embedded Content** section should now look like this

![Choose frameworks and libraries to add](macos-add-dependencies-3.png)

* In the project navigator select the `ViewController.swift` file

![Select ViewController file](macos-select-viewcontroller.png)

* Add an `import` for the OpenCV framework below the existing import for `Cocoa`

```swift, copy
import opencv2
```

* Replace the implementation of the `viewDidLoad` function with the following code

```swift, copy
override func viewDidLoad() {
    super.viewDidLoad()
    let white = Scalar(255, 255, 255, 255)
    let black = Scalar(0, 0, 0, 255)
    let m = Mat(rows: 400, cols: Int32(view.frame.width), type: CvType.CV_8UC4, scalar: white)
    Imgproc.putText(img: m, text: "Hello World", org: Point(x: 10, y: 250), fontFace: HersheyFonts.FONT_HERSHEY_DUPLEX, fontScale: 2, color: black)
    let image = m.toNSImage()
    let imageView = NSImageView(frame: NSRect(x: 0, y: 0, width: Int(m.cols()), height: Int(m.rows())))
    imageView.image = image
    self.view.addSubview(imageView)
}
```

* The `ViewController.swift` file should now look like this

![Updated ViewController code](macos-add-opencv-code.png)

* Launch the app

![Launch app button](macos-launch-app.png)

* The app home screen should look like this

![App screenshot](macos-app-screenshot.png)

If that has all worked well then congratulations. If not then feel free to reach out to the OpenCV community for help. Also feel free to contribute fixes and improvements.
