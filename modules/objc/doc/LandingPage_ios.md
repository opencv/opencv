
This is the documentation for the iOS OpenCV framework.

## Quick Start Tutorial

Follow the steps below to get a simple "Hello World" app running on iOS

* Open Xcode and create a new project from the **File** > **New** > **Project...** menu

![New Project menu](common-create-project.png)

* From template selection dialog select the `iOS` tab and then select `App`

![Template selection dialog](ios-create-app.png)

* Enter a name for the project and set **Interface** to **Storyboard** and **Language** to **Swift**

![App settings dialog](ios-create-app-settings.png)

* Choose a location to save the project

![Choose project location dialog](common-create-app-location.png)

* Create a new folder `Framework`

![Create new folder menu](ios-create-framework-folder.png)

* In a Finder window copy the framework file to the newly created folder. The framework should appear in the project navigator as below. (Note: in this case the framework has extension `.framework` but depending on how you built it or from where you obtained it your framework may have extension `.xcframework`)

![Add framework](ios-add-framework.png)

* In the project navigator select the `ViewController.swift` file

![Select ViewController file](ios-select-viewcontroller.png)

* Add an `import` for the OpenCV framework below the existing import for `UIKit`

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
    Imgproc.putText(img: m, text: "Hello World", org: Point(x: 10, y: 150), fontFace: HersheyFonts.FONT_HERSHEY_DUPLEX, fontScale: 2, color: black)
    let image = m.toUIImage()
    let imageView = UIImageView(image: image)
    self.view.addSubview(imageView)
}
```

* The `ViewController.swift` file should now look like this

![Updated ViewController code](ios-add-opencv-code.png)

* Launch the app

![Launch app button](ios-launch-app.png)

* The app home screen should look like this

![App screenshot](ios-app-screenshot.png)

If that has all worked well then congratulations. If not then feel free to reach out to the OpenCV community for help. Also feel free to contribute fixes and improvements.
