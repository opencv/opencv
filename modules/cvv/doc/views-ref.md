#Views
##General information:
Most views offer an `ImageInformation` collapsable in their accordion menus.  
The zoom can be found here.  
`Ctrl`+`Mouse wheel` is also zoom; `Ctrl`+`Shift`+`Mouse wheel` is a slower zoom.  
If the zoom is deeper than 60%, the image's pixels will be overlaid with their channel values; usually, the order is BGR[+alpha] from the top.    
  
##Single Image View:
Associated with the `debugSingleImage()` function.  
Shows one single image with no features other than `Image Information`.  

##Filter Views:
Associated with the `debugFilter()` function.  

###DefaultFilterView:
Shows two images with only the basic features of `ImageInformation`, synchronized zoom and `Histogram`.  

###DualFilterView:
Shows the two images given to the CVVisual function and _Result Image_ inbetween 
which represents the result of a filter that was applied to the others via the `Filter selection` collapsable,
like a difference image between the two.  
  
###SingleFilterView:
Allows to apply filters to the images it shows via the `Select a filter` collapsable.  

##Match Views:
Associated with the `debugDMatch()` function.  

###PointMatchView: 
Interprets the translation of matches as depth value. 

###LineMatchView:
Connects matching key points in the images with lines.  

###Rawview:
Shows in a table data of the matches.  
The table entries can be filtered, sorted and grouped by using commands from CVVisual's [filter query language](filterquery-ref.html) in the text box.  

###TranslationMatchView
Shows the distance between a keypoint in one image to its match in the other as an arrow or line in one image.
