#Über CVVisual
CVVisual ist eine Debug-Bibliothek für OpenCV, die verschiedene Möglichkeiten der Darstellung von Bildern und Ergebnissen von beispielsweise Filter- und Match-Operationen von OpenCV anbietet.  

##Benutzung: Beispiel
Ist die Bibliothek eingebunden, das CVVISUAL\_DEBUG-Makro definiert und die benötigten Header in den Code eingebunden, kann durch den Aufruf einer CVVisual-Funktion mit den von OpenCV gelieferten Daten als Argumenten das CVV-Hauptfenster geöffnet werden.

Beispielsweise könnte ein Codestück folgendermaßen aussehen:

	//...
	cvv::debugDMatch(src, keypoints1, src, keypoints2, match, CVVISUAL\_LOCATION);

![](../images_ueberblick/MainWindow.PNG)

Die Bilder werden zusammen mit Informationen und Metadaten in der Overview-Tabelle angezeigt.  
Ein Doppelklick darauf öffnet ein Tab, in dem die Bilder und Matches groß angezeigt werden.

![](../images_ueberblick/LineMatchViewTab.PNG)

In dieser Ansicht, genannt *Line Match View* werden die KeyPoints der Matches, d.h. die von OpenCV gelieferten ähnlichen Bildpunkte, durch Linien verbunden. Im Akkordeonmenü kann man beispielsweise deren Farbe änder. `Strg + Mausrad` erlaubt, zu zoomen.

![](../images_ueberblick/LineMatchViewZoomed.PNG)

Die Art der Darstellung kann im `View`-Dropdown-Menü geändert werden; so können die Matches etwa auch als Translationslinien angezeigt werden.

![](../images_ueberblick/TranslationMatchViewTab.PNG)

Zudem gibt es bei Matches auch die Möglichkeit, die Daten in einer Tabelle anzuzeigen, im sogenannten 
*Raw View*. Die Daten können hier über einen Linksklick als JSON oder CSV ins Clipboard kopiert 
werden.

![](../images_ueberblick/RawviewTab.PNG)

Wird `Step` geklickt wird die Ausführung des zu debuggenden Programmes, das beim Aufruf des Hauptfensters angehalten wurde fortgesetzt, bis es auf eine weitere CVVisual-Funktion
stößt:

	//...
	cvv::debugFilter(src, dest, CVVISUAL\_LOCATION, filename);

Das Hauptfenster erscheint erneut, wobei der neuen Datensatz der Tabelle hinzugefügt wird.

![](../images_ueberblick/MainWindowTwoCalls.PNG)

Da es sich hier um eine Filter-Operation handelt, ist die Anzeige im Tab eine andere:

![](../images_ueberblick/DefaultFilterViewTab.PNG)

Auch die möglichen Anzeigen unterscheiden sich von denen für Match-Operationen.
Der *Dual Filter View* erlaubt zum Beispiel zusätzlich, ein Differenzbild der beiden übergebenen anzuzeigen.

![](../images_ueberblick/DualfilterViewDiffImg.PNG)

Nach einem *fast-forward* (`>>`) über die weiteren Schritte des Programms

	//...
	cvv::debugDMatch(src, keypoints1, src, keypoints2, match, CVVISUAL\_LOCATION)
	//...
	cvv::debugFilter(src, dest, CVVISUAL\_LOCATION, filename);
	//...
	cvv::debugFilter(src, dest, CVVISUAL\_LOCATION, filename); 
	//...
	cvv::debugDMatch(src, keypoints1, src, keypoints2, match, CVVISUAL\_LOCATION);
	//...
	cvv::showImage(img, CVVISUAL\_LOCATION);
	//...
	cvv::finalShow();
ergibt sich im Overview folgendes Bild:

![](../images_ueberblick/MainWindowFull.PNG)

Dabei wird durch den letzten Aufruf nur ein einziges Bild zur Anzeige übergeben:

![](../images_ueberblick/SingleImageTab.PNG)

Mithilfe der Textzeile lassen sich durch Kommandos der *Filter Query Language* von CVVisual die Datensätze ordnen, filtern und gruppieren. Hier wurde nach ID gruppiert:

![](../images_ueberblick/OverviewFilterQueryGroupByID.PNG)

Dies funktioniert auch im *Raw View*.

Hinter dem letzten Aufruf einer regulären CVVisual-Funktion muss, wie oben gesehen, `finalShow` aufgerufen werden:

	//...
	cvv::finalShow();
	//...

Es wird ein weiteres Mal das Hauptfenster angezeigt; wird jedoch der nun der einzige verbleibende, der `Close`-Knopf betätigt, wird das Hauptfenster endgültig geschlossen.
  
Dies beschließt die Debug-Sitzung.

[Quelle des zur Demonstration benutzten Bildes.](http://commons.wikimedia.org/wiki/File:PNG-Gradient.png)

