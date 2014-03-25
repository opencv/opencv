#Tests for the Overview*
(The OverviewPanel, OverviewTable, etc. are tested together)

First off all: For all of the following tests, call the test_all (or the filters) test and click on the fast forward (">>") button for all but the first two tests.

1. Click the "Step" button. A new item should appear in the list. Enter a filter query and hit <ENTER>, then hit "Step" button several times, the list should be updated according to your filter query each time.
2. Hit the fast forward button. The program should take some time and then present all items to you. The fast forward and the step button should be hidden now. Restart the test and hit the fast forward button after you typed a filter query and hit <ENTER>. The items should be sorted, filtered and grouped accordingly.
3. Hit the close button. The program should close without further GUI changes (it might take some time).
3. Try essentially every single filter sub query query possible (according to the [reference](http://cvv.mostlynerdless.de/ref/filterquery-ref.html)), the items should be filtered as written in the reference.
4. Try every possible single sort filter query (each time appending "asc", "desc" or ""), the items should be sorted as written in the reference.
5. Try the same with every grouping command
6. Combine some sub queries randomly to at least 5 queries consisting of a least 3 different sub queries. Type them each in, hit <ENTER>, the items should be filter, sorted and grouped correctly.
7. Use the context menu on at least 2 randomly chosen items to remove them and then try the 6. test.
8. Resize the main window and the use the slider at the bottom of the overview. The images in the table should be resized appropriately.
9. Use the other two context menu items on at least 3 randomly chosen items, they should do the right thing according to their titles.
10. Double click on at least 4 items. They should each be opened in a new tab in the main window.
11. Type random text into the filter query widget (while hitting enter), the application should not crash.
12. Try 11. while testing 6. to 10.. 
13. Hit the "Help" button. A web browser should open to the reference.