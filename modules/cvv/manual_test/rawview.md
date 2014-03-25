#Tests for the Rawview*
(The RawviewPanel, STFLEngine, RawviewTable, etc. are tested together)

First off all: Open the match test, click the step button and open the first item in the overview in a tab. Choose the RawView in the combobox at the top.

1. Try essentially every single filter sub query query possible (according to the [reference](http://cvv.mostlynerdless.de/ref/filterquery-ref.html)), the items should be filtered as written in the reference.
2. Try every possible single sort filter query (each time appending "asc", "desc" or ""), the items should be sorted as written in the reference.
3. Try the same with every grouping command
4. Combine some sub queries randomly to at least 5 queries consisting of a least 3 different sub queries. Type them each in, hit <ENTER>, the items should be filter, sorted and grouped correctly.
5. Select several different items, then use the context menu to copy them into your clipboard. Use every possible output mode and check the validity of the output in your clipboard.
6. Type random text into the filter query widget (while hitting enter), the application should not crash.
7. Hit the "Help" button. A web browser should open to the reference.