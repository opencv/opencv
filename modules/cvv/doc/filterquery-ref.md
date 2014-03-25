#Filter query language

The filter query language is the query language used in the overview and the raw match view to simply the task of filtering, sorting and grouping data sets in a table UI.
The following is a description of the simple syntax and the supported commands.

Just type `#` into the search field to see some supported commands, using the suggestions feature (it's inspired by the awesome z shell).

##Syntax
A query consist basically of many subqueries starting with a `#`:

`[raw filter subquery] #[subquery 1] [...] #[subquery n]`

The optional first part of the query doesn't start with a `#`, it's short for `#raw [...]`.


There three different types of subqueries:

###Sort query
A sort query has the following structure:

`sort by [sort subquery 1], [...], [sort subquery n]`

A sort subquery consist of a sort command (aka "the feature by which you want to sort the table") and a sort order:
- `[command]`: equivalent to `[command] asc`
- `[command] asc`: sorts in ascending order
- `[command] desc`: sorts in descending order

(The sort command is typically a single word.)

For your interest: The `[subquery n]` has higher priority than the `[subquery n+1]`. 

###Group query
A group query has the following structure:

`group by [command 1], [...], [command n]`

A group command is a single word declaring the feature you want to group the data sets in the table by.
The group header consist of the `n` items.

For your interest: The raw view currently doesn't support group queries.

###Filter query
A filter query is the basic type of query, allowing you to filter the data sets by several criterias.

It has the following structure:

`#[filter command] [argument]`

It also supports several arguments for one filter command (via the comma seperated filters feature):

`#[cs filter command] [argument 1], [...], [argument n]`


####Range filter query
A range filter query uses basically a comma seperated filter command with two arguments, allowing you to
filter for a range of elements (`[lower bound]` <= `element` <= `[upper bound]`).

It has the following structure:

`#[filter command] [lower bound], [upper bound]`



##Overview
The following commands are supported in the overview:

feauture/command | sorting supported | grouping supported | filtering supported | description
-----------------|:-----------------:|:------------------:|:--------------------|:---------------------
id               | yes               | yes                | yes, also range     |
raw              | yes               | yes                | only basic filter   | alias for description
description      | yes               | yes                | only basic filter   |
image_count      | yes               | yes                | yes, also range     | number of images
function         | yes               | yes                | yes                 | calling function
file             | yes               | yes                | yes                 | inheriting file
line             | yes               | yes                | yes, also range     |
type             | yes               | yes                | yes                 | call type
            

##Rawview
The following command are supported in the raw (match) view:

feauture/command | numeric type | description/property
-----------------|:-------------|:---------------------------------------------
match_distance   | float        | match distance
img_idx          | integer      | match img idx 
query_idx        | integer      | match query idx
train_idx        | integer      | match train idx
x_1              | float        | x coordinate of the "left" key point
y_1              | float        | y coordinate of the "left" key point 
size_1           | float        | size of the "left" key point
angle_1          | float        | angle of the "left" key point
response_1       | float        | response (or strength) of the "left" key point
octave_1         | integer      | octave of the "left" key point
x_2              | float        | x coordinate of the "right" key point
y_2              | float        | y coordinate of the "right" key point 
size_2           | float        | size of the "right" key point
angle_2          | float        | angle of the "right" key point
response_2       | float        | response (or strength) of the "right" key point
octave_2         | integer      | octave of the "right" key point



All commands support range filtering, sorting and grouping and therefore only the used numeric type
(integer or float) is given.

See the opencv documentation for more information about the features.
