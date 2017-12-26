Understanding Features {#tutorial_py_features_meaning}
======================

Goal
----

In this chapter, we will just try to understand what are features, why are they important, why
corners are important etc.

Explanation
-----------

Most of you will have played the jigsaw puzzle games. You get a lot of small pieces of an image,
where you need to assemble them correctly to form a big real image. **The question is, how you do
it?** What about the projecting the same theory to a computer program so that computer can play
jigsaw puzzles? If the computer can play jigsaw puzzles, why can't we give a lot of real-life images
of a good natural scenery to computer and tell it to stitch all those images to a big single image?
If the computer can stitch several natural images to one, what about giving a lot of pictures of a
building or any structure and tell computer to create a 3D model out of it?

Well, the questions and imaginations continue. But it all depends on the most basic question: How do
you play jigsaw puzzles? How do you arrange lots of scrambled image pieces into a big single image?
How can you stitch a lot of natural images to a single image?

The answer is, we are looking for specific patterns or specific features which are unique, can
be easily tracked and can be easily compared. If we go for a definition of such a feature, we may
find it difficult to express it in words, but we know what they are. If someone asks you to point
out one good feature which can be compared across several images, you can point out one. That is
why even small children can simply play these games. We search for these features in an image,
find them, look for the same features in other images and align them. That's it. (In jigsaw puzzle,
we look more into continuity of different images). All these abilities are present in us inherently.

So our one basic question expands to more in number, but becomes more specific. **What are these
features?**. (The answer should be understandable also to a computer.)

It is difficult to say how humans find these features. This is already programmed in our brain.
But if we look deep into some pictures and search for different patterns, we will find something
interesting. For example, take below image:

![image](images/feature_building.jpg)

The image is very simple. At the top of image, six small image patches are given. Question for you is to
find the exact location of these patches in the original image. How many correct results can you
find?

A and B are flat surfaces and they are spread over a lot of area. It is difficult to find the exact
location of these patches.

C and D are much more simple. They are edges of the building. You can find an approximate location,
but exact location is still difficult. This is because the pattern is same everywhere along the edge.
At the edge, however, it is different. An edge is therefore better feature compared to flat area, but
not good enough (It is good in jigsaw puzzle for comparing continuity of edges).

Finally, E and F are some corners of the building. And they can be easily found. Because at the
corners, wherever you move this patch, it will look different. So they can be considered as good
features. So now we move into simpler (and widely used image) for better understanding.

![image](images/feature_simple.png)

Just like above, the blue patch is flat area and difficult to find and track. Wherever you move the blue
patch it looks the same. The black patch has an edge. If you move it in vertical direction (i.e.
along the gradient) it changes. Moved along the edge (parallel to edge), it looks the same. And for
red patch, it is a corner. Wherever you move the patch, it looks different, means it is unique. So
basically, corners are considered to be good features in an image. (Not just corners, in some cases
blobs are considered good features).

So now we answered our question, "what are these features?". But next question arises. How do we
find them? Or how do we find the corners?. We answered that in an intuitive way, i.e., look for
the regions in images which have maximum variation when moved (by a small amount) in all regions
around it. This would be projected into computer language in coming chapters. So finding these image
features is called **Feature Detection**.

We found the features in the images. Once you have found it, you should be able to find the same
in the other images. How is this done? We take a region around the feature, we explain it in our own
words, like "upper part is blue sky, lower part is region from a building, on that building there is
glass etc" and you search for the same area in the other images. Basically, you are describing the
feature. Similarly, a computer also should describe the region around the feature so that it can
find it in other images. So called description is called **Feature Description**. Once you have the
features and its description, you can find same features in all images and align them, stitch them together
or do whatever you want.

So in this module, we are looking to different algorithms in OpenCV to find features, describe them,
match them etc.

Additional Resources
--------------------

Exercises
---------
