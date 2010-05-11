#!/usr/bin/octave -q

addpath(getenv("OCTAVEPATH"));

highgui;
cv;

I=cvLoadImage("frame.jpg");
a=cv2im(I);
I2=im2cv(a, CV_8UC(1));

imshow(cv2im(I));
imshow(cv2im(I2));

a=rand(3,3,3);
b=mat2cv(a,CV_64FC(1));
c=cv2mat(b);
assert(all(a==c));

a=eye(3);
b=mat2cv(a,CV_64FC(1));
c=cv2mat(b);
assert(all(a==c));

assert(all(cv2mat(mat2cv(eye(3),6))==eye(3)));

I=cvLoadImage("frame.jpg");
a=cv2im(I);
I2=cvCloneImage(I);
cvSobel(I,I2,2,2);
imshow(cv2im(I2));

imshow(a);
