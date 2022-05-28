clc
clear all
close all
pkg load image

im1=imread('C:\Users\User\OneDrive\Desktop\MUIS\Hev_tanilt\2\2b\coffee_1.jpg');
im2=imread('C:\Users\User\OneDrive\Desktop\MUIS\Hev_tanilt\2\2b\coffee_2.jpg');
im3=imread('C:\Users\User\OneDrive\Desktop\MUIS\Hev_tanilt\2\2b\coffee_3.jpg');
im4=imread('C:\Users\User\OneDrive\Desktop\MUIS\Hev_tanilt\2\2b\coffee_4.jpg');

if(size(im1,3)==3)
  img1=rgb2gray(im1);
else
  img1=im1;
end

if(size(im2,3)==3)
  img2=rgb2gray(im2);
else
  img2=im2;
end

if(size(im3,3)==3)
  img3=rgb2gray(im3);
else
  img3=im3;
end

if(size(im4,3)==3)
  img4=rgb2gray(im4);
else
  img4=im4;
end

subplot(4,4,1), imshow(img1);
subplot(4,4,5), imshow(img2);
subplot(4,4,9), imshow(img3);
subplot(4,4,13), imshow(img4);

subplot(4,4,2), imhist(img1);
subplot(4,4,6), imhist(img2);
subplot(4,4,10), imhist(img3);
subplot(4,4,14), imhist(img4);

subplot(4,4,3), imshow(histeq(img1));
subplot(4,4,7), imshow(histeq(img2));
subplot(4,4,11), imshow(histeq(img3));
subplot(4,4,15), imshow(histeq(img4));

subplot(4,4,4), imhist(histeq(img1));
subplot(4,4,8), imhist(histeq(img2));
subplot(4,4,12), imhist(histeq(img3));
subplot(4,4,16), imhist(histeq(img4));
