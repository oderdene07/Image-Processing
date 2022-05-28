clc
clear all
close all
pkg load image

im=imread('C:\Users\User\OneDrive\Desktop\MUIS\Hev_tanilt\2\2a\flower.jpg');

if(size(im,3)==3)
  img=rgb2gray(im);
else
  img=im;
end

first=imresize(img, [128 128], 'nearest');
subplot(2,3,1), imshow(first);

second=imresize(img, [64 64], 'nearest');
subplot(2,3,2), imshow(second);

third=imresize(img, [32 32], 'nearest');
subplot(2,3,3), imshow(third);


first_b=imresize(img, [128 128], 'bilinear');
subplot(2,3,4), imshow(first_b);

second_b=imresize(img, [64 64], 'bilinear');
subplot(2,3,5), imshow(second_b);

third_b=imresize(img, [32 32], 'bilinear');
subplot(2,3,6), imshow(third_b);
