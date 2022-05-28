clc
clear all
close all
pkg load image

im=imread('C:\Users\User\OneDrive\Desktop\MUIS\Hev_tanilt\2\2b\einstein.jpg');

if(size(im,3)==3)
  img=rgb2gray(im);
else
  img=im;
end

subplot(2,3,1), imshow(img);

and_mask = 0*ones(size(img));
and_mask(10:300,150:300) = 1;
subplot(2,3,2), imshow(and_mask);

and_mask_combined = img.*uint8(and_mask);
subplot(2,3,3), imshow(and_mask_combined);

subplot(2,3,4), imshow(img);

or_mask = 255*ones(size(img));
or_mask(10:300,150:300) = 0;
subplot(2,3,5), imshow(or_mask);

or_mask_combined = img+uint8(or_mask);
subplot(2,3,6), imshow(or_mask_combined);

