clc
clear all
close all
pkg load image

im=imread('C:\Users\User\OneDrive\Desktop\MUIS\Hev_tanilt\2\2a\dollar.jpg');

if(size(im,3)==3)
  img=rgb2gray(im);
else
  img=im;
end

subplot(3,3,1), imshow(img);
for i = 1:8
  out_img = bitget(img, i);
  subplot(3,3,i+1);
  imshow(logical(out_img));
endfor