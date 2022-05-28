clc
clear all
close all
pkg load image

im=imread('C:\Users\User\OneDrive\Desktop\MUIS\Hev_tanilt\2\2b\embedded_square.jpg');

if(size(im,3)==3)
  img=rgb2gray(im);
else
  img=im;
end

mask = 1/9*ones(3,3);
out_img=double(img);
for i = 2:size(img,1)-2
  for j = 2:size(img,2)-2
    out_img(i,j) = sum(sum(double(img(i-1:i+1,j-1:j+1)).*mask));
  endfor
endfor

subplot(1,3,1), imshow(img);
subplot(1,3,2), imshow(histeq(img));
subplot(1,3,3), imshow(out_img);
