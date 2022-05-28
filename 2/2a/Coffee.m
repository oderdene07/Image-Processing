clc
clear all
close all
pkg load image

im=imread('C:\Users\User\OneDrive\Desktop\MUIS\Hev_tanilt\2\2a\coffee.jpg');

if(size(im,3)==3)
  img=rgb2gray(im);
else
  img=im;
end

subplot(2,2,1), imshow(img);

[row,col] = size(img);
first=imadjust(img);
subplot(2,2,2), imshow(first);

out_img = zeros(row,col);
for i=1:size(first,1)
  for j=1:size(first,2)
      if(first(i,j) > 150)
        out_img(i,j)=255;
      else
        out_img(i,j)=0;
      endif
  endfor
endfor

subplot(2,2,3),imshow(out_img);