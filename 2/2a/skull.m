clc
clear all
close all
pkg load image

im=imread('C:\Users\User\OneDrive\Desktop\MUIS\Hev_tanilt\2\2a\skull.jpg');

if(size(im,3)==3)
  img=rgb2gray(im);
else
  img=im;
end

subplot(2,4,1), imshow(img);

for i = 1:6
  out_img = (2^i)*round(img/(2^i));
  subplot(2,4,1+i), imshow(out_img);
endfor

out_img = img;
for i=1:size(img,1)
  for j=1:size(img,2)
      if(img(i,j) > 150)
        out_img(i,j)=255;
      else
        out_img(i,j)=0;
      endif
  endfor
endfor
subplot(2,4,8), imshow(out_img);
