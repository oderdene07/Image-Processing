clc
clear all
close all
pkg load image

im=imread('C:\Users\User\OneDrive\Desktop\MUIS\Hev_tanilt\2\2b\hubble.jpg');

if(size(im,3)==3)
  img=double(rgb2gray(im));
else
  img=double(im);
end

subplot(1,3,1), imshow(uint8(img));

mask15 = 1/225*ones(size(15,15));
for i = 8:size(img,1)-8
  for j = 8:size(img,2)-8
    out_img15(i,j) = sum(sum(img(i-7:i+7,j-7:j+7).*mask15));
  endfor
endfor
subplot(1,3,2), imshow(uint8(out_img15));

out_img = out_img15;
for i = 1:size(out_img15,1)
  for j = 1:size(out_img15,2)
    if(out_img15(i,j)>80)
      out_img(i,j) = 255;
    else 
      out_img(i,j) = 0;
    endif
  endfor
endfor
subplot(1,3,3), imshow(out_img);
