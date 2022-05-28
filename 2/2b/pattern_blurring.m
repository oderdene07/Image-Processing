clc
clear all
close all
pkg load image

im=imread('C:\Users\User\OneDrive\Desktop\MUIS\Hev_tanilt\2\2b\pattern_blurring.jpg');

if(size(im,3)==3)
  img=double(rgb2gray(im));
else
  img=double(im);
end

out_img3=double(zeros(500,500));
out_img5=double(zeros(500,500));
out_img9=double(zeros(500,500));
out_img15=double(zeros(500,500));
out_img35=double(zeros(500,500));

mask3 = 1/9*ones(size(3,3));
for i = 2:size(img,1)-2
  for j = 2:size(img,2)-2
    out_img3(i,j) = sum(sum(img(i-1:i+1,j-1:j+1).*mask3));
  endfor
endfor

mask5 = 1/25*ones(size(5,5));
for i = 3:size(img,1)-3
  for j = 3:size(img,2)-3
    out_img5(i,j) = sum(sum(img(i-2:i+2,j-2:j+2).*mask5));
  endfor
endfor

mask9 = 1/81*ones(size(9,9));
for i = 5:size(img,1)-5
  for j = 5:size(img,2)-5
    out_img9(i,j) = sum(sum(img(i-4:i+4,j-4:j+4).*mask9));
  endfor
endfor

mask15 = 1/225*ones(size(15,15));
for i = 8:size(img,1)-8
  for j = 8:size(img,2)-8
    out_img15(i,j) = sum(sum(img(i-7:i+7,j-7:j+7).*mask15));
  endfor
endfor

mask35 = 1/1225*ones(size(35,35));
for i = 18:size(img,1)-18
  for j = 18:size(img,2)-18
    out_img35(i,j) = sum(sum(img(i-17:i+17,j-17:j+17).*mask35));
  endfor
endfor

subplot(3,2,1), imshow(uint8(img));
subplot(3,2,2), imshow(uint8(out_img3));
subplot(3,2,3), imshow(uint8(out_img5));
subplot(3,2,4), imshow(uint8(out_img9));
subplot(3,2,5), imshow(uint8(out_img15));
subplot(3,2,6), imshow(uint8(out_img35));
