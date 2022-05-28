clc
clear all
close all
pkg load image

im=imread('C:\Users\User\OneDrive\Desktop\MUIS\Hev_tanilt\2\2a\kidney.jpg');

if(size(im,3)==3)
  img=rgb2gray(im);
else
  img=im;
end

second=img;
third=img;
[row,col] = size(img);

for x=1:row
  for y=1:col
    if(third(x,y)>150)
      second(x,y)=255;
      third(x,y)=240;
    elseif(third(x,y)>70 && third(x,y)<150)
      second(x,y)=0;
      third(x,y)=50;
    else
      second(x,y)=0;
      third(x,y)=142;
    endif
  endfor
endfor
subplot(1,3,1), imshow(img);
subplot(1,3,2), imshow(second);
subplot(1,3,3), imshow(third);