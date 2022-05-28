clc
clear all
close all
pkg load image

imagefiles = dir('C:/Users/User/OneDrive/Desktop/Hev_tanilt/lab1/*.jpg');      
nfiles = length(imagefiles);
avg_img = 0;
resized = []

for i = 1:nfiles
   currentfile = strcat('C:/Users/User/OneDrive/Desktop/Hev_tanilt/lab1/', imagefiles(i).name)
   currentimage = imread(currentfile);
   figure, imshow(currentimage);
   edited = rgb2gray(currentimage);
   resized{i} = imresize(edited, [400,500]);
   avg_img = avg_img + resized{i} / nfiles;
end

for i=1:nfiles
   result = avg_img - resized{i};
   figure, imshow(result);
   imwrite(result, strcat('C:/Users/User/OneDrive/Desktop/Hev_tanilt/lab1/Results/', imagefiles(i).name));
end
