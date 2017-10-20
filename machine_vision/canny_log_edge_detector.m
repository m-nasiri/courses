%#---------------------------------------------------------------------------##
% @file    Canny.py
% @author  Majid Nasiri
% @ID      95340651
% @version V1.0.0
% @date    2017-April-22
% @brief   Canny Edge Detector
%#---------------------------------------------------------------------------##
clc
clear
close all

im = imread('..\results\66.jpg');
img = rgb2gray(im);

%%%%%%%%%%%%%%%%%%%%%% canny fix thre
outputVideo_canny_fixThr = VideoWriter('..\results\canny_ther=0.4 sigma=0.1-0.05-4','MPEG-4');
outputVideo_canny_fixThr.FrameRate = 10;
open(outputVideo_canny_fixThr);
thresh = 0.4;
for i=0.1:0.05:4
    sigma = i;
    img_edge = edge(img, 'canny', thresh, sigma);
    img_ind = gray2ind(img_edge*255);
    writeVideo(outputVideo_canny_fixThr, img_ind);
end
close(outputVideo_canny_fixThr);

%%%%%%%%%%%%%%%%%%%%%% canny fix sigma
outputVideo_canny_fixSigma = VideoWriter('..\results\canny_sigma=1.41 thre=0-0.01-0.99','MPEG-4');
outputVideo_canny_fixSigma.FrameRate = 10;
open(outputVideo_canny_fixSigma);
sigma = 1.41;
for i=0:0.01:0.99
    thresh = i;
    img_edge = edge(img, 'canny', thresh, sigma);
    img_ind = gray2ind(img_edge*255);
    writeVideo(outputVideo_canny_fixSigma, img_ind);
end
close(outputVideo_canny_fixSigma);

%%%%%%%%%%%%%%%%%%%%%% log fix thre
outputVideo_log_fixThr = VideoWriter('..\results\log_ther=0.4 sigma=0.01-0.02-1','MPEG-4');
outputVideo_log_fixThr.FrameRate = 10;
open(outputVideo_log_fixThr);
thresh = 0.4;
for i=0.01:0.02:1
    sigma = i;
    img_edge = edge(img, 'log', thresh, sigma);
    img_ind = gray2ind(img_edge*255);
    writeVideo(outputVideo_log_fixThr, img_ind);
end
close(outputVideo_log_fixThr);


%%%%%%%%%%%%%%%%%%%%%% log fix sigma
outputVideo_log_fixSigma = VideoWriter('..\results\log_sigma=0.5 thre=0-0.01-0.99','MPEG-4');
outputVideo_log_fixSigma.FrameRate = 10;
open(outputVideo_log_fixSigma);
sigma = 0.5;
for i=0:0.01:0.99
    thresh = i;
    img_edge = edge(img, 'log', thresh, sigma);
    img_ind = gray2ind(img_edge*255);
    writeVideo(outputVideo_log_fixSigma, img_ind);
end
close(outputVideo_log_fixSigma);


% end of file


