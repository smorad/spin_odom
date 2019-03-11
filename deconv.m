close all

%theta = atan2(464 - 262, 142 - 277)

f = imread('/home/smorad/spin_odom/images/speer_data/20190130_170303_4856/frame1715_19003.jpg');
%psf = fspecial('motion', 100, rad2deg(-0.1024));
psf = fspecial('motion', 219, rad2deg(-0.1024));

figure()
image(f)
hold on
title('before')

plot([141 462], [228 261])
slope = atan2(228- 261, 462 - 141)
hold off

figure()
%image(deconvwnr(f, psf, 0.1))
wnr = deconvlucy(f, psf);


image(wnr)
title('after')

% ft = real(ifft2(fft2(im2bw(f))));
% figure()
% imshow(ft, [])
% ft2 = real(ifft2(fft2(im2bw(wnr))));
% figure()
% imshow(ft2, [])
% figure()
% im2bw(wnr)

% gradient along angle
% want lines to be parallel along axis of rotation
% curvy rock on right looks like curvy rock on left

% edge detection and then convolution across axis of rotation


% how paper works
% find edges
% predict sharp edge by finding min/max pixel values along the edge

% what we should do
% find edges
% march along smear direction
% max is first pixel that is less than 90% of the prev pixel
% min is the first pixel greater than 90% of the prev pixel
