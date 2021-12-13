% File to do statistical analysis


% Select original image and output
covid_image = imread('./input_images/Lena.jpg');
covid_image_output = imread('./outputs/cancer-normal.png');


% Convert image to double for calculation
covid_image_d = double(covid_image);
covid_image_output_d = double(covid_image_output);


% Preprocess image for calculate PSNR
ref = im2single(covid_image);
dlref = dlarray(ref);
ref1 = im2single(covid_image_output);
dlref = dlarray(ref1);


% Calculate Error
ERROR = (covid_image_d - covid_image_output_d).^2;
MSE_output =  sum(ERROR(:))/length(covid_image_output(:));
[peaksnr,snr] = psnr(ref1,ref);


% Display output
MSE_output
peaksnr