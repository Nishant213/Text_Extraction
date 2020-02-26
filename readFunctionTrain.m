
function I = readFunctionTrain(filename)
I = imread(filename);
%Converting image to fit the VGG16 input parameters (224*224)
I = imresize(I, [224 224]);
%Converting input images from grayscale to colour format
I = cat(3, I, I, I);
