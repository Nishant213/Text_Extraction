%Importing the previously trained model
load('convnet.mat');
s = convnet;
rootFolder = 'handwritten-characters/Validation';
categories = {'#','$','&','0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z'};
%Loading the test data
testDS = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
testDS = splitEachLabel(testDS, 100, 'randomize') 
testDS.ReadFcn = @readFunctionTrain;
%Using model to extract characters from test images
[labels,err_test] = classify(s, testDS, 'MiniBatchSize', 10);
%Find and print average accuracy
confMat = confusionmat(testDS.Labels, labels);
confMat = confMat./sum(confMat,2);
mean(diag(confMat))