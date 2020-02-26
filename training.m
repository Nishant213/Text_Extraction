%Loading VGG16 preexisting network
net = vgg16;
layers = net.Layers;


rootFolder = 'handwritten-characters/Train';
categories = {'#','$','&','0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z'};
%Loading the training data
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imds = splitEachLabel(imds, 4000, 'randomize') 
imds.ReadFcn = @readFunctionTrain;
%Removing the last 3 image classification layers 
%(refer vgg16 algorithm for more details)
layers = layers(1:end-3);
%Adding custom image classification layers for our specific input
layers(end+1) = fullyConnectedLayer(1444, 'Name', 'special_2');
layers(end+1) = reluLayer;
layers(end+1) = fullyConnectedLayer(38, 'Name', 'fc8_2 ');
layers(end+1) = softmaxLayer;
layers(end+1) = classificationLayer()


layers 
opts = trainingOptions('sgdm', ...
    'LearnRateSchedule', 'none',...
    'InitialLearnRate', .001,... 
    'MaxEpochs', 1, ...
    'MiniBatchSize', 50, ...
    'ExecutionEnvironment','multi-gpu');
%Stores the trained model in convnet
convnet = trainNetwork(imds, layers, opts);
%Saves the trained model
textnet_2 = convnet;
save textnet_2;


