trainImagesFile = 'train-images-idx3-ubyte';
testImagesFile = 't10k-images-idx3-ubyte';
testLabelsFile = 't10k-labels-idx1-ubyte';

XTrain = processImagesMNIST(trainImagesFile);
numTrainImages = size(XTrain,4);
XTest = processImagesMNIST(testImagesFile);
YTest = processLabelsMNIST(testLabelsFile);

imageSize = [28 28 1];

numClasses = 2;
encoderDepth = 1;
seglayers = segnetLayers(imageSize,numClasses,encoderDepth);
seglayers = removeLayers(seglayers,{'pixelLabels','softmax'});
seglayers = replaceLayer(seglayers,'inputImage',imageInputLayer(imageSize,'Name','input_encoder','Normalization','none'));

figure
plot(seglayers)

segnet = dlnetwork(seglayers);

executionEnvironment = "auto";

numEpochs = 20;
miniBatchSize = 512;
lr = 1e-3;
numIterations = floor(numTrainImages/miniBatchSize);
iteration = 0;

avgGradients = [];
avgGradientsSquared = [];

for epoch = 1:numEpochs
    tic;
    for i = 1:numIterations
        iteration = iteration + 1;
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        XBatch = XTrain(:,:,:,idx);
        XBatch = dlarray(single(XBatch), 'SSCB');
        
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            XBatch = gpuArray(XBatch);           
        end 
            
        grad = dlfeval(...
            @segGradients, segnet, XBatch);
        
        [segnet.Learnables, avgGradients, avgGradientsSquared] = ...
            adamupdate(segnet.Learnables, ...
                grad, avgGradients, avgGradientsSquared, iteration, lr);
    end
    elapsedTime = toc;
    
    xPred = sigmoid(forward(segnet, XTest));
    elbo = ELBOloss(XTest, xPred, 1, 0);
    disp("Epoch : "+epoch+" Test ELBO loss = "+gather(extractdata(elbo))+...
        ". Time taken for epoch = "+ elapsedTime + "s")    
end

visualizeSegReconstruction(XTest,YTest,segnet)

save segnet.mat segnet