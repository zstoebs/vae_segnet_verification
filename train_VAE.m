% Adapted from: 
% https://www.mathworks.com/help/deeplearning/ug/train-a-variational-autoencoder-vae-to-generate-images.html

trainImagesFile = 'train-images-idx3-ubyte';
testImagesFile = 't10k-images-idx3-ubyte';
testLabelsFile = 't10k-labels-idx1-ubyte';

XTrain = processImagesMNIST(trainImagesFile);
numTrainImages = size(XTrain,4);
XTest = processImagesMNIST(testImagesFile);
YTest = processLabelsMNIST(testLabelsFile);

latentDim = 20;
imageSize = [28 28 1];

encoderLG = layerGraph([
    imageInputLayer(imageSize,'Name','input_encoder','Normalization','none')
    convolution2dLayer(3, 32, 'Padding','same', 'Stride', 2, 'Name', 'encoder_conv1')
    reluLayer('Name','encoder_relu1')
    convolution2dLayer(3, 64, 'Padding','same', 'Stride', 2, 'Name', 'encoder_conv2')
    reluLayer('Name','encoder_relu2')
    fullyConnectedLayer(2 * latentDim, 'Name', 'fc_encoder')
    ]);

decoderLG = layerGraph([
    imageInputLayer([1 1 latentDim],'Name','i','Normalization','none')
    transposedConv2dLayer(7, 64, 'Cropping', 'same', 'Stride', 7, 'Name', 'decoder_transpose1')
    reluLayer('Name','decoder_relu1')
    transposedConv2dLayer(3, 64, 'Cropping', 'same', 'Stride', 2, 'Name', 'decoder_transpose2')
    reluLayer('Name','decoder_relu2')
    transposedConv2dLayer(3, 32, 'Cropping', 'same', 'Stride', 2, 'Name', 'decoder_transpose3')
    reluLayer('Name','decoder_relu3')
    transposedConv2dLayer(3, 1, 'Cropping', 'same', 'Name', 'decoder_transpose4')
    ]);

encoderNet = dlnetwork(encoderLG);
decoderNet = dlnetwork(decoderLG);

executionEnvironment = "auto";

numEpochs = 50;
miniBatchSize = 512;
lr = 1e-3;
numIterations = floor(numTrainImages/miniBatchSize);
iteration = 0;

avgGradientsEncoder = [];
avgGradientsSquaredEncoder = [];
avgGradientsDecoder = [];
avgGradientsSquaredDecoder = [];

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
            
        [infGrad, genGrad] = dlfeval(...
            @modelGradients, encoderNet, decoderNet, XBatch);
        
        [decoderNet.Learnables, avgGradientsDecoder, avgGradientsSquaredDecoder] = ...
            adamupdate(decoderNet.Learnables, ...
                genGrad, avgGradientsDecoder, avgGradientsSquaredDecoder, iteration, lr);
        [encoderNet.Learnables, avgGradientsEncoder, avgGradientsSquaredEncoder] = ...
            adamupdate(encoderNet.Learnables, ...
                infGrad, avgGradientsEncoder, avgGradientsSquaredEncoder, iteration, lr);
    end
    elapsedTime = toc;
    
    [z, zMean, zLogvar] = sampling(encoderNet, XTest);
    xPred = sigmoid(forward(decoderNet, z));
    elbo = ELBOloss(XTest, xPred, zMean, zLogvar);
    disp("Epoch : "+epoch+" Test ELBO loss = "+gather(extractdata(elbo))+...
        ". Time taken for epoch = "+ elapsedTime + "s")    
end

visualizeReconstruction(XTest, YTest, encoderNet, decoderNet)
visualizeLatentSpace(XTest, YTest, encoderNet)
generate(decoderNet, latentDim)

figure
plot(encoderLG)

figure
plot(decoderLG)

save encoder.mat encoderNet
save decoder.mat decoderNet