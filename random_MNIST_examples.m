function img_array = random_MNIST_examples()

testImagesFile = 't10k-images-idx3-ubyte';
testLabelsFile = 't10k-labels-idx1-ubyte';

XTest = processImagesMNIST(testImagesFile);
YTest = processLabelsMNIST(testLabelsFile);

imgs = zeros(size(XTest,1),size(XTest,2),size(XTest,3),10);

for i=0:9
    idx = iRandomIdxOfClass(YTest,i);
    X = XTest(:,:,:,idx);
    imgs(:,:,:,i+1) = X;
end
img_array = dlarray(imgs);
end