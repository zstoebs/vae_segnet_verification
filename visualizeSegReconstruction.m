function visualizeSegReconstruction(XTest,YTest, net)
f = figure;
figure(f)
title("Example ground truth image vs. reconstructed image")
for i = 1:2
    for c=0:9
        idx = iRandomIdxOfClass(YTest,c);
        X = XTest(:,:,:,idx);
        
        XPred = sigmoid(forward(net, X));
        
        X = gather(extractdata(X));
        XPred = gather(extractdata(XPred));

        comparison = [X, ones(size(X,1),1), mean(XPred,3)];
        subplot(4,5,(i-1)*10+c+1), imshow(comparison,[]),
    end
end
end