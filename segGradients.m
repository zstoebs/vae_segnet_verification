function grad = segGradients(net, x)
xPred = sigmoid(forward(net, x));
loss = ELBOloss(x, xPred, 1, 0);
grad = dlgradient(loss, net.Learnables);
end