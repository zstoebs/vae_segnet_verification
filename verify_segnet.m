close all
clear 

load segnet.mat
load Small_ConvNet.mat

nnvNet = CNN.parse(net,'MNIST_conv_classifier');
imgs = random_MNIST_examples();

for label=1:size(imgs,4)

    X = dlarray(imgs(:,:,:,label),'SSCB');
    
    % get prediction
    XPred = sigmoid(forward(segnet, X));
        
    XPred = gather(extractdata(XPred))*255;
    XPred = mean(XPred,3);
    XPred(XPred < 150) = 0;
    
    % brightness attack
    flat = reshape(XPred, [784 1]);
    sorted = sort(flat);
    
    t = sorted(floor(784*0.99));
    lb = flat;
    ub = flat;
    for i=1:784
        if flat(i) >= t
            lb(i) = 0;
            ub(i) = 0.05*flat(i);
        end
    end
    
    % inputs
    lb = reshape(lb, [28 28 1]);
    ub = reshape(ub, [28 28 1]);
    inputZono = ImageZono(lb,ub); % Small_Conv trained on [0,255] inputs
    inputStar = inputZono.toImageStar;
    
    figure;
    subplot(1,3,1)
    imshow(XPred,[0 255])
    title(gca,'Predicted');
    
    subplot(1,3,2);
    imshow(lb,[0 255]);
    title(gca,'Lower Bound');
    
    subplot(1,3,3);
    imshow(ub,[0 255]);
    title(gca,'Upper Bound');
    
    saveas(gca,sprintf('segnet/segnet_example_attack_%d.png',label-1));
    
    % reachability
    fprintf('Approximating zonotope reach set\n')
    [OS_zono,zono_time] = nnvNet.reach(inputZono,'approx-zono');
    fprintf('Approximating star reach set\n')
    [OS_star,star_time] = nnvNet.reach(inputStar,'approx-star');
    fprintf('Approximating polytope reach set\n')
    [OS_absdom,absdom_time] = nnvNet.reach(inputStar,'abs-dom');
    
    % plot
    [lb1,ub1] = OS_zono.getRanges;
    [lb2,ub2] = OS_star.getRanges;
    [lb3,ub3] = OS_absdom.getRanges;
    
    lb1 = reshape(lb1, [10 1]);
    ub1 = reshape(ub1, [10 1]);
    im_center1 = (lb1+ub1)/2;
    err1 = (ub1-lb1)/2;
    x1 = 0:1:9;
    y1 = im_center1;
    
    lb2 = reshape(lb2, [10 1]);
    ub2 = reshape(ub2, [10 1]);
    im_center2 = (lb2+ub2)/2;
    err2 = (ub2-lb2)/2;
    x2 = 0:1:9;
    y2 = im_center2;
    
    lb3 = reshape(lb3, [10 1]);
    ub3 = reshape(ub3, [10 1]);
    im_center3 = (lb3+ub3)/2;
    err3 = (ub3-lb3)/2;
    x3 = 0:1:9;
    y3 = im_center3;
    
    figure;
    subplot(1,3,1);
    e = errorbar(x1,y1,err1);
    e.LineStyle = 'none';
    e.LineWidth = 1;
    e.Color = 'red';
    xlabel('Output','FontSize',11);
    ylabel('Ranges','FontSize',11);
    xlim([0 9]);
    title('Zonotope','FontSize',11);
    xticks([0 5 9]);
    xticklabels({'0', '5', '9'});
    set(gca,'FontSize',10);
    
    subplot(1,3,2);
    e = errorbar(x2,y2,err2);
    e.LineStyle = 'none';
    e.LineWidth = 1;
    e.Color = 'red';
    xlabel('Output','FontSize',11);
    ylabel('Ranges','FontSize',11);
    xlim([0 9]);
    title('ImageStar','FontSize',11);
    xticks([0 5 9]);
    xticklabels({'0', '5', '9'});
    set(gca,'FontSize',10);
    
    subplot(1,3,3);
    e = errorbar(x3,y3,err3);
    e.LineStyle = 'none';
    e.LineWidth = 1;
    e.Color = 'red';
    xlabel('Output','FontSize',11);
    ylabel('Ranges','FontSize',11);
    xlim([0 9]);
    title('Polytope','FontSize',11);
    xticks([0 5 9]);
    xticklabels({'0', '5', '9'});
    set(gca,'FontSize',10);
    
    saveas(gcf,sprintf('segnet/segnet_reach_attack_%d.png',label-1))
 end
