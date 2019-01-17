input = dlmread('iris.csv');
y = input(:,5);
% Converting output y to matrix where each column has only 1
Y = zeros(3,size(y,1));
for i=1:1:size(y,1)
    if(y(i) == 1)
        Y(:,i) = [1;0;0];
    end
    if(y(i) == 2)
        Y(:,i) = [0;1;0];
    end
    if(y(i) == 3)
        Y(:,i) = [0;0;1];
    end
end
X = input(:,1:4);
% adding a bias parameter to x
X(:,5) = 1;
% Actual X according to notation required where each column depicts a single input.
X = X';
% ramdomizing inputs and output to have training values from all class
randomIndex = randperm(size(X,2));
X = X(:,randomIndex);
Y = Y(:, randomIndex);
% Distributing data as training and test and training the model
distribution= [0.12,0.3,0.5];
lambda = 3000;
for i=1:3
    confusionTest = zeros(3);
    consfusionTrain = zeros(3);
    trainCount = round(distribution(i)*size(X,2));
    Xtrain = X(:,1:trainCount);
    Ytrain = Y(:,1:trainCount);
    Xtest = X(:,trainCount+1: end);
    Ytest = Y(:,trainCount+1: end);
    W = pinv((Xtrain*Xtrain') + lambda)* (Xtrain*Ytrain')
    Ytrainpredicted = W'*Xtrain;
    errorTrain = (sum(((Ytrain - Ytrainpredicted).^2),2)) + lambda*trace(W'*W)
    for j=1:1:size(Ytrainpredicted,2)
        actualClassTrain = find(Ytrain(:,j)==1);
        predictedClassTrain = find(Ytrainpredicted(:,j) == max(Ytrainpredicted(:,j)));
        consfusionTrain(actualClassTrain,predictedClassTrain) = consfusionTrain(actualClassTrain,predictedClassTrain) + 1;
    end
    consfusionTrain
    % Test set error and confusion matrix calclation
    Ytestpredicted = W'*Xtest;
    error = (sum(((Ytest - Ytestpredicted).^2),2)) + lambda*trace(W'*W)
    for j=1:1:size(Ytestpredicted,2)
        actualClass = find(Ytest(:,j)==1);
        predictedClass = find(Ytestpredicted(:,j) == max(Ytestpredicted(:,j)));
        confusionTest(actualClass,predictedClass) = confusionTest(actualClass,predictedClass) + 1;
    end
    confusionTest
end