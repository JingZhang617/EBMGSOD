
%%
clear, clc, 
close all
addpath('Funcs');

%% 1. Parameter Settings
doFrameRemoving = false;
useSP = true;           %You can set useSP = false to use regular grid for speed consideration
doMAEEval = true;       %Evaluate MAE measure after saliency map calculation
doPRCEval = true;       %Evaluate PR Curves after saliency map calculation


SRC = ['/home/jingzhang/jing_files/RGBD_Dataset/BBSNet/test/DUT-RGBD/depth/'];       %Path of input images
srcSuffix = '.png';     %suffix for your input image

%% 2. Saliency Map Calculation
files = dir(fullfile(SRC, strcat('*', srcSuffix)));
mean_complexity = zeros(length(files),1);
for k=1:length(files)
    disp(k);
    srcName = files(k).name;
    noSuffixName = srcName(1:end-length(srcSuffix));
    %% Pre-Processing: Remove Image Frames
    srcImg = imread(fullfile(SRC, srcName));
    if (size(srcImg,3) == 1)
        srcImg = cat(3,srcImg,srcImg,srcImg);
    end
    if doFrameRemoving
        [noFrameImg, frameRecord] = removeframe(srcImg, 'sobel');
        [h, w, chn] = size(noFrameImg);
    else
        noFrameImg = srcImg;
        [h, w, chn] = size(noFrameImg);
        frameRecord = [h, w, 1, h, 1, w];
    end
    
    %% Segment input rgb image into patches (SP/Grid)
    pixNumInSP = 900;                           %pixels in each superpixel
    spnumber = round( h * w / pixNumInSP );     %super-pixel number for current image
    
    if useSP
        [idxImg, adjcMatrix, pixelList] = SLIC_Split(noFrameImg, spnumber);
    else
        [idxImg, adjcMatrix, pixelList] = Grid_Split(noFrameImg, spnumber);        
    end
    %% Get super-pixel properties
    spNum = size(adjcMatrix, 1);
    meanRgbCol = GetMeanColor(noFrameImg, pixelList);
    meanLabCol = colorspace('Lab<-', double(meanRgbCol)/255);
    meanPos = GetNormedMeanPos(pixelList, h, w);
    bdIds = GetBndPatchIds(idxImg);
    colDistM = GetDistanceMatrix(meanLabCol);
    posDistM = GetDistanceMatrix(meanPos);
    [clipVal, geoSigma, neiSigma] = EstimateDynamicParas(adjcMatrix, colDistM);
    
%     %% Saliency Objectness
    [bgProb, bdCon, bgWeight] = EstimateBgProb(colDistM, adjcMatrix, bdIds, clipVal, geoSigma);
    wCtr = CalWeightedContrast(colDistM, posDistM, bgProb);
    optwCtr = SaliencyOptimization(adjcMatrix, bdIds, colDistM, neiSigma, bgWeight, wCtr);
    a = -optwCtr.*log(optwCtr);
    mean_complexity(k) = mean(a);
end
