
%%
clear, clc, 
close all
addpath('Funcs');

%% 1. Parameter Settings
doFrameRemoving = false;
useSP = true;    %You can set useSP = false to use regular grid for speed consideration

SRC = '/home/jing-zhang/jing_file/CVPR2020/scribble/annotations/gt/scribble1/';       %Path of input images       %Path for saving superpixel index image and mean color image
img_dir = '/home/jing-zhang/jing_file/RGB_sal_dataset/train/DUTS/img/';
RES = '/home/jing-zhang/jing_file/CVPR2020/scribble/annotations/gt/scribble1_2/';       %Path for saving saliency maps
srcSuffix = '.png';     %suffix for your input image

if ~exist(RES, 'dir')
    mkdir(RES);
end
%% 2. Saliency Map Calculation
files = dir(fullfile(SRC, strcat('*', srcSuffix)));
% if matlabpool('size') <= 0
%     matlabpool('open', 'local', 8);
% end
for k=1:length(files)
    disp(k);
    srcName = files(k).name;
    noSuffixName = srcName(1:end-length(srcSuffix));
    %% Pre-Processing: Remove Image Frames
    srcImg = imread([img_dir noSuffixName '.jpg']);
    src_scribble = imread([SRC noSuffixName '.png']);
    scribble_fore = src_scribble==1;
    scribble_back = src_scribble==2;
    if doFrameRemoving
        [noFrameImg, frameRecord] = removeframe(srcImg, 'sobel');
        [h, w, chn] = size(noFrameImg);
    else
        noFrameImg = srcImg;
        [h, w, chn] = size(noFrameImg);
        frameRecord = [h, w, 1, h, 1, w];
    end
    
    %% Segment input rgb image into patches (SP/Grid)
    spnumber = 100;     %super-pixel number for current image
    
    if useSP
        [idxImg, adjcMatrix, pixelList] = SLIC_Split(noFrameImg, spnumber);
    else
        [idxImg, adjcMatrix, pixelList] = Grid_Split(noFrameImg, spnumber);        
    end
    %% Get super-pixel properties
    spNum = size(adjcMatrix, 1);
    
    %% foreground superpixels
    fore_sp = idxImg.*scribble_fore;
    fore_sp_index = unique(fore_sp(:));
    fore_sp_index = fore_sp_index(2:end);
    
    
    %% background superpixels
    back_sp = idxImg.*scribble_back;
    back_sp_index = unique(back_sp(:));
    back_sp_index = back_sp_index(2:end);
    
    %% assign foreground to foreground superpixels
    for kk = 1:length(fore_sp_index)
        aa = pixelList{fore_sp_index(kk)};
        src_scribble(aa) = 1;
    end
    
     %% assign foreground to foreground superpixels
    for kk = 1:length(back_sp_index)
        aa = pixelList{back_sp_index(kk)};
        src_scribble(aa) = 2;
    end
    imwrite(src_scribble,[RES noSuffixName '.png']);

end