dataset = 'SOD';
img1_dir = ['./resultsM/' dataset '/' dataset '/sal0/'];
img2_dir = ['./resultsM/' dataset '/' dataset '/sal1/'];
img3_dir = ['./resultsM/' dataset '/' dataset '/sal2/'];
img4_dir = ['./resultsM/' dataset '/' dataset '/sal3/'];
img5_dir = ['./resultsM/' dataset '/' dataset '/sal4/'];
img6_dir = ['./resultsM/' dataset '/' dataset '/sal5/'];
mean_dir = ['./mean/' dataset '/'];
predictive_dir = ['./predictive/' dataset '/'];
aleatoric_dir = ['./aleatoric/' dataset '/'];
epistemic_dir = ['./epistemic/' dataset '/'];

img_list = dir([img1_dir '*' '.png']);
for i = 1:length(img_list)
    i
    img1 = im2double(imread([img1_dir img_list(i).name]));
    img2 = im2double(imread([img2_dir img_list(i).name]));
    img3 = im2double(imread([img3_dir img_list(i).name]));
    img4 = im2double(imread([img4_dir img_list(i).name]));
    img5 = im2double(imread([img5_dir img_list(i).name]));
    mean = (img1+img2+img3+img4+img5)/5;
    predictive = compute_entropy(mean);
    aleatoric = (compute_entropy(img1)+compute_entropy(img2)+compute_entropy(img3)+compute_entropy(img4)+compute_entropy(img5))/5;
    episteic = predictive-aleatoric;
    predictive = uint8(255*mat2gray(predictive));
    aleatoric = uint8(255*mat2gray(aleatoric));
    episteic = uint8(255*mat2gray(episteic));
    imwrite(predictive,[predictive_dir img_list(i).name]);
    imwrite(aleatoric,[aleatoric_dir img_list(i).name]);
    imwrite(episteic,[epistemic_dir img_list(i).name]);
    mean = uint8(255*mean);
    imwrite(mean,[mean_dir img_list(i).name]);
%     figure,
%     subplot(131)
%     imshow(predictive)
%     subplot(132)
%     imshow(aleatoric)
%     subplot(133)
%     imshow(episteic)
%     close all
end