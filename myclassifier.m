function S = myclassifier(im)
    S = [];
    load Mdl;
    
    F = FeatureExtraction(im);
    num_feats = size(F,1);
    
    feats = [];
    for i = 1:num_feats
        feats(end+1,:) = F(i,:,:);
    end
    
    guesses = predict(Mdl, feats);
    
    if num_feats == 3
        S(1) = 0;
        S(2) = str2num(guesses{1});
        S(3) = str2num(guesses{2});
        S(4) = str2num(guesses{3});
    else
        S(1) = str2num(guesses{1});
        S(2) = str2num(guesses{2});
        S(3) = str2num(guesses{3});
        S(4) = str2num(guesses{4});
    end
end