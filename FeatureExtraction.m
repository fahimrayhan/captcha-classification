function F = FeatureExtraction(I)
    gray = rgb2gray(I);
    
    % FFT Parameters
    D0 = 30;
    n = 2;
    notch_width = 3;
    
    [rows, cols] = size(gray);
    
    % FFT Denoise
    F_fft = fft2(gray); 
    F_shifted = fftshift(F_fft);
    [X, Y] = meshgrid(1:cols, 1:rows);
    centerX = cols/2;
    centerY = rows/2;
    D = sqrt((X - centerX).^2 + (Y - centerY).^2);
    
    % Butterworth filter
    H_butter = 1 ./ (1 + (D/D0).^(2*n));
    
    % Notch filter
    H_notch = ones(rows, cols);
    nc1 = max(1, round(centerX - notch_width));
    nc2 = min(cols, round(centerX + notch_width));
    nr1 = max(1, round(centerY - 2));
    nr2 = min(rows, round(centerY + 2));
    H_notch(:, nc1:nc2) = 0;
    H_notch(nr1:nr2, :) = 1;
    H_notch = imgaussfilt(H_notch, 2);
    
    % Apply filter
    H_combined = H_butter .* H_notch;
    denoised = real(ifft2(ifftshift(F_shifted .* H_combined)));
    
    I = imbinarize(uint8(denoised));
    I = imerode(I, strel('disk', 4));
    I = bwareaopen(I, 500);
    I = imdilate(I, strel('disk', 3));
    
    connected_components = bwconncomp(I, 4);
    all_idx = vertcat(connected_components.PixelIdxList{:});
    [conn_rows, conn_cols] = ind2sub(size(I), all_idx);
    
    leftmost_col = min(conn_cols);
    rightmost_col = max(conn_cols);
    topmost_row = min(conn_rows);
    bottommost_row = max(conn_rows);
    
    full_width = rightmost_col - leftmost_col + 1;
    h = bottommost_row - topmost_row + 1;
    
    if full_width > 264
        num_digits = 4;
    else
        num_digits = 3;
    end
    
    F = [];
    
    if num_digits == 3
        w = full_width / 3;
        F(1,:,:) = Feats(imcrop(I, [leftmost_col topmost_row w h]));
        F(2,:,:) = Feats(imcrop(I, [leftmost_col+w topmost_row w h]));
        F(3,:,:) = Feats(imcrop(I, [leftmost_col+w*2 topmost_row w h]));
    else
        w = full_width / 4;
        F(1,:,:) = Feats(imcrop(I, [leftmost_col topmost_row w h]));
        F(2,:,:) = Feats(imcrop(I, [leftmost_col+w topmost_row w h]));
        F(3,:,:) = Feats(imcrop(I, [leftmost_col+w*2 topmost_row w h]));
        F(4,:,:) = Feats(imcrop(I, [leftmost_col+w*3 topmost_row w h]));
    end
end

function F = Feats(S)
    F = [];
    
    if sum(S(:)) > 0
        H = fft2(double(S));
        H = abs(H) .* log(abs(H)+eps);
        F = [F, sum(H(:))];
    else
        F = [F, zeros(1, 8)];
    end
    
    shape_feats = ShapeFeats(S);
    F = [F, shape_feats];
    
    euler_num = bweuler(S);
    F = [F, euler_num];
    
    props = regionprops('Table', S, 'Extent', 'ConvexArea', 'Area');
    if height(props) > 0
        [~, idx] = max(props.Area);
        convexity = props.Area(idx) / (props.ConvexArea(idx) + eps);
        F = [F, props.Extent(idx), convexity];
    else
        F = [F, 0, 0];
    end
    
    se = strel('line', 15, 90);
    vertical_strength = sum(imopen(S, se), 'all') / sum(S(:));
    F = [F, vertical_strength];
    
    R = RadialZoning(S, 5);
    F = [F, R];
    
    props = regionprops(S, 'Area', 'Orientation');
    if ~isempty(props)
        [~, idx] = max([props.Area]);
        theta = props(idx).Orientation;
        Srot = imrotate(S, theta, 'nearest', 'crop');
    else
        Srot = S;
    end
    
    Z = ZoningFeats(Srot, 4, 32);
    F = [F, Z];
    
    angles = [0 45 90 135];
    for a = angles
        se = strel('line', 15, a);
        strength = sum(imopen(S, se), 'all') / sum(S(:));
        F = [F, strength];
    end
    
    filled = imfill(S, 'holes');
    holes = filled & ~S;
    props = regionprops(holes, 'Area', 'Centroid');
    if ~isempty(props)
        [~, idx] = max([props.Area]);
        cy = props(idx).Centroid(2) / size(S,1);
    else
        cy = 0;
    end
    F = [F, cy];
    
    [h,w] = size(S);
    upper = S(1:floor(h/2), :);
    lower = S(floor(h/2)+1:end, :);
    F = [F, sum(upper(:))/sum(S(:)), sum(lower(:))/sum(S(:))];
    
    LBP = extractLBPFeatures(S, 'NumNeighbors', 8, 'Radius', 1, 'Upright', true);
    F = [F, LBP];
    
    digit_size = 32;
    Sresized = imresize(S, [digit_size digit_size]);
    cellSize = [8 8];
    H = extractHOGFeatures(Sresized, 'CellSize', cellSize);
    F = [F, H];
end

function F = ShapeFeats(S)
    fts = {'Circularity', 'Area', 'Centroid', 'Orientation', 'Solidity', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength'};
    Ft = regionprops('Table', S, fts{:});
    if height(Ft) > 0
        [~, idx] = max(Ft.Area);
        F = [Ft(idx).Variables];
    else
        F = zeros(1, length(fts) + 1);
    end
end

function R = RadialZoning(S, numRings)
    [h,w] = size(S);
    [X,Y] = meshgrid(1:w, 1:h);
    cx = mean(X(S));
    cy = mean(Y(S));
    D = sqrt((X-cx).^2 + (Y-cy).^2);
    D = D / max(D(:));
    R = zeros(1, numRings);
    for i = 1:numRings
        mask = (D > (i-1)/numRings) & (D <= i/numRings);
        R(i) = sum(S(mask)) / (sum(mask(:)) + eps);
    end
end

function Z = ZoningFeats(S, gridSize, outSize)
    S = logical(S);
    Z = zeros(1, gridSize^2);
    if sum(S(:)) == 0
        return;
    end
    props = regionprops(S, 'Area', 'BoundingBox');
    if isempty(props)
        return;
    end
    [~, idx] = max([props.Area]);
    bb = props(idx).BoundingBox;
    x1 = max(1, floor(bb(1)));
    y1 = max(1, floor(bb(2)));
    x2 = min(size(S,2), ceil(bb(1) + bb(3) - 1));
    y2 = min(size(S,1), ceil(bb(2) + bb(4) - 1));
    if x2 <= x1 || y2 <= y1
        return;
    end
    digit = S(y1:y2, x1:x2);
    digit = imresize(digit, [outSize outSize], 'nearest');
    zoneSize = floor(outSize / gridSize);
    k = 1;
    for r = 1:gridSize
        for c = 1:gridSize
            r1 = (r-1)*zoneSize + 1;
            r2 = r*zoneSize;
            c1 = (c-1)*zoneSize + 1;
            c2 = c*zoneSize;
            r2 = min(r2, outSize);
            c2 = min(c2, outSize);
            block = digit(r1:r2, c1:c2);
            Z(k) = sum(block(:)) / numel(block);
            k = k + 1;
        end
    end
end

function H = hu_moments(I)
    [i, j, v] = find(I);
    if isrow(i)
        i = i(:); j = j(:); v = v(:);
    end
    mu00 = sum(v);
    ctr = sum([i,j] .* v, 1) ./ mu00;
    i = i - ctr(1);
    j = j - ctr(2);
    eta = nan(4,4);
    for p = 0:3
        for q = max(0, 2-p):3-p
            eta(p+1, q+1) = sum(i.^p .* j.^q .* v) / mu00^(1+(p+q)/2);
        end
    end
    H(1) = eta(3,1) + eta(1,3);
    H(2) = (eta(3,1) - eta(1,3))^2 + 4*eta(2,2)^2;
    H(3) = (eta(4,1) - 3*eta(2,3))^2 + (3*eta(3,2) - eta(1,4))^2;
    H(4) = (eta(4,1) + eta(2,3))^2 + (eta(3,2) + eta(1,4))^2;
    H(5) = (eta(4,1) - 3*eta(2,3)) * (eta(4,1) + eta(2,3)) * ((eta(4,1) + eta(2,3))^2 - 3*(eta(3,2) + eta(1,4))^2) + ...
           (3*eta(3,2) - eta(1,4)) * (eta(3,2) + eta(1,4)) * (3*(eta(4,1) + eta(2,3))^2 - (eta(3,2) + eta(1,4))^2);
    H(6) = (eta(3,1) - eta(1,3)) * ((eta(4,1) + eta(2,3))^2 - (eta(3,2) + eta(1,4))^2) + ...
           4*eta(2,2) * (eta(4,1) + eta(2,3)) * (eta(3,2) + eta(1,4));
    H(7) = (3*eta(3,2) - eta(1,4)) * (eta(4,1) + eta(2,3)) * ((eta(4,1) + eta(2,3))^2 - 3*(eta(3,2) + eta(1,4))^2) - ...
           (eta(4,1) - 3*eta(2,3)) * (eta(3,2) + eta(1,4)) * (3*(eta(4,1) + eta(2,3))^2 - (eta(3,2) + eta(1,4))^2);
    H(8) = eta(2,2) * ((eta(4,1) + eta(2,3))^2 - (eta(3,2) + eta(1,4))^2) - (eta(3,1) - eta(1,3)) * (eta(4,1) + eta(2,3)) * (eta(3,2) + eta(1,4));
end