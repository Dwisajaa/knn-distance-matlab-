function vocab = construct_vocabulary(image_paths, vocab_size)
features = [];

for i = 1:length(image_paths)
    img = imread(image_paths{i});
    if size(img, 3) > 1
        img = rgb2gray(img);
    end
    img = imresize(img, [250 250]);
    points = detectHarrisFeatures(img);
    
    [descriptor, ~] = extractFeatures(img, points);
    
    % Konversi binaryFeatures ke matriks numerik
    if isa(descriptor, 'binaryFeatures')
        descriptor = descriptor.Features; % Ambil data mentah binaryFeatures
    end
    
    % Tambahkan descriptor ke array features
    features = [features; single(descriptor)];
end

% Lakukan clustering dengan k-means
[~, vocab] = kmeans(features, vocab_size, 'MaxIter', 900, 'Display', 'final');
end
