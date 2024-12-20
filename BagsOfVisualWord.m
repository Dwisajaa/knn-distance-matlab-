function image_feats = BagsOfVisualWord(image_paths, vocab)
% Menginisialisasi matriks untuk fitur BOVW
num_images = length(image_paths);
num_vocab = size(vocab, 1);
image_feats = zeros(num_images, num_vocab);

for i = 1:num_images
    % Baca gambar
    img = imread(image_paths{i});
    if size(img, 3) > 1
        img = rgb2gray(img);
    end
    img = imresize(img, [250 250]);

    % Deteksi fitur dan ekstraksi deskriptor
    points = detectHarrisFeatures(img);
    [descriptor, ~] = extractFeatures(img, points);
    
    % Konversi binaryFeatures ke matriks numerik
    if isa(descriptor, 'binaryFeatures')
        descriptor = descriptor.Features; % Ambil data mentah binaryFeatures
    end
    
    % Hitung jarak antara deskriptor dan vocabulary
    distances = pdist2(single(descriptor), vocab);

    % Identifikasi kata terdekat untuk setiap deskriptor
    [~, closest_word] = min(distances, [], 2);

    % Buat histogram dari kata-kata terdekat
    histogram = histcounts(closest_word, 1:(num_vocab + 1));

    % Normalisasi histogram
    image_feats(i, :) = histogram / sum(histogram);
end
end
