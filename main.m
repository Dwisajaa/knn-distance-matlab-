clc;
clear;

% Set parameters
data_path = 'C:/deteksiknn/';  % Path to your dataset
categories = {'n02093647-Bedlington_terrier', 'n02093754-Border_terrier'};
num_train_per_cat = 100;
num_valid_per_cat = 40;
vocab_size = 128;

% Get image paths and labels
[train_image_paths, valid_image_paths, train_labels, valid_labels] = ...
    get_image_paths(data_path, categories, num_train_per_cat, num_valid_per_cat);

% Construct vocabulary
vocab = construct_vocabulary(train_image_paths, vocab_size);

% Extract features
train_feats = BagsOfVisualWord(train_image_paths, vocab);
valid_feats = BagsOfVisualWord(valid_image_paths, vocab);

% Train models with different hyperparameters
k_values = [3, 5, 7];  % Different k values for KNN
distance_metrics = {'euclidean', 'cosine', 'cityblock'};  % Different distance metrics
accuracies = zeros(length(k_values), length(distance_metrics));
models = cell(length(k_values), length(distance_metrics));

% Train and evaluate for different k and distance metrics
for i = 1:length(k_values)
    for j = 1:length(distance_metrics)
        models{i, j} = knnclassification_cornerfeature(train_feats, train_labels, ...
            k_values(i), distance_metrics{j});
        
        % Predictions for validation set
        pred_labels = testing(models{i, j}, valid_feats);
        accuracies(i, j) = mean(strcmp(pred_labels, valid_labels));
        
        % Print predictions for each combination of k and distance metric
        fprintf('\nPredictions for K = %d, Distance Metric = %s:\n', k_values(i), distance_metrics{j});
        disp(pred_labels');
    end
end

% Find the best model based on validation accuracy
[best_accuracy, best_idx] = max(accuracies(:));
[best_k_idx, best_metric_idx] = ind2sub(size(accuracies), best_idx);
best_k = k_values(best_k_idx);
best_metric = distance_metrics{best_metric_idx};
best_model = models{best_k_idx, best_metric_idx};

fprintf('\nBest K: %d\n', best_k);
fprintf('Best Distance Metric: %s\n', best_metric);
fprintf('Validation Accuracy: %.2f%%\n', best_accuracy * 100);

% Test the best model
[test_image_paths, ~] = get_image_paths(data_path, categories, 5, 0);  % Get test set with 5 images
test_feats = BagsOfVisualWord(test_image_paths, vocab);
test_predictions = testing(best_model, test_feats);

% Print test predictions
disp('\nTest Predictions (Best Model):');
disp(test_predictions);

% Print predictions for K = 3, K = 5, and K = 10 on test set
k_values_test = [3, 5, 10];
for i = 1:length(k_values_test)
    fprintf('\nPredictions for K = %d on Test Set:\n', k_values_test(i));
    model = knnclassification_cornerfeature(train_feats, train_labels, k_values_test(i), best_metric);
    test_preds = testing(model, test_feats);
    disp(test_preds);
end

% File untuk menyimpan hasil prediksi
output_file = 'test_predictions.txt';

% Menyimpan hasil prediksi untuk setiap K
for i = 1:length(k_values_test)
    k_value = k_values_test(i);
    fprintf('\nPredictions for K = %d on Test Set:\n', k_value);
    
    % Lakukan prediksi
    model = knnclassification_cornerfeature(train_feats, train_labels, k_value, best_metric);
    test_preds = testing(model, test_feats);
    
    % Mencetak dan menyimpan hasil prediksi
    disp(test_preds);
    fid = fopen(output_file, 'a'); % Menambah ke file
    fprintf(fid, 'Predictions for K = %d:\n', k_value);
    for j = 1:length(test_preds)
        fprintf(fid, '%s\n', test_preds{j});
    end
    fclose(fid);
end
