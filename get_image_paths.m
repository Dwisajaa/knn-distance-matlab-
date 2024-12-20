function [train_image_paths, valid_image_paths, train_labels, valid_labels] = ...
    get_image_paths(data_path, categories, num_train_per_cat, num_valid_per_cat)

num_categories = length(categories);

% Initialize cell arrays
train_image_paths = cell(num_categories * num_train_per_cat, 1);
valid_image_paths = cell(num_categories * num_valid_per_cat, 1);
train_labels = cell(num_categories * num_train_per_cat, 1);
valid_labels = cell(num_categories * num_valid_per_cat, 1);

for i = 1:num_categories
   train_images = dir(fullfile(data_path, 'train', categories{i}, '*.jpg'));
   valid_images = dir(fullfile(data_path, 'valid', categories{i}, '*.jpg'));
   
   % Assign training images and labels
   for j = 1:num_train_per_cat
       train_image_paths{(i-1)*num_train_per_cat + j} = ...
           fullfile(data_path, 'train', categories{i}, train_images(j).name);
       train_labels{(i-1)*num_train_per_cat + j} = categories{i};
   end
   
   % Assign validation images and labels
   for j = 1:num_valid_per_cat
       valid_image_paths{(i-1)*num_valid_per_cat + j} = ...
           fullfile(data_path, 'valid', categories{i}, valid_images(j).name);
       valid_labels{(i-1)*num_valid_per_cat + j} = categories{i};
   end
end
end
