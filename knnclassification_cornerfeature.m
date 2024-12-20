function model = knnclassification_cornerfeature(train_feats, train_labels, k, distance_metric)
model = fitcknn(train_feats, train_labels, ...
    'NumNeighbors', k, ...
    'Distance', distance_metric);
end
