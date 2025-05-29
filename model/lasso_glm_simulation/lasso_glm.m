%/usr/bin/env matlab
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Created on 14:49, May. 15th, 2023
% 
% @author: Norbert Zheng
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [acc_pred]=lasso_glm(params, X, y)
    % Use the inner `lassoglm` function to fit the given brain recordings.
    %
    % Args:
    %     params: struct - Model parameters initialized by lasso_glm_params.
    %     X: (11[tuple],) - The input data (X_train, X_test, X_test_2,...),
    %         each item is of shape (n_samples, n_channels).
    %     y: (11[tuple],) - The target labels (y_train, y_test, y_test_2,...), each item is of shape (n_samples,).
    %
    % Returns:
    %     acc_pred: (2,) - The predicted accuracy of ecah lasso classifier trained
    %         at each time point. The first row is test-accuracy, and the second row is test-accuracy.

    %% Prepare data & parameters for lasso glm.
    % Initialize train-set & test-set & test-set from `X` & `y`.
    X_train = X{1}; X_test = X{2}; X_test_2 = X{3}; X_test_3 = X{4};X_test_4 = X{5};X_test_5 = X{6};X_test_6 = X{7};X_test_7 = X{8};X_test_8 = X{9};X_test_9 = X{10};X_test_10 = X{11};
    y_train = y{1}; y_test = y{2}; y_test_2 = y{3};y_test_3 = y{4};y_test_4 = y{5};y_test_5 = y{6};y_test_6 = y{7};y_test_7 = y{8};y_test_8 = y{9};y_test_9 = y{10};y_test_10 = y{11};
    % Initialize `n_labels`, we train `n_labels` binomial lasso classifier for each time point.
    labels = unique(y_train); n_labels = length(labels);
    % Initialize other parameters for lasso glm.
    l1_penalty = params.l1_penalty; l2_penalty = params.l2_penalty;

    %% Correct data for lasso glm.
    % TODO: Execute artifect correction.

    %% Execute lasso glm training.
    % Initialize `acc_pred` as nans.
    acc_pred = nan(10,1);
    % Train lasso glm.
    % Scale `X` to improve the generalization ability of lasso glm.
    % Note: percentile-normalization is better than std-normalzation.
    X_train = X_train ./ prctile(abs(X_train), 95);
    X_test = X_test ./ prctile(abs(X_test), 95);
    X_test_2 = X_test_2 ./ prctile(abs(X_test_2), 95);
    X_test_3 = X_test_3 ./ prctile(abs(X_test_3), 95);
    X_test_4 = X_test_4 ./ prctile(abs(X_test_4), 95);
    X_test_5 = X_test_5 ./ prctile(abs(X_test_5), 95);
    X_test_6 = X_test_6 ./ prctile(abs(X_test_6), 95);
    X_test_7 = X_test_7 ./ prctile(abs(X_test_7), 95);
    X_test_8 = X_test_8 ./ prctile(abs(X_test_8), 95);
    X_test_9 = X_test_9 ./ prctile(abs(X_test_9), 95);
    X_test_10 = X_test_10 ./ prctile(abs(X_test_10), 95);

    % Initialize `pred_i` as nans.
    % pred_i - (n_labels, n_samples)
    pred_test_i = nan(n_labels, size(y_test, 1)); pred_test_2_i = nan(n_labels, size(y_test_2, 1));
    pred_test_3_i = nan(n_labels, size(y_test_3, 1)); pred_test_4_i = nan(n_labels, size(y_test_4, 1));
    pred_test_5_i = nan(n_labels, size(y_test_5, 1)); pred_test_6_i = nan(n_labels, size(y_test_6, 1));
    pred_test_7_i = nan(n_labels, size(y_test_7, 1)); pred_test_8_i = nan(n_labels, size(y_test_8, 1));
    pred_test_9_i = nan(n_labels, size(y_test_9, 1)); pred_test_10_i = nan(n_labels, size(y_test_10, 1));
    % Train lasso glm for current time point.
    for label_train_idx = 1:n_labels
        % Fit lasso glm for current label using train-set.
        [W_i,fitinfo_i] = lassoglm(X_train, (y_train == labels(label_train_idx)), 'binomial', 'Standardize', false, ...
            'Alpha', (l1_penalty / (2 * l2_penalty + l1_penalty)), 'Lambda', (2 * l2_penalty + l1_penalty));
        % Test the fitted lasso glm on test-set & test-set.
        pred_test_i(label_train_idx,:) = 1 ./ (1 + exp(-(X_test * W_i + fitinfo_i.Intercept)));
        pred_test_2_i(label_train_idx,:) = 1 ./ (1 + exp(-(X_test_2 * W_i + fitinfo_i.Intercept)));
        pred_test_3_i(label_train_idx,:) = 1 ./ (1 + exp(-(X_test_3 * W_i + fitinfo_i.Intercept)));
        pred_test_4_i(label_train_idx,:) = 1 ./ (1 + exp(-(X_test_4 * W_i + fitinfo_i.Intercept)));
        pred_test_5_i(label_train_idx,:) = 1 ./ (1 + exp(-(X_test_5 * W_i + fitinfo_i.Intercept)));
        pred_test_6_i(label_train_idx,:) = 1 ./ (1 + exp(-(X_test_6 * W_i + fitinfo_i.Intercept)));
        pred_test_7_i(label_train_idx,:) = 1 ./ (1 + exp(-(X_test_7 * W_i + fitinfo_i.Intercept)));
        pred_test_8_i(label_train_idx,:) = 1 ./ (1 + exp(-(X_test_8 * W_i + fitinfo_i.Intercept)));
        pred_test_9_i(label_train_idx,:) = 1 ./ (1 + exp(-(X_test_9 * W_i + fitinfo_i.Intercept)));
        pred_test_10_i(label_train_idx,:) = 1 ./ (1 + exp(-(X_test_10 * W_i + fitinfo_i.Intercept)));
    end
    % Calculate `acc_i` from `pred_i`.
    if strcmp(params.acc_mode, 'default')
        % Calculate the accuracy on test-set.
        % y_pred_test - (n_samples, 1)
        [~,y_pred_test_idxs] = max(pred_test_i, [], 1);
        y_pred_test_idxs = reshape(y_pred_test_idxs, [], 1);
        y_pred_test = labels(y_pred_test_idxs);
        acc_pred(1,:) = mean(y_test == y_pred_test);
        % Calculate the accuracy on test-set.
        % y_pred_test - (n_samples, 1)
        [~,y_pred_test_2_idxs] = max(pred_test_2_i, [], 1);
        y_pred_test_2_idxs = reshape(y_pred_test_2_idxs, [], 1);
        y_pred_test_2 = labels(y_pred_test_2_idxs);
        acc_pred(2,:) = mean(y_test_2 == y_pred_test_2);
        % y_pred_test - (n_samples, 1)
        [~,y_pred_test_3_idxs] = max(pred_test_3_i, [], 1);
        y_pred_test_3_idxs = reshape(y_pred_test_3_idxs, [], 1);
        y_pred_test_3 = labels(y_pred_test_3_idxs);
        acc_pred(3,:) = mean(y_test_3 == y_pred_test_3);
        % y_pred_test - (n_samples, 1)
        [~,y_pred_test_4_idxs] = max(pred_test_4_i, [], 1);
        y_pred_test_4_idxs = reshape(y_pred_test_4_idxs, [], 1);
        y_pred_test_4 = labels(y_pred_test_4_idxs);
        acc_pred(4,:) = mean(y_test_4 == y_pred_test_4);
        % y_pred_test - (n_samples, 1)
        [~,y_pred_test_5_idxs] = max(pred_test_5_i, [], 1);
        y_pred_test_5_idxs = reshape(y_pred_test_5_idxs, [], 1);
        y_pred_test_5 = labels(y_pred_test_5_idxs);
        acc_pred(5,:) = mean(y_test_5 == y_pred_test_5);
        % y_pred_test - (n_samples, 1)
        [~,y_pred_test_6_idxs] = max(pred_test_6_i, [], 1);
        y_pred_test_6_idxs = reshape(y_pred_test_6_idxs, [], 1);
        y_pred_test_6 = labels(y_pred_test_6_idxs);
        acc_pred(6,:) = mean(y_test_6 == y_pred_test_6);
        % y_pred_test - (n_samples, 1)
        [~,y_pred_test_7_idxs] = max(pred_test_7_i, [], 1);
        y_pred_test_7_idxs = reshape(y_pred_test_7_idxs, [], 1);
        y_pred_test_7 = labels(y_pred_test_7_idxs);
        acc_pred(7,:) = mean(y_test_7 == y_pred_test_7);
        % y_pred_test - (n_samples, 1)
        [~,y_pred_test_8_idxs] = max(pred_test_8_i, [], 1);
        y_pred_test_8_idxs = reshape(y_pred_test_8_idxs, [], 1);
        y_pred_test_8 = labels(y_pred_test_8_idxs);
        acc_pred(8,:) = mean(y_test_8 == y_pred_test_8);
        % y_pred_test - (n_samples, 1)
        [~,y_pred_test_9_idxs] = max(pred_test_9_i, [], 1);
        y_pred_test_9_idxs = reshape(y_pred_test_9_idxs, [], 1);
        y_pred_test_9 = labels(y_pred_test_9_idxs);
        acc_pred(9,:) = mean(y_test_9 == y_pred_test_9);
        % y_pred_test - (n_samples, 1)
        [~,y_pred_test_10_idxs] = max(pred_test_10_i, [], 1);
        y_pred_test_10_idxs = reshape(y_pred_test_10_idxs, [], 1);
        y_pred_test_10 = labels(y_pred_test_10_idxs);
        acc_pred(10,:) = mean(y_test_10 == y_pred_test_10);
    elseif strcmp(params.acc_mode, 'lvbj')
        % Calculate the accuracy on test-set.
        % classifier_ratio_test - (n_labels, n_labels)
        classifier_ratio_test = nan(n_labels, n_labels);
        for label_test_idx = 1:n_labels
            % Note: We have assumed that each category has balanced samples!
            classifier_ratio_test(:,label_test_idx) = ...
                mean(pred_test_i(:,(y_test == labels(label_test_idx))), 2);
        end
        % classifier_test_idxs - (n_labels, 1)
        [~,classifier_test_idxs] = max(classifier_ratio_test, [], 1);
        acc_pred(1,:) = mean(classifier_test_idxs == 1:n_labels);
        % Calculate the accuracy on test-set.
        % classifier_ratio_test - (n_labels, n_labels)
        classifier_ratio_test_2 = nan(n_labels, n_labels);
        for label_test_idx = 1:n_labels
            % Note: We have assumed that each category has balanced samples!
            classifier_ratio_test_2(:,label_test_idx) = ...
                mean(pred_test_2_i(:,(y_test_2 == labels(label_test_idx))), 2);
        end
        % classifier_test_idxs - (n_labels, 1)
        [~,classifier_test_2_idxs] = max(classifier_ratio_test_2, [], 1);
        acc_pred(2,:) = mean(classifier_test_2_idxs == 1:n_labels);

        classifier_ratio_test_3 = nan(n_labels, n_labels);
        for label_test_idx = 1:n_labels
            % Note: We have assumed that each category has balanced samples!
            classifier_ratio_test_3(:,label_test_idx) = ...
                mean(pred_test_3_i(:,(y_test_3 == labels(label_test_idx))), 2);
        end
        % classifier_test_idxs - (n_labels, 1)
        [~,classifier_test_3_idxs] = max(classifier_ratio_test_3, [], 1);
        acc_pred(3,:) = mean(classifier_test_3_idxs == 1:n_labels);

        classifier_ratio_test_4 = nan(n_labels, n_labels);
        for label_test_idx = 1:n_labels
            % Note: We have assumed that each category has balanced samples!
            classifier_ratio_test_4(:,label_test_idx) = ...
                mean(pred_test_4_i(:,(y_test_4 == labels(label_test_idx))), 2);
        end
        % classifier_test_idxs - (n_labels, 1)
        [~,classifier_test_4_idxs] = max(classifier_ratio_test_4, [], 1);
        acc_pred(4,:) = mean(classifier_test_4_idxs == 1:n_labels);

        classifier_ratio_test_5 = nan(n_labels, n_labels);
        for label_test_idx = 1:n_labels
            % Note: We have assumed that each category has balanced samples!
            classifier_ratio_test_5(:,label_test_idx) = ...
                mean(pred_test_5_i(:,(y_test_5 == labels(label_test_idx))), 2);
        end
        % classifier_test_idxs - (n_labels, 1)
        [~,classifier_test_5_idxs] = max(classifier_ratio_test_5, [], 1);
        acc_pred(5,:) = mean(classifier_test_5_idxs == 1:n_labels);

        classifier_ratio_test_6 = nan(n_labels, n_labels);
        for label_test_idx = 1:n_labels
            % Note: We have assumed that each category has balanced samples!
            classifier_ratio_test_6(:,label_test_idx) = ...
                mean(pred_test_6_i(:,(y_test_6 == labels(label_test_idx))), 2);
        end
        % classifier_test_idxs - (n_labels, 1)
        [~,classifier_test_6_idxs] = max(classifier_ratio_test_6, [], 1);
        acc_pred(6,:) = mean(classifier_test_6_idxs == 1:n_labels);

        classifier_ratio_test_7 = nan(n_labels, n_labels);
        for label_test_idx = 1:n_labels
            % Note: We have assumed that each category has balanced samples!
            classifier_ratio_test_7(:,label_test_idx) = ...
                mean(pred_test_7_i(:,(y_test_7 == labels(label_test_idx))), 2);
        end
        % classifier_test_idxs - (n_labels, 1)
        [~,classifier_test_7_idxs] = max(classifier_ratio_test_7, [], 1);
        acc_pred(7,:) = mean(classifier_test_7_idxs == 1:n_labels);

        classifier_ratio_test_8 = nan(n_labels, n_labels);
        for label_test_idx = 1:n_labels
            % Note: We have assumed that each category has balanced samples!
            classifier_ratio_test_8(:,label_test_idx) = ...
                mean(pred_test_8_i(:,(y_test_8 == labels(label_test_idx))), 2);
        end
        % classifier_test_idxs - (n_labels, 1)
        [~,classifier_test_8_idxs] = max(classifier_ratio_test_8, [], 1);
        acc_pred(8,:) = mean(classifier_test_8_idxs == 1:n_labels);

        classifier_ratio_test_9 = nan(n_labels, n_labels);
        for label_test_idx = 1:n_labels
            % Note: We have assumed that each category has balanced samples!
            classifier_ratio_test_9(:,label_test_idx) = ...
                mean(pred_test_9_i(:,(y_test_9 == labels(label_test_idx))), 2);
        end
        % classifier_test_idxs - (n_labels, 1)
        [~,classifier_test_9_idxs] = max(classifier_ratio_test_9, [], 1);
        acc_pred(9,:) = mean(classifier_test_9_idxs == 1:n_labels);

        classifier_ratio_test_10 = nan(n_labels, n_labels);
        for label_test_idx = 1:n_labels
            % Note: We have assumed that each category has balanced samples!
            classifier_ratio_test_10(:,label_test_idx) = ...
                mean(pred_test_10_i(:,(y_test_10 == labels(label_test_idx))), 2);
        end
        % classifier_test_idxs - (n_labels, 1)
        [~,classifier_test_10_idxs] = max(classifier_ratio_test_10, [], 1);
        acc_pred(10,:) = mean(classifier_test_10_idxs == 1:n_labels);
    end
end

