%/usr/bin/env matlab

function [acc_pred]=lasso_glm(params, X, y)
    % Use the inner `lassoglm` function to fit the given brain recordings.
    %
    % Args:
    %     params: struct - Model parameters initialized by lasso_glm_params.
    %     X: (3[tuple],) - The input data (X_train, X_test1, X_test2),
    %         each item is of shape (n_samples, seq_len, n_channels).
    %     y: (3[tuple],) - The target labels (y_train, y_test1, y_test2), each item is of shape (n_samples,).
    %
    % Returns:
    %     acc_pred: (2, seq_len) - The predicted accuracy of ecah lasso classifier trained
    %         at each time point. The first row is test1-accuracy, and the second row is test2-accuracy.

    %% Prepare data & parameters for lasso glm.
    % Initialize train-set & test1-set & test2-set from `X` & `y`.
    X_train = X{1}; X_test1 = X{2}; X_test2 = X{3}; y_train = y{1}; y_test1 = y{2}; y_test2 = y{3};
    % Initialize `seq_len`, we train lasso classifier for each time point.
    [~,seq_len,~] = size(X_train);
    assert(size(X_train, 2) == size(X_test1, 2));
    assert(size(X_train, 2) == size(X_test2, 2));
    % Initialize `n_labels`, we train `n_labels` binomial lasso classifier for each time point.
    labels = unique(y_train); n_labels = length(labels);
    % assert(length(unique(y_train)) == length(unique(y_test1)));
    % assert(length(unique(y_train)) == length(unique(y_test2)));
    % Initialize other parameters for lasso glm.
    l1_penalty = params.l1_penalty; l2_penalty = params.l2_penalty;

    %% Correct data for lasso glm.
    % TODO: Execute artifect correction.

    %% Execute lasso glm training.
    % Initialize `acc_pred` as nans.
    acc_pred = nan(2, seq_len);
    % Train lasso glm for each time point.
    for time_idx = 1:seq_len
        % Select `X_i` for current time point.
        % X_i - (n_samples, n_channels)
        X_train_i = squeeze(X_train(:,time_idx,:));
        X_test1_i = squeeze(X_test1(:,time_idx,:));
        X_test2_i = squeeze(X_test2(:,time_idx,:));
        % Scale `X_i` to improve the generalization ability of lasso glm.
        % Note: percentile-normalization is better than std-normalzation.
        X_train_i = X_train_i ./ prctile(abs(X_train_i), 95);
        X_test1_i = X_test1_i ./ prctile(abs(X_test1_i), 95);
        X_test2_i = X_test2_i ./ prctile(abs(X_test2_i), 95);

        % Initialize `pred_i` as nans.
        % pred_i - (n_labels, n_samples)
        pred_test1_i = nan(n_labels, size(y_test1, 1)); pred_test2_i = nan(n_labels, size(y_test2, 1));
        % Train lasso glm for current time point.
        for label_train_idx = 1:n_labels
            % Fit lasso glm for current label using train-set.
            [W_i,fitinfo_i] = lassoglm(X_train_i, (y_train == labels(label_train_idx)), 'binomial', 'Standardize', false, ...
                'Alpha', (l1_penalty / (2 * l2_penalty + l1_penalty)), 'Lambda', (2 * l2_penalty + l1_penalty));
            % test2 the fitted lasso glm on test1-set & test2-set.
            pred_test1_i(label_train_idx,:) = 1 ./ (1 + exp(-(X_test1_i * W_i + fitinfo_i.Intercept)));
            pred_test2_i(label_train_idx,:) = 1 ./ (1 + exp(-(X_test2_i * W_i + fitinfo_i.Intercept)));
        end
        % Calculate `acc_i` from `pred_i`.
        if strcmp(params.acc_mode, 'default')
            % Calculate the accuracy on test1-set.
            % y_pred_test1 - (n_samples, 1)
            [~,y_pred_test1_idxs] = max(pred_test1_i, [], 1);
            y_pred_test1_idxs = reshape(y_pred_test1_idxs, [], 1);
            y_pred_test1 = labels(y_pred_test1_idxs);
            acc_pred(1,time_idx) = mean(y_test1 == y_pred_test1);
            % Calculate the accuracy on test2-set.
            % y_pred_test2 - (n_samples, 1)
            [~,y_pred_test2_idxs] = max(pred_test2_i, [], 1);
            y_pred_test2_idxs = reshape(y_pred_test2_idxs, [], 1);
            y_pred_test2 = labels(y_pred_test2_idxs);
            acc_pred(2,time_idx) = mean(y_test2 == y_pred_test2);
        elseif strcmp(params.acc_mode, 'lvbj')
            % Calculate the accuracy on test1-set.
            % classifier_ratio_test1 - (n_labels, n_labels)
            classifier_ratio_test1 = nan(n_labels, n_labels);
            for label_test1_idx = 1:n_labels
                % Note: We have assumed that each category has balanced samples!
                classifier_ratio_test1(:,label_test1_idx) = ...
                    mean(pred_test1_i(:,(y_test1 == labels(label_test1_idx))), 2);
            end
            % classifier_test1_idxs - (n_labels, 1)
            [~,classifier_test1_idxs] = max(classifier_ratio_test1, [], 1);
            acc_pred(1,time_idx) = mean(classifier_test1_idxs == 1:n_labels);
            % Calculate the accuracy on test2-set.
            % classifier_ratio_test2 - (n_labels, n_labels)
            classifier_ratio_test2 = nan(n_labels, n_labels);
            for label_test2_idx = 1:n_labels
                % Note: We have assumed that each category has balanced samples!
                classifier_ratio_test2(:,label_test2_idx) = ...
                    mean(pred_test2_i(:,(y_test2 == labels(label_test2_idx))), 2);
            end
            % classifier_test2_idxs - (n_labels, 1)
            [~,classifier_test2_idxs] = max(classifier_ratio_test2, [], 1);
            acc_pred(2,time_idx) = mean(classifier_test2_idxs == 1:n_labels);
        end
    end
end

