%/usr/bin/env matlab
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [acc_pred]=lasso_glm(params, X, y)
    % Use the inner `lassoglm` function to fit the given brain recordings.
    %
    % Args:
    %     params: struct - Model parameters initialized by lasso_glm_params.
    %     X: (3[tuple],) - The input data (X_train, X_test1, X_test),
    %         each item is of shape (n_samples, seq_len, n_channels).
    %     y: (3[tuple],) - The target labels (y_train, y_test1, y_test), each item is of shape (n_samples,).
    %
    % Returns:
    %     acc_pred: (2, train_seq_len, test_seq_len) - The predicted accuracy 
    %         of each lasso classifier trained at each training time point 
    %         tested on each test1/test time point. The first row is 
    %         test1-accuracy, and the second row is test-accuracy.

    %% Prepare data & parameters for lasso glm.
    % Initialize train-set & test1-set & test-set from `X` & `y`.
    X_train = X{1}; X_test1 = X{2}; X_test = X{3}; y_train = y{1}; y_test1 = y{2}; y_test = y{3};
    % Determine train and test (test1) sequence lengths
    train_seq_len = size(X_train, 2);
    test1_seq_len = size(X_test1, 2);
    test_seq_len = size(X_test, 2);
    % Initialize `n_labels`, we train `n_labels` binomial lasso classifier for each time point.
    labels = unique(y_train);n_labels = length(labels);

    % Initialize other parameters for lasso glm.
    l1_penalty = params.l1_penalty; l2_penalty = params.l2_penalty;

    %% Note on approach:
    % We will:
    % 1) Train a separate model for each time point in the train data.
    % 2) For each trained model, we test it on every time point of test1 and test data.

    %% Train lasso glm for each time point in the training data
    % We will store the trained weights/intercepts for each time point and each label
    W = cell(train_seq_len, n_labels);
    fitinfo = cell(train_seq_len, n_labels);

    for train_time_idx = 1:train_seq_len
        % Extract training data at this specific time point
        X_train_i = squeeze(X_train(:, train_time_idx, :));
        % Scale training data
        X_train_i = X_train_i ./ prctile(abs(X_train_i), 95);

        for label_train_idx = 1:n_labels
            % Fit lasso glm for current label using train-set at this time point
            [W_i, fitinfo_i] = lassoglm(X_train_i, (y_train == labels(label_train_idx)), 'binomial', 'Standardize', false, ...
                'Alpha', (l1_penalty / (2 * l2_penalty + l1_penalty)), 'Lambda', (2 * l2_penalty + l1_penalty));
            % Store the resulting model
            W{train_time_idx, label_train_idx} = W_i;
            fitinfo{train_time_idx, label_train_idx} = fitinfo_i.Intercept;
        end
    end

    %% Test each trained model on all time points of the test1 and test sets
    % Initialize `acc_pred` as nans. Now it is (2, train_seq_len, test_seq_len)
    acc_pred = nan(2, train_seq_len, test_seq_len);

    for train_time_idx = 1:train_seq_len
        % For each test/test1 time point, we predict using the model trained at train_time_idx
        for test_time_idx = 1:test_seq_len
            % Extract test1/test data at this test_time_idx
            X_test1_i = squeeze(X_test1(:, test_time_idx, :));
            X_test1_i = X_test1_i ./ prctile(abs(X_test1_i), 95);

            X_test_i = squeeze(X_test(:, test_time_idx, :));
            X_test_i = X_test_i ./ prctile(abs(X_test_i), 95);

            % Predict probabilities for each label
            pred_test1_i = nan(n_labels, size(y_test1, 1));
            pred_test_i = nan(n_labels, size(y_test, 1));

            for label_train_idx = 1:n_labels
                W_current = W{train_time_idx, label_train_idx};
                b_current = fitinfo{train_time_idx, label_train_idx};

                pred_test1_i(label_train_idx,:) = 1 ./ (1 + exp(-(X_test1_i * W_current + b_current)));
                pred_test_i(label_train_idx,:) = 1 ./ (1 + exp(-(X_test_i * W_current + b_current)));
            end

            % Calculate `acc_i` according to the specified mode
            if strcmp(params.acc_mode, 'default')
                % test1 accuracy
                [~,y_pred_test1_idxs] = max(pred_test1_i, [], 1);
                y_pred_test1_idxs = reshape(y_pred_test1_idxs, [], 1);
                y_pred_test1 = labels(y_pred_test1_idxs);
                acc_pred(1,train_time_idx,test_time_idx) = mean(y_test1 == y_pred_test1);

                % Test accuracy
                [~,y_pred_test_idxs] = max(pred_test_i, [], 1);
                y_pred_test_idxs = reshape(y_pred_test_idxs, [], 1);
                y_pred_test = labels(y_pred_test_idxs);
                acc_pred(2,train_time_idx,test_time_idx) = mean(y_test == y_pred_test);
            elseif strcmp(params.acc_mode, 'lvbj')
                % Calculate the accuracy on test1-set.
                classifier_ratio_test1 = nan(n_labels, n_labels);
                for label_test1_idx = 1:n_labels
                    classifier_ratio_test1(:,label_test1_idx) = ...
                        mean(pred_test1_i(:,(y_test1 == labels(label_test1_idx))), 2);
                end
                [~,classifier_test1_idxs] = max(classifier_ratio_test1, [], 1);
                acc_pred(1,train_time_idx,test_time_idx) = mean(classifier_test1_idxs == 1:n_labels);

                % Calculate the accuracy on test-set.
                classifier_ratio_test = nan(n_labels, n_labels);
                for label_test_idx = 1:n_labels
                    classifier_ratio_test(:,label_test_idx) = ...
                        mean(pred_test_i(:,(y_test == labels(label_test_idx))), 2);
                end
                [~,classifier_test_idxs] = max(classifier_ratio_test, [], 1);
                acc_pred(2,train_time_idx,test_time_idx) = mean(classifier_test_idxs == 1:n_labels);
            end
        end
    end
end

