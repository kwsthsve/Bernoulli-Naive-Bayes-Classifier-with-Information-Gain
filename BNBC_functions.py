import numpy as np

IG_TOP = 400

# P(X=1) * (1 - P(X=1))
VAR_THRESHOLD = 0.8 * 0.2


# Class separation function

def separate_classes(training_set, response):
    class_0 = list()
    class_1 = list()

    for i in range(len(training_set)):
        if response[i] == 0:
            class_0.append(training_set[i])
        else:
            class_1.append(training_set[i])

    class_0 = np.array(class_0)
    class_1 = np.array(class_1)

    return class_0, class_1


# Calculate entropy for 2 classes

def twoClassEntropy(class_Prob):
    entropy = 0
    if (class_Prob != 0) and (class_Prob != 1):
        entropy = - (class_Prob * np.log2(class_Prob)) - ((1 - class_Prob) * np.log2(1 - class_Prob))

    return entropy


# Calculate Information Gain for each feature

def calculate_IG(features, response):
    examples = len(response)

    InfoGain = list()
    prior_1 = response.sum(axis=0) / len(response)

    # Class 1 entropy
    HC = twoClassEntropy(prior_1)

    prob_feature_1 = list()
    prob_feature_0_given_class_1 = list()
    prob_feature_1_given_class_1 = list()
    HC_11 = list()
    HC_01 = list()

    class_0, class_1 = separate_classes(np.transpose(features), response)

    for i in range(len(features)):

        feature_1_count = features[i].sum(axis=0)
        feature_0_given_class_1_count = np.count_nonzero(class_1.T[i] == 0)
        feature_1_given_class_1_count = np.count_nonzero(class_1.T[i] == 1)

        # P(X=1)
        prob_feature_1.append(feature_1_count / examples)

        # P(X=1/C=1)
        if feature_1_count == 0:
            prob_feature_1_given_class_1.append(0)
        else:
            prob_feature_1_given_class_1.append(feature_1_given_class_1_count / feature_1_count)

        # P(X=0/C=1)
        if feature_1_count == examples:
            prob_feature_0_given_class_1.append(0)
        else:
            prob_feature_0_given_class_1.append(feature_0_given_class_1_count / (examples - feature_1_count))

        # P(X=0/C=1) entropy
        HC_01.append(twoClassEntropy(prob_feature_0_given_class_1[i]))
        # P(X=1/C=1) entropy
        HC_11.append(twoClassEntropy(prob_feature_1_given_class_1[i]))

        # IG formula...
        InfoGain.append(HC - ((prob_feature_1[i] * HC_11[i]) + ((1 - prob_feature_1[i]) * HC_01[i])))

    return InfoGain


# Return indices of the 'IG_TOP' features with the highest IG

def sort_features(features, response):
    IG = calculate_IG(features, response)
    highest_IG = list()

    for i in range(IG_TOP):
        highest_IG.append(np.argmax(IG))
        IG = np.delete(IG, highest_IG[i])

    return highest_IG


# Second method to sort the features (Do not use both..)

def sort_features_by_var(features):
    high_var_features = list()

    for i in range(len(features)):
        var = np.var(features[i])
        if var > VAR_THRESHOLD:
            high_var_features.append(i)

    return high_var_features


# Calculate prior log probability for each class (smoothed to avoid 0 values)

def calculate_prior_y(response):
    prior_1 = (response.sum(axis=0) + 1) / (len(response) + 2)
    return [np.log(1 - prior_1), np.log(prior_1)]


# Calculate log Likelihood for each feature (smoothed to avoid 0 values)
# returns a vector like [ [log(P(X=0/C=0)), log(P(X=1/C=0))],[log(P(X=0/C=1)), log(P(X=1/C=1))] ]
# we use logarithm to smooth the data and avoid floating errors

def calculate_log_likelihood(training_set, response):
    class_0, class_1 = separate_classes(training_set, response)

    log_likelihood_vector = list()

    for i in range(len(np.transpose(training_set))):
        log_likelihood_00 = np.log(np.count_nonzero(class_0.T[i] == 0) + 1) - np.log(len(class_0) + 2)
        log_likelihood_10 = np.log(np.count_nonzero(class_0.T[i] == 1) + 1) - np.log(len(class_0) + 2)
        log_likelihood_01 = np.log(np.count_nonzero(class_1.T[i] == 0) + 1) - np.log(len(class_1) + 2)
        log_likelihood_11 = np.log(np.count_nonzero(class_1.T[i] == 1) + 1) - np.log(len(class_1) + 2)

        log_likelihood_vector.append([[log_likelihood_00, log_likelihood_10], [log_likelihood_01, log_likelihood_11]])

    return np.array(log_likelihood_vector)


# Calculate log post probability log(P(C=0/X=<...>)) and log(P(C=1/X=<...>)) for each example
# and compare to make final prediction

def fit_and_predict(testing_set, response, sorted_features, training_set):
    classes = np.array([0, 1])
    predictions = list()

    training_set = np.transpose(training_set)[sorted_features]
    testing_set = np.transpose(testing_set)[sorted_features]

    log_prior = calculate_prior_y(response)
    log_likelihood = calculate_log_likelihood(training_set.T, response)

    for test_obs in testing_set.T:

        likelihood_vector = [1] * len(classes)

        for j in range(len(classes)):

            for f in range(len(sorted_features)):
                # feature_index = int(feature)
                feature_value = int(test_obs[f])
                likelihood_vector[j] += log_likelihood[f][j][feature_value]

        log_post_prob = [1] * len(classes)
        for j in range(len(classes)):
            log_post_prob[j] = likelihood_vector[j] + log_prior[j]

        predictions.append(np.argmax(log_post_prob))

    return np.array(predictions)


# Calculate the metrics to evaluate the model

def calculate_metrics(y_prediction_set, y_real):
    corrects = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i in range(len(y_prediction_set)):
        if y_prediction_set[i] == y_real[i]:
            corrects += 1
        if (y_prediction_set[i] == 1) and (y_real[i] == 1):
            true_positives += 1
        if (y_prediction_set[i] == 1) and (y_real[i] == 0):
            false_positives += 1
        if (y_prediction_set[i] == 0) and (y_real[i] == 1):
            false_negatives += 1

    accuracy = int((corrects * 100) / len(y_prediction_set))
    recall = float(true_positives / (true_positives + false_negatives))
    precision = float(true_positives / (true_positives + false_positives))
    mse = np.mean(np.square(np.subtract(y_real, y_prediction_set)))
    f1_score = 2 * ((recall * precision) / (recall + precision))

    return accuracy, mse, recall, precision, f1_score
