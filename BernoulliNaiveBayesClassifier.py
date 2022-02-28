import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as mtpl
from BNBC_functions import fit_and_predict, calculate_metrics, sort_features, sort_features_by_var



def main():

	# Hyper parameters to tune the model

	MAX_FEATURES = 8000
	MOST_FREQUENT = 10
	EXPERIMENTS = 20
	train_set_size = 100
	test_set_size = 50
	set_size_step = 100

	print("Loading data...")
	print()


	# Input examples..

	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=MAX_FEATURES, skip_top=MOST_FREQUENT, seed=123)

	word_index = tf.keras.datasets.imdb.get_word_index()
	index2word = dict((i + 3, word) for (word, i) in word_index.items())
	index2word[0] = '[pad]'
	index2word[1] = '[bos]'
	index2word[2] = '[oov]'

	x_train = np.array([' '.join([index2word[idx] for idx in text]) for text in x_train])
	x_test = np.array([' '.join([index2word[idx] for idx in text]) for text in x_test])


	# Fill the vocabulary with distinct words

	vocabulary = list()
	for text in x_train:
		tokens = text.split()
		vocabulary.extend(tokens)

	vocabulary = set(vocabulary)

	x_train_binary = list()
	x_test_binary = list()


	# Binarize the examples

	for text in tqdm(x_train):
		tokens = text.split()
		binary_vector = list()
		for vocab_token in vocabulary:
			if vocab_token in tokens:
				binary_vector.append(1)
			else:
				binary_vector.append(0)
		x_train_binary.append(binary_vector)

	x_train_binary = np.array(x_train_binary)

	print()

	for text in tqdm(x_test):
		tokens = text.split()
		binary_vector = list()
		for vocab_token in vocabulary:
			if vocab_token in tokens:
				binary_vector.append(1)
			else:
				binary_vector.append(0)
		x_test_binary.append(binary_vector)

	x_test_binary = np.array(x_test_binary)


	# Initialize the essential lists for the metrics

	accuracy_test = list()
	accuracy_train = list()

	error_test = list()
	error_train = list()

	recall_vector_val = list()
	recall_vector_train = list()

	precision_vector_val = list()
	precision_vector_train = list()

	f1_vector_val = list()
	f1_vector_train = list()

	training_vector = list()
	validation_vector = list()

	print()
	print("Fitting data to model...")
	print()


	# Begin the experiments to find optimal training set size

	for exp in range(EXPERIMENTS):

		# Sampling for each experiment....for estimator's accuracy

		for s in range(25):
			sample_accuracy_val = list()
			sample_accuracy_train = list()

			sample_error_val = list()
			sample_error_train = list()

			sample_recall_vector_val = list()
			sample_recall_vector_train = list()

			sample_precision_vector_val = list()
			sample_precision_vector_train = list()

			sample_f1_vector_val = list()
			sample_f1_vector_train = list()


			# Option 2 for train set size (comment the 1st --> line 14)
			# test_set_size = int(train_set_size*0.2)


			# Random indices' generator for the training set sample

			indices_train = np.random.choice(np.arange(len(x_train_binary)), train_set_size, replace=False)
			indices_test = np.random.choice(np.arange(len(x_test_binary)), test_set_size, replace=False)

			x_training_set_sample = x_train_binary[indices_train]
			y_training_set_sample = y_train[indices_train]

			x_validation_set_sample = x_test_binary[indices_test]
			y_validation_set_sample = y_test[indices_test]

			prediction_vector = fit_and_predict(x_validation_set_sample, y_training_set_sample, sort_features(np.transpose(x_training_set_sample), y_training_set_sample), x_training_set_sample)
			prediction_vector_t = fit_and_predict(x_training_set_sample, y_training_set_sample, sort_features(np.transpose(x_training_set_sample), y_training_set_sample), x_training_set_sample)


			# Option 2 to sort the features (Comment the above..)
			# prediction_vector = fit_and_predict(x_validation_set_sample, y_training_set_sample, sort_features_by_var(np.transpose(x_training_set_sample)), x_training_set_sample)
			# prediction_vector_t = fit_and_predict(x_training_set_sample, y_training_set_sample, sort_features_by_var(np.transpose(x_training_set_sample)), x_training_set_sample)


			training_accuracy, training_error, training_recall, training_precision, training_f1 = calculate_metrics(prediction_vector_t, y_training_set_sample)
			validation_accuracy, validation_error, validation_recall, validation_precision, validation_f1 = calculate_metrics(prediction_vector, y_validation_set_sample)

			sample_accuracy_val.append(validation_accuracy)
			sample_accuracy_train.append(training_accuracy)

			sample_error_val.append(validation_error)
			sample_error_train.append(training_error)

			sample_recall_vector_train.append(training_recall)
			sample_recall_vector_val.append(validation_recall)

			sample_precision_vector_train.append(training_precision)
			sample_precision_vector_val.append(validation_precision)

			sample_f1_vector_train.append(training_f1)
			sample_f1_vector_val.append(validation_f1)

		training_vector.append(train_set_size)
		validation_vector.append(test_set_size)


		# Calculate estimators of experiment after 25 samples

		accuracy_test.append(np.mean(sample_accuracy_val))
		accuracy_train.append(np.mean(sample_accuracy_train))

		error_test.append(np.mean(sample_error_val))
		error_train.append(np.mean(sample_error_train))

		recall_vector_val.append(np.mean(sample_recall_vector_val))
		recall_vector_train.append(np.mean(sample_recall_vector_train))

		precision_vector_val.append(np.mean(sample_precision_vector_val))
		precision_vector_train.append(np.mean(sample_precision_vector_train))

		f1_vector_val.append(np.mean(sample_f1_vector_val))
		f1_vector_train.append(np.mean(sample_f1_vector_train))

		print('Experiment {0} : <{1} training examples>  <{2} validation examples>......Completed {0}/{3}'.format(exp + 1, train_set_size, test_set_size, EXPERIMENTS))
		print('<Loss> : <Training : {0:.2f}>  <Validation : {1:.2f}>'.format(error_train[exp], error_test[exp]))
		print('<Accuracy> : <Training : {0:.2f}>  <Validation : {1:.2f}>'.format(accuracy_train[exp] / 100, accuracy_test[exp] / 100))
		print()
		train_set_size += set_size_step


	# Show the results of the model

	fig, (axs) = mtpl.subplots(2, 2, sharex='all', figsize=(16, 12))

	fig.suptitle('Custom Bernoulli Naive Bayes Classifier', fontsize=18)

	axs[0][0].plot(training_vector, accuracy_test, 'ro-', label='Validation Accuracy')
	axs[0][0].plot(training_vector, accuracy_train, 'bo-', label='Training Accuracy')
	axs[0][0].set_title('Model Accuracy', fontsize=16)
	axs[0][0].set_xlabel('Training Set Size', fontsize=14)
	axs[0][0].set_ylabel('Accuracy %', fontsize=14)
	axs[0][0].grid(visible=True)
	axs[0][0].legend()

	axs[0][1].plot(training_vector, error_test, 'ro-', label='Validation Error')
	axs[0][1].plot(training_vector, error_train, 'bo-', label='Training Error')
	axs[0][1].set_title('Model Error', fontsize=16)
	axs[0][1].set_xlabel('Training Set Size', fontsize=14)
	axs[0][1].set_ylabel('MSE', fontsize=14)
	axs[0][1].grid(visible=True)
	axs[0][1].legend()

	axs[1][0].plot(training_vector, np.array(recall_vector_val) * 100, 'ro-', label='Recall')
	axs[1][0].plot(training_vector, np.array(precision_vector_val) * 100, 'bo-', label='Precision')
	axs[1][0].set_title('Model Validation Set Recall-Precision', fontsize=16)
	axs[1][0].set_xlabel('Training Set Size', fontsize=14)
	axs[1][0].set_ylabel('%', fontsize=14)
	axs[1][0].grid(visible=True)
	axs[1][0].legend()

	axs[1][1].plot(training_vector, f1_vector_val, 'go-', label='F1 Score')
	axs[1][1].set_title('Model Validation Set F1 Score', fontsize=16)
	axs[1][1].set_xlabel('Training Set Size', fontsize=14)
	axs[1][1].set_ylabel('F1', fontsize=14)
	axs[1][1].grid(visible=True)
	axs[1][1].legend()


	# Mean values of each plot to have a general look of the model

	print()
	print('----------------------------------------------------------')
	print()
	print('<Recall> : <Training : {0:.2f}>   <Validation : {1:.2f}>'.format(np.mean(recall_vector_train), np.mean(recall_vector_val)))
	print()
	print('<Precision> : <Training : {0:.2f}>   <Validation : {1:.2f}>'.format(np.mean(precision_vector_train), np.mean(precision_vector_val)))
	print()
	print('<F1 Score> : <Training : {0:.2f}>   <Validation : {1:.2f}>'.format(np.mean(f1_vector_train), np.mean(f1_vector_val)))
	print()
	print('<Accuracy> : <Training : {0:.2f}>   <Validation : {1:.2f}>'.format(np.mean(accuracy_train) / 100, np.mean(accuracy_test) / 100))
	print()
	print('<Loss> : <Training : {0:.2f}>   <Validation : {1:.2f}>'.format(np.mean(error_train), np.mean(error_test)))
	print()

	mtpl.show()

	print('----------------------------------------------------------')
	print()
	print('Fitting best experiment to all validation examples.....')
	print()

	optimal_train_set_size = training_vector[np.argmin(error_test)]


	# Option 2 for train set size (comment the 1st --> line 14)
	# test_set_size = int(optimal_train_set_size*0.2)


	# Predict all the validation examples

	y_prediction_all = list()
	y_validated = list()

	while np.any(x_test_binary):

		if len(x_test_binary) < test_set_size:
			test_set_size = len(x_test_binary)

		indices_train = np.random.choice(np.arange(len(x_train_binary)), optimal_train_set_size, replace=False)
		indices_test = np.random.choice(np.arange(len(x_test_binary)), test_set_size, replace=False)

		x_training_set_sample = x_train_binary[indices_train]
		y_training_set_sample = y_train[indices_train]

		x_validation_set_sample = x_test_binary[indices_test]
		y_validation_set_sample = y_test[indices_test]
		x_test_binary = np.delete(x_test_binary, indices_test, axis=0)
		y_test = np.delete(y_test, indices_test)

		y_prediction = fit_and_predict(x_validation_set_sample, y_training_set_sample, sort_features(np.transpose(x_training_set_sample), y_training_set_sample), x_training_set_sample)


		# Option 2 to sort the features (Comment the above..)
		# y_prediction = fit_and_predict(x_validation_set_sample, y_training_set_sample, sort_features_by_var(np.transpose(x_training_set_sample)), x_training_set_sample)


		y_validated.extend(y_validation_set_sample)
		y_prediction_all.extend(y_prediction.tolist())

	final_accuracy, final_error, final_recall, final_precision, final_f1 = calculate_metrics(y_prediction_all, y_validated)


	# Print the metrics for the validation data

	print('----------------------------------------------------------')
	print()
	print('<Recall> : {0:.2f}'.format(final_recall))
	print()
	print('<Precision> : {0:.2f}'.format(final_precision))
	print()
	print('<F1 Score> : {0:.2f}'.format(final_precision))
	print()
	print('<Accuracy> : {0:.2f}'.format(final_accuracy / 100))
	print()
	print('<Loss> : {0:.2f}'.format(final_error))
	print()


if __name__ == "__main__":
	main()