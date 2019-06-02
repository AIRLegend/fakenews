from bert.run_classifier import convert_examples_to_features, InputExample, input_fn_builder
from bert.tokenization import FullTokenizer
from bert.optimization import create_optimizer


import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


class BertClassifier(object):

    def __init__(self, ):
        self.BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
        self.tokenizer = None
        self.model = None

        self.max_seq_len = 128
        self.tokenizer = self.__create_tokenizer_from_hub_module()

    def __create_tokenizer_from_hub_module(self):
        if self.tokenizer is not None:
            return self.tokenizer

        with tf.Graph().as_default():
            bert_module = hub.Module(self.BERT_MODEL_HUB)
            tokenization_info = bert_module(signature="tokenization_info",
                                            as_dict=True)
            with tf.Session() as sess:
                vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"], 
                            tokenization_info["do_lower_case"]])

                tokenizer = FullTokenizer(vocab_file=vocab_file,
                                     do_lower_case=do_lower_case)

                self.tokenizer = tokenizer
                return tokenizer

    def __create_features(self, pd_dataset, label_list,
                        max_seq_len, tokenizer,
                        data_column, label_column):
        input_examples = pd_dataset.apply(lambda x: InputExample(guid=None,
                                          text_a=x[data_column],
                                          text_b=None,
                                          label=x[label_column]), axis=1)
        return convert_examples_to_features(input_examples, label_list,
                                            max_seq_len, tokenizer)

    def __create_model(self, input_ids, input_mask, segment_ids,
                       labels, num_labels, is_predicting=True):

        bert_module = hub.Module(
            self.BERT_MODEL_HUB,
            trainable=True)

        bert_inputs = dict(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids)

        bert_outputs = bert_module(
            inputs=bert_inputs,
            signature="tokens",
            as_dict=True)

        output_layer = bert_outputs["pooled_output"]

        hidden_size = output_layer.shape[-1].value

        # Create our own layer to tune for politeness data.
        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            # Dropout helps prevent overfitting
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            # Convert labels into one-hot encoding
            one_hot_labels = tf.one_hot(labels, depth=num_labels,
                                        dtype=tf.float32)

            predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1,
                                          output_type=tf.int32))
            # If we're predicting, we want predicted labels
            # and the probabiltiies.
            if is_predicting:
                return (predicted_labels, log_probs)

            # If we're train/eval, compute loss between predicted and actual label
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return (loss, predicted_labels, log_probs)

    def __model_fn_builder(self, num_labels, learning_rate,
                           num_train_steps,
                           num_warmup_steps):
        """Returns `model_fn` closure for TPUEstimator."""

        def model_fn(features, labels, mode, params):
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]

            is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

            # TRAIN and EVAL
            if not is_predicting:

                (loss, predicted_labels, log_probs) = self.__create_model(
                    input_ids,
                    input_mask, segment_ids, label_ids, num_labels,
                    is_predicting=is_predicting
                )

                train_op = create_optimizer(
                    loss, learning_rate, num_train_steps, num_warmup_steps,
                    use_tpu=False)

                # Calculate evaluation metrics.
                def metric_fn(label_ids, predicted_labels):
                    accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                    f1_score = tf.contrib.metrics.f1_score(
                        label_ids,
                        predicted_labels)
                    auc = tf.metrics.auc(
                        label_ids,
                        predicted_labels)
                    recall = tf.metrics.recall(
                        label_ids,
                        predicted_labels)
                    precision = tf.metrics.precision(
                        label_ids,
                        predicted_labels)
                    true_pos = tf.metrics.true_positives(
                        label_ids,
                        predicted_labels)
                    true_neg = tf.metrics.true_negatives(
                        label_ids,
                        predicted_labels)
                    false_pos = tf.metrics.false_positives(
                        label_ids,
                        predicted_labels)
                    false_neg = tf.metrics.false_negatives(
                        label_ids,
                        predicted_labels)
                    return {
                        "eval_accuracy": accuracy,
                        "f1_score": f1_score,
                        "auc": auc,
                        "precision": precision,
                        "recall": recall,
                        "true_positives": true_pos,
                        "true_negatives": true_neg,
                        "false_positives": false_pos,
                        "false_negatives": false_neg
                    }

                eval_metrics = metric_fn(label_ids, predicted_labels)

                if mode == tf.estimator.ModeKeys.TRAIN:
                    return tf.estimator.EstimatorSpec(mode=mode,
                                                      loss=loss,
                                                      train_op=train_op)
                else:
                    return tf.estimator.EstimatorSpec(mode=mode,
                                                      loss=loss,
                                                      eval_metric_ops=eval_metrics)
            else:
                (predicted_labels, log_probs) = self.__create_model(
                    input_ids,
                    input_mask, segment_ids, label_ids, num_labels,
                    is_predicting=is_predicting
                )

                predictions = {
                    'probabilities': log_probs,
                    'labels': predicted_labels
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # Return the actual model function in the closure
        return model_fn

    def __create_estimator(self, label_list, lr, batch_size, n_train, n_warm):
        model_fn = self.__model_fn_builder(
            num_labels=len(label_list),
            learning_rate=lr,
            num_train_steps=n_train,
            num_warmup_steps=n_warm
        )

        estimator = tf.estimator.Estimator(model_fn=model_fn,
                                           params={"batch_size": batch_size})

        return estimator, model_fn

    def train(self, train, test, data_col, lbl_col,
              batch_size=32,
              lr=2e-5,
              epochs=3,
              warmup=0.1):
        """
        Trains a BERT based model to classify fake/true news

        Params:

        train -- Pandas dataframe to train with at least (text, type) columns
        test -- Pandas dataframe to evaluate with at least (text, type) columns
        data_col -- Name of the Text column
        lbl_col -- Name of the Type column
        batch_size -- Training batch size (default = 32)
        epochs -- Epochs to train (default = 3)
        warmup -- Warmup percent to train. Defined in BERT paper (default = 0.1)

        Returns:

        Rictionary with evaluation results
        """
        label_list = train[lbl_col].unique().tolist()
        tokenizer = self.__create_tokenizer_from_hub_module()

        train_features = self.__create_features(
            train, label_list,
            self.max_seq_len, tokenizer, data_col, lbl_col
        )
        test_features = self.__create_features(
            test, label_list,
            self.max_seq_len, tokenizer, data_col, lbl_col
        )

        num_train_steps = int(len(train_features) / batch_size * epochs)
        num_warmup_steps = int(num_train_steps * warmup)

        estimator, model_fn = self.__create_estimator(
            label_list,
            lr,
            batch_size,
            num_train_steps,
            num_warmup_steps
        )

        train_input_fn = input_fn_builder(
            features=train_features,
            seq_length=self.max_seq_len,
            is_training=True,
            drop_remainder=False)

        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

        test_input_fn = input_fn_builder(
            features=test_features,
            seq_length=self.max_seq_len,
            is_training=False,
            drop_remainder=False)

        result_dict = estimator.evaluate(input_fn=test_input_fn, steps=None)

        self.model = estimator

        return result_dict

    def predict(self, df):
        """
        Predicts over a pandas dataframe.

        Params:
        df -- Pandas dataframe to train with at least (text, type) columns

        Returns:

        Dictionary with predicted labels and probabilities.
        """
        # TODO: REMOVE type column

        tokenizer = self.__create_tokenizer_from_hub_module()
        label_list = test_other[LABEL_COLUMN].unique().tolist()
        #label_list = [0, 1]
        test_features = self.__create_features(
            df, label_list,
            self.max_seq_len, tokenizer, 'text', 'type'
        )

        preds = []
        if type(self.model) == tf.estimator.Estimator:
            # Is trained
            input_fn = input_fn_builder(
                features=test_features,
                seq_length=self.max_seq_len,
                is_training=False,
                drop_remainder=False)
            pred = self.model.predict(input_fn=input_fn)
            for p in pred:
                preds.append(p)
        else:
            # Is loaded from a SavedModel
            # Format inputs
            inpu = {
                'label_ids': np.array([x.label_id for x in test_features]).reshape(-1,),
                'input_ids': np.array([x.input_ids for x in test_features]).reshape(-1, self.max_seq_len),
                'input_mask': np.array([x.input_mask for x in test_features]).reshape(-1, self.max_seq_len),
                'segment_ids': np.array([x.segment_ids for x in test_features]).reshape(-1, self.max_seq_len)
            }
            preds = self.model(inpu)

        return preds

    def save_model(self, directory):
        """Saves model in the specified path"""
        def serving_input_fn():
            label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
            input_ids = tf.placeholder(tf.int32, [None, self.max_seq_len], name='input_ids')
            input_mask = tf.placeholder(tf.int32, [None, self.max_seq_len], name='input_mask')
            segment_ids = tf.placeholder(tf.int32, [None, self.max_seq_len], name='segment_ids')
            input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
                'label_ids': label_ids,
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
            })()
            return input_fn

        self.model._export_to_tpu = False  # this is important
        self.model.export_savedmodel(directory, serving_input_fn)

    def load_model(self, directory):
        """
        Restores a previously saved model.

        Params:

        directory -- Folder in which is the .pb file
        """
        with tf.Session() as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], directory)
            self.model = tf.contrib.predictor.from_saved_model(directory)
