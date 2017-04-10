import math
import os
import random
import sys
import time
import logging
import pickle
import tensorflow as tf
import numpy as np
from enum import Enum
import scipy.spatial.distance
from collections import deque
from tensorflow.python import debug as tf_debug
from tensorflow.python.training import saver as save_mod
from tensorflow.python.ops import embedding_ops

import data_utils
import autoencode_model
from autoencode_model import State

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 50,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 700, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size",700, "Word Embedding Dimension.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("source_vocab_size", 90000, "Source vocabulary size.")
tf.app.flags.DEFINE_integer("target_vocab_size", 90000, "Target vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/home/cliffrwong/Documents/data/opensubtitles/en-de/lowerZero3/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/home/cliffrwong/Documents/data/opensubtitles/en-de/lowerZero3/checkpoint3/", "Training directory.")
tf.app.flags.DEFINE_bool("source_lang", "en", "Language.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 1000,
                            "How many training steps to do per checkpoint.")
FLAGS = tf.app.flags.FLAGS

_buckets = [(7, 7), (15, 15), (25, 25), (50,50)]

def read_data(data_dir, max_size=None):
  """Read data from source and target files and put into buckets.
  """
  print("Reading data in {0}".format(data_dir))
  data_set = [[] for _ in _buckets]

  langFiles = [file for file in os.listdir(data_dir)]

  # Iterate over each english and foreign text pair
  for i in range(len(langFiles))[::2]:
    file1 = langFiles[i]
    file2 = langFiles[i+1]
    f1_name, f1_extension = os.path.splitext(file1)
    f2_name, f2_extension = os.path.splitext(file2)

    # Check that they are language file pairs
    if f1_name != f2_name:
      raise ValueError("Files {0} and {1} are not language pairs"
                       .format(f1_name, f2_name))
    if f1_extension == ".en":
      file_eng = file1
      file_source = file2
    else if f2_extension === ".en":
      file_eng = file2
      file_source = file1
    else:
      raise ValueError("Missing English Text File")

    with open(file_eng, "r") as en_file:
      with open(file_source, "r") as source_file:
        source, target = source_file.readline(), en_file.readline()
        counter = 0
        while source and target and (not max_size or counter < max_size):
          counter += 1
          if counter % 100000 == 0:
            print("  reading data line %d" % counter)
            sys.stdout.flush()
          source_ids = [int(x) for x in source.split()]
          target_ids = [int(x) for x in target.split()]
          for bucket_id, (source_size, target_size) in enumerate(_buckets):
            if len(source_ids) < source_size and len(target_ids) < target_size:
              data_set[bucket_id].append([target_ids, source_ids+[data_utils.EOS_ID]])
              data_set[bucket_id].append([source_ids, source_ids+[data_utils.EOS_ID]])
              break
          source, target = source_file.readline(), target_file.readline()
  return data_set

def create_model(session, state):
  """Create autoencoder model and initialize or load parameters in session."""
  # dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  dtype = tf.float32
  model = autoencode_model.AutoencodeModel(
      FLAGS.source_vocab_size,
      FLAGS.target_vocab_size,
      FLAGS.embedding_size,
      _buckets,
      FLAGS.size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      state=state,
      dtype=dtype)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and save_mod.checkpoint_exists(ckpt.model_checkpoint_path):
      print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
      print("Created model with fresh parameters.")
      session.run(tf.global_variables_initializer())
  return model

def train():
  with tf.Session() as sess:
    # Create model.
    print("Creating {0} layers of {1} units."
          .format(FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, State.TRAIN)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)

    train_set = read_data(FLAGS.data_dir, FLAGS.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]
    
    dev_set = read_data(source_dev, target_dev)
    dev_bucket_sizes = [len(dev_set[b]) for b in range(len(_buckets))]
    dev_total_size = float(sum(dev_bucket_sizes))
    dev_buckets_frac = [i/dev_total_size for i in dev_bucket_sizes]
    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in range(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, State.TRAIN)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))

        # Decrease learning rate if no improvement was seen over last 2 times.
        if len(previous_losses) > 1 and loss > max(previous_losses[-2:]):
            sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        dev_loss = 0.0
        # Run evals on development set and print their perplexity.
        for bucket_id in range(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, State.TEST)
          dev_loss += eval_loss*dev_buckets_frac[bucket_id]
          eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
              "inf")
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
          
        # Write to summary writer
        dev_ppx = math.exp(float(dev_loss)) if dev_loss < 10 else math.exp(10)
        print(" eval: all bucket perplexity %.2f" % (dev_ppx))
        sys.stdout.flush()

        summary_str = tf.Summary(value=[
          tf.Summary.Value(tag="dev. perplexity", simple_value=dev_ppx),
          tf.Summary.Value(tag="train perplexity", simple_value=perplexity)])
        summary_writer.add_summary(summary_str, current_step)

def getVocab():
    source_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab{0}.{1}".format(
                                    FLAGS.source_vocab_size, FLAGS.source_lang))
    target_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab{0}.{1}".format(
                                    FLAGS.target_vocab_size, FLAGS.target_lang))
    source_vocab, rev_source_vocab = data_utils.initialize_vocabulary(source_vocab_path)
    target_vocab, rev_target_vocab = data_utils.initialize_vocabulary(target_vocab_path)
    return source_vocab, target_vocab, rev_target_vocab

# Tensor("proj_w/read:0", shape=(90500, 1000), dtype=float32)
# Tensor("proj_b/read:0", shape=(90500,), dtype=float32)
# Tensor("embedding_tied_rnn_seq2seq/embedding/read:0", shape=(90500, 1000), dtype=float32)
# Tensor("embedding_tied_rnn_seq2seq/combined_tied_rnn_seq2seq/tied_rnn_seq2seq/basic_lstm_cell/weights/read:0", shape=(2000, 4000), dtype=float32)
# Tensor("embedding_tied_rnn_seq2seq/combined_tied_rnn_seq2seq/tied_rnn_seq2seq/basic_lstm_cell/biases/read:0", shape=(4000,), dtype=float32)

def saveParmas():
  with tf.Session() as sess:
    # Create model.
    model = create_model(sess, State.TRAIN)
    for tf_var in tf.trainable_variables():
      print(tf_var)
    # var = "embedding_tied_rnn_seq2seq/embedding"
    # # var = "embedding_tied_rnn_seq2seq/combined_tied_rnn_seq2seq/tied_rnn_seq2seq/basic_lstm_cell/weights"
    # var = "embedding_tied_rnn_seq2seq/combined_tied_rnn_seq2seq/tied_rnn_seq2seq/basic_lstm_cell/biases"
    # weight_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, var)[0]
    # weight_var_value = sess.run(weight_var)
    # with open(FLAGS.data_dir+"lstm_bias.pkl", 'wb') as fout:
    #   pickle.dump(weight_var_value, fout)

def main(_):
    train()
    # saveParmas()

if __name__ == "__main__":
  tf.app.run()
