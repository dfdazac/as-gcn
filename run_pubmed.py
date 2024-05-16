from __future__ import division
from __future__ import print_function

# Warning: depending on Pytorch and Tensorflow versions, it's necessary to
# import Pytorch before Tensorflow (even if Pytorch is not needed here),
# to prevent the code from getting stuck.
# See https://github.com/pytorch/pytorch/issues/97580
#     https://github.com/tensorflow/tensorflow/issues/60109

# noinspection PyUnresolvedReferences
import torch
import uuid
import shutil

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import time
import os
from inits import *
from sampler import *
from models import GCNAdapt, GCNAdaptMix

# Set random seed
seed = 123


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'

flags.DEFINE_string('model', 'gcn_adapt', 'Model string.')  # 'gcn', 'gcn_appr'

flags.DEFINE_integer('emb_dim', 128, 'Dimension of embeddings when trained')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_bool('attention', True, 'If use attention mechanism.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('skip', 0, 'If use skip connection.')
flags.DEFINE_float('var', 0.5, 'If use variance reduction.')

flags.DEFINE_string('objective', 'multiclass', 'Training objective: multiclass or multilabel.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('epochs', 300, 'Number of epochs to train.')
flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')

flags.DEFINE_integer('rank', 128, 'The number of nodes per layer.')
flags.DEFINE_integer('max_degree', 32, 'Maximum degree for constructing the adjacent matrix.')
flags.DEFINE_string('gpu', '0', 'The gpu to be applied.')
flags.DEFINE_string('sampler_device', 'cpu', 'The device for sampling: cpu or gpu.')
flags.DEFINE_integer('eval_frequency', 10, 'Number of epochs between evaluations.')

flags.DEFINE_integer('seed', 123, 'Random seed.')
flags.DEFINE_integer('split_id', 0, 'Random seed.')
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

seed = FLAGS.seed
np.random.seed(seed)
tf.set_random_seed(seed)

def main(rank1, rank0):

    # Prepare data
    adj, adj_train, adj_val_train, features, train_features, y_train, y_test, test_index = prepare_dataset(FLAGS.dataset, FLAGS.max_degree, FLAGS.seed, FLAGS.split_id)
    print('preparation done!')
    learn_embeddings = features.shape[1] == 1

    max_degree = FLAGS.max_degree
    num_train = adj_train.shape[0] - 1
    num_nodes = adj.shape[0]

    # num_train = adj_train.shape[0]
    if learn_embeddings:        
        input_dim = FLAGS.emb_dim
        print(f'Dataset has 1 feature,'
              f' switching to learned embeddings with dimension {input_dim}')
    else:
        input_dim = features.shape[1]
    scope = 'test'

    if FLAGS.model == 'gcn_adapt_mix':
        num_supports = 1
        propagator = GCNAdaptMix
        test_supports = [sparse_to_tuple(adj[test_index, :])]
        test_features = [features, features[test_index, :]]
        test_probs = [np.ones(adj.shape[0])]
        layer_sizes = [rank1, 256]
    elif FLAGS.model == 'gcn_adapt':
        num_supports = 2
        propagator = GCNAdapt
        test_supports = [sparse_to_tuple(adj), sparse_to_tuple(adj[test_index, :])]
        test_features = [features, features, features[test_index, :]]
        test_probs = [np.ones(adj.shape[0]), np.ones(adj.shape[0])]
        layer_sizes = [rank0, rank1, 256]
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define placeholders
    placeholders = {
        'batch': tf.placeholder(tf.int32),
        'adj': tf.placeholder(tf.int32, shape=(num_train+1, max_degree)),
        'adj_val': tf.placeholder(tf.float32, shape=(num_train+1, max_degree)),
        'features': tf.placeholder(tf.float32, shape=train_features.shape),
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'prob': [tf.placeholder(tf.float32) for _ in range(num_supports)],
        'features_inputs': [tf.placeholder(tf.float32, shape=(None, input_dim)) for _ in range(num_supports+1)],
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
        'node_ids': [tf.placeholder(tf.int32) for _ in range(num_supports + 1)],
    }

    # Sampling parameters shared by the sampler and model
    with tf.variable_scope(scope):
        w_s = glorot([input_dim, 2], name='sample_weights')

    # Create sampler
    if FLAGS.sampler_device == 'cpu':
        with tf.device('/cpu:0'):
            sampler_tf = SamplerAdapt(placeholders,
                                      input_dim=input_dim,
                                      learn_embeddings=learn_embeddings,
                                      num_nodes=num_nodes,
                                      layer_sizes=layer_sizes, scope=scope)
            features_sampled, support_sampled, p_u_sampled, u_sampled = sampler_tf.sampling(placeholders['batch'])
    else:
        sampler_tf = SamplerAdapt(placeholders, input_dim=input_dim, layer_sizes=layer_sizes, scope=scope)
        features_sampled, support_sampled, p_u_sampled = sampler_tf.sampling(placeholders['batch'])

    # Create model
    model = propagator(placeholders,
                       input_dim=input_dim,
                       learn_embeddings=learn_embeddings,
                       num_nodes=num_nodes,
                       logging=True,
                       name=scope)

    # Initialize session
    config = tf.ConfigProto(device_count={"CPU": 1},
                            inter_op_parallelism_threads=0,
                            intra_op_parallelism_threads=0,
                            allow_soft_placement=True,
                            log_device_placement=False)
    sess = tf.Session(config=config)

    def save_model(sess, saver, checkpoint_path):
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver.save(sess, checkpoint_path)

    # Evaluation function explicitly set to use only CPU
    def evaluate(features, support, prob_norm, labels, mask, placeholders, checkpoint_path):
        # Create a new CPU-only session configuration
        config_cpu = tf.ConfigProto(
            device_count={"GPU": 0},  # Disable GPU usage
            allow_soft_placement=True,
            log_device_placement=False
        )

        with tf.Session(config=config_cpu) as cpu_sess:
            # Import the current graph into the new CPU-only session
            cpu_sess.graph.as_default()

            # Ensure variables from the original session are loaded into the CPU-only session
            # Restore model parameters from the correct checkpoint
            saver.restore(cpu_sess, checkpoint_path)

            # Perform the evaluation using the CPU
            t_test = time.time()
            feed_dict_val = construct_feed_dict_with_prob(features, support, prob_norm, labels, mask, placeholders)

            if learn_embeddings:
                feed_dict_val.update({placeholders['node_ids'][i]: nids.squeeze() for i, nids in enumerate(features)})
                for p in placeholders['features_inputs']:
                    feed_dict_val.pop(p)

            outs_val = cpu_sess.run([model.loss, model.accuracy, model.f1_score], feed_dict=feed_dict_val)

        return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)

    # Init variables
    sess.run(tf.global_variables_initializer(), feed_dict={placeholders['adj']: adj_train,
                                                           placeholders['adj_val']: adj_val_train,
                                                           placeholders['features']: train_features})

    # Prepare training
    saver = tf.train.Saver()
    checkpoint_path = os.path.join('checkpoints', f'{FLAGS.dataset}-{FLAGS.hidden1}-{FLAGS.learning_rate}-{FLAGS.seed}-{uuid.uuid4()}', 'model.ckpt')
    # save_dir = "tmp/" + FLAGS.dataset + '_' + str(FLAGS.skip) + '_' + str(FLAGS.var) + '_' + str(FLAGS.gpu)
    acc_val = []
    acc_train = []
    train_time = []
    train_time_sample = []
    max_acc = 0
    t = time.time()
    # Train model
    for epoch in range(FLAGS.epochs):

        sample_time = 0
        t1 = time.time()

        for batch in iterate_minibatches_listinputs([y_train, np.arange(num_train)], batchsize=256, shuffle=True):
            [y_train_batch, train_batch] = batch

            if sum(train_batch) < 1:
                continue
            ts = time.time()
            features_inputs, supports, probs, node_ids = sess.run([features_sampled, support_sampled, p_u_sampled, u_sampled],
                                                        feed_dict={placeholders['batch']:train_batch})
            sample_time += time.time()-ts

            # Construct feed dictionary
            feed_dict = construct_feed_dict_with_prob(features_inputs, supports, probs, y_train_batch, [],
                                                      placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            if learn_embeddings:
                feed_dict.update({placeholders['node_ids'][i]: nids for i, nids in enumerate(node_ids)})

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
            acc_train.append(outs[-2])

        train_time_sample.append(time.time()-t1)
        train_time.append(time.time()-t1-sample_time)

        # Validation
        if (epoch + 1) % FLAGS.eval_frequency == 0:
            save_model(sess, saver, checkpoint_path)
            cost, acc, f1, duration = evaluate(test_features, test_supports, test_probs, y_test, [], placeholders, checkpoint_path)
            acc_val.append(acc)

            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                  "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
                  "val_acc=", "{:.5f}".format(acc),
                  "val_f1=",  "{:.5f}".format(f1),  "time=", "{:.5f}".format(train_time_sample[epoch]))
        else:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]))

    train_duration = np.mean(np.array(train_time_sample))

    test_cost, test_acc, test_f1, test_duration = evaluate(test_features, test_supports, test_probs, y_test, [], placeholders, checkpoint_path)
    print("rank1 = {}".format(rank1), "rank0 = {}".format(rank0), "test_loss=", "{:.5f}".format(test_cost),
          "test_acc=", "{:.5f}".format(test_acc),
          "test_f1=", "{:.5f}".format(test_f1), "training time per epoch=", "{:.5f}".format(train_duration))

    checkpoint_dir = os.path.dirname(checkpoint_path)
    print(f'Cleaning up, deleting {checkpoint_dir}')
    shutil.rmtree(checkpoint_dir)


if __name__ == "__main__":

    print("DATASET:", FLAGS.dataset)
    main(FLAGS.rank,FLAGS.rank)
