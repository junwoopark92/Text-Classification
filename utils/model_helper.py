import numpy as np


def make_train_feed_dict(model, batch, lr):
    """make train feed dict for training"""
    feed_dict = {model.x: batch[0],
                 model.label: batch[1],
                 model.keep_prob: .5,
                 model.LEARNING_RATE: lr}
    return feed_dict


def make_test_feed_dict(model, batch):
    feed_dict = {model.x: batch[0],
                 model.label: batch[1],
                 model.keep_prob: 1.0}
    return feed_dict


def run_train_step(model, sess, batch, lr):
    feed_dict = make_train_feed_dict(model, batch, lr)
    to_return = {
        'train_op': model.train_op,
        'loss': model.loss,
        'global_step': model.global_step
    }
    return sess.run(to_return, feed_dict)



def run_eval_step(model, sess, batch):
    feed_dict = make_test_feed_dict(model, batch)
    prediction = sess.run(model.prediction, feed_dict)
    acc = np.sum(np.equal(prediction, batch[1])) / len(prediction)
    return acc


def get_prediction(model, sess, batch):
    feed_dict = make_test_feed_dict(model, batch)
    prediction = sess.run(model.prediction, feed_dict)
    proba = sess.run(model.proba, feed_dict)
    return prediction, proba


def get_attn_weight(model, sess, batch, lr):
    feed_dict = make_train_feed_dict(model, batch, lr)
    return sess.run(model.alpha, feed_dict)
