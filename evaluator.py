import tensorflow as tf
from scipy.sparse import lil_matrix, coo_matrix

import copy
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
import numpy as np

class RecallEvaluator(object):
    def __init__(self, model, train_user_item_matrix, test_user_item_matrix):
        """
        Create a evaluator for recall@K evaluation
        :param model: the model we are going to evaluate
        :param train_user_item_matrix: the user-item pairs used in the training set. These pairs will be ignored
               in the recall calculation
        :param test_user_item_matrix: the held-out user-item pairs we make prediction against
        """
        self.model = model
        self.train_user_item_matrix = lil_matrix(train_user_item_matrix)
        self.test_user_item_matrix = lil_matrix(test_user_item_matrix)
        n_users = train_user_item_matrix.shape[0]
        self.user_to_test_set = {u: set(self.test_user_item_matrix.rows[u])
                                 for u in range(n_users) if self.test_user_item_matrix.rows[u]}

        if self.train_user_item_matrix is not None:
            self.user_to_train_set = {u: set(self.train_user_item_matrix.rows[u])
                                      for u in range(n_users) if self.train_user_item_matrix.rows[u]}
            self.max_train_count = max(len(row) for row in self.train_user_item_matrix.rows)
        else:
            self.max_train_count = 0


    def eval_roc(self, sess, users, test):
        test_mat = copy.copy(test)
        test_mat = coo_matrix(test_mat).toarray()
        scoresMat = sess.run(self.model.item_scores, {self.model.score_user_ids: users})

        n_items = self.train_user_item_matrix.shape[1]
        n_aupr_values = np.zeros([len(users), 1])

        items_set = list(range(n_items))
        t_num = -1

        for user_id in users:
            t_num = t_num + 1
            train_set = self.user_to_train_set.get(user_id, set())

            t_label = copy.copy(test_mat[user_id, :])

            items_set_t = copy.copy(items_set)
            for i in train_set:
                items_set_t.remove(i)

            y_true = copy.copy(t_label)
            y_true = y_true[items_set_t]
            y_true = np.array(y_true)
            y_score = copy.copy(scoresMat[t_num, :])
            y_score = y_score[items_set_t]
            y_score = np.array(y_score)

            # AUPR;
            precision_r, recall_r, thresholds_r = precision_recall_curve(y_true, y_score)
            aupr_value = metrics.auc(recall_r, precision_r)
            n_aupr_values[t_num] = aupr_value

        m_n_aupr_values = np.mean(n_aupr_values)

        return m_n_aupr_values

    def eval(self, sess, users, k=50):
        """
        Compute the Top-K recall for a particular user given the predicted scores to items
        :param users: the users to eval the recall
        :param k: compute the recall for the top K items
        :return: recall@K
        """
        # compute the top (K +  Max Number Of Training Items for any user) items for each user

        _, user_tops = sess.run(tf.nn.top_k(self.model.item_scores, k + self.max_train_count),
                                {self.model.score_user_ids: users})
        recalls = []
        precisions = []

        for user_id, tops in zip(users, user_tops):
            train_set = self.user_to_train_set.get(user_id, set())
            test_set = self.user_to_test_set.get(user_id, set())

            top_n_items = 0
            hits = 0
            for i in tops:
                # ignore item in the training set
                if i in test_set:
                    hits += 1
                elif i in train_set:
                    continue

                top_n_items += 1
                if top_n_items == k:
                    break
            recalls.append(hits / float(len(test_set)))
            precisions.append(hits / float(top_n_items))
        return recalls, precisions
