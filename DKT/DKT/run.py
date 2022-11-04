from .data_divide import divide_data
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error
from .data_loader import TrainDataLoader, ValTestDataLoader, GeneratroDataLoader
from .model import Net
from .utils import CommonArgParser
from EduKTM import KTM
import tqdm
import os
import logging

storage = {}
user_list = []
path = os.path.join('data')


def same_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_auc(all_target, all_pred):
    return roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return accuracy_score(all_target, all_pred)


def compute_rmse(all_target, all_pred):
    return np.sqrt(mean_absolute_error(all_target, all_pred))


def compute_mae(all_target, all_pred):
    return mean_absolute_error(all_target, all_pred)


def process_raw_pred(user_id, raw_question_matrix, raw_pred, num_questions: int, isStorage=False) -> tuple:
    # nonzero:返回一个包含输入input中非零元素索引的张量。输出张量中的每行包含输入中非零元素的索引
    questions = torch.nonzero(raw_question_matrix)[1:, 1] % num_questions
    # 把是1的下标取出来，然后再对num_questions取模
    length = questions.shape[0]
    # shape: length-1, num_questions  ,length表示该组step有多少个非零的做题序列
    pred = raw_pred[: length]
    if (isStorage):
        if (pred.shape[0] == 0):
            pass
        else:
            if storage.get(user_id.numpy().tolist()[0]) is None:
                storage[user_id.numpy().tolist()[0]
                        ] = pred[:].detach().numpy().tolist()
            else:
                storage[user_id.numpy().tolist()[0]
                        ] += pred[:].detach().numpy().tolist()

    pred = pred.gather(1, questions.view(-1, 1)
                       ).flatten()  # 从预测的结果中取出所答知识点的预测结果
    truth = torch.nonzero(raw_question_matrix)[1:, 1] // num_questions
    return pred, truth


class DKT(KTM):
    def __init__(self, num_questions, hidden_size, num_layers):
        super(DKT, self).__init__()
        self.num_questions = num_questions
        self.dkt_model = Net(num_questions, hidden_size, num_layers)

    def train(self, train_data, test_data, *, epoch: int, lr=0.002):
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(self.dkt_model.parameters(), lr)
        # resu=None

        best_auc = 0
        best_acc = 0
        best_mae = 1.0
        best_rmse = 1.0
        isStorage = False

        for e in range(epoch):
            losses = []
            for _, (batch, user_list) in enumerate(train_data):
                # print(batch.shape)
                # input()
                integrated_pred = self.dkt_model(batch)
                # print(integrated_pred.shape)
                # print(integrated_pred)
                # input()
                batch_size = batch.shape[0]
                loss = torch.Tensor([0.0])
                for student in range(batch_size):
                    pred, truth = process_raw_pred(
                        user_list[student], batch[student], integrated_pred[student], self.num_questions, isStorage)
                    if pred.shape[0] != 0:
                        loss += loss_function(pred, truth.float())

                optimizer.zero_grad()  # 参数清零
                loss.backward()  # 反向传播，将整个计算过程进行微分操作
                optimizer.step()  # 参数进行更新

                losses.append(loss.mean().item())
                # resu = integrated_pred[-1][-1]
            auc, acc, mae, rmse = self.eval(test_data)
            if best_acc < acc:
                best_acc = acc
            if best_auc < auc:
                best_auc = auc
            print("[Epoch %d] LogisticLoss: %.6f" %
                  (e, float(np.mean(losses))), ' auc:', auc, ' acc:', acc, ' best_auc:', best_auc, ' best_acc:', best_acc)
            self.save(os.path.join(path, 'model', 'epoch' + str(e)))

    def eval(self, test_data) -> float:
        self.dkt_model.eval()
        y_pred = torch.Tensor([])
        y_truth = torch.Tensor([])
        for batch in tqdm.tqdm(test_data, "evaluating"):
            integrated_pred = self.dkt_model(batch)
            batch_size = batch.shape[0]
            for student in range(batch_size):
                pred, truth = process_raw_pred(None,
                                               batch[student], integrated_pred[student], self.num_questions, False)
                y_pred = torch.cat([y_pred, pred])
                y_truth = torch.cat([y_truth, truth])
        return compute_auc(y_truth.detach().numpy(), y_pred.detach().numpy()), compute_accuracy(y_truth.detach().numpy(), y_pred.detach().numpy()), compute_mae(y_truth.detach().numpy(), y_pred.detach().numpy()), compute_rmse(y_truth.detach().numpy(), y_pred.detach().numpy())

    def save(self, filepath):
        torch.save(self.dkt_model.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.dkt_model.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)


def run():
    args = CommonArgParser().parse_args()
    same_seeds(2022)
    divide_data(args.division_ratio)
    dkt = DKT(args.knowledge_n, args.hidden_size, args.num_layers)
    te = ValTestDataLoader(args.batch_size)
    t = TrainDataLoader(args.batch_size)
    dkt.train(t.train_loader, te.test_loader, epoch=args.epoch_n, lr=args.lr)
    return storage
