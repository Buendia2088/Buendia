from Backend import MpModel, pretty
from Frontend import McModel
from Selection_bias import Multi_env_selection_bias, generate_test, modified_Multi_env_selection_bias
import torch
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HRM:
    def __init__(self, front_params, back_params, X, y, test_X=None, test_y=None):
        self.X = X.to(device)
        self.y = y.to(device)
        if test_X is not None:
            self.test_X = [test_X.to(device)]
            self.test_y = [test_y.to(device)]
        else:
            self.test_X = None
            self.test_y = None
        self.frontend = McModel(front_params['num_clusters'], self.X, self.y)
        self.backend = MpModel(input_dim=back_params['input_dim'],
                               output_dim=back_params['output_dim'],
                               sigma=back_params['sigma'],
                               lam=back_params['lam'],
                               alpha=back_params['alpha'],
                               hard_sum=back_params['hard_sum'])
        self.domains = None
        self.weight = torch.zeros(self.X.shape[1], dtype=torch.float32, device=device)

    def solve(self, iters):
        self.density_result = None
        density_record = []
        flag = False
        for i in range(iters):
            environments, self.domains = self.frontend.cluster(self.weight, self.domains, flag)
            weight, density = self.backend.train(environments, epochs=6000)
            density_record.append(density)
            self.density_result = density
            self.weight = density
            self.backend.lam *= 1.05
            self.backend.alpha *= 1.05
            print('Selection Ratio is %s' % self.weight)
        with open('./save.txt', 'a+') as f:
            print('Density results:')
            for i in range(len(density_record)):
                print("Iter %d Density %s" % (i, pretty(density_record[i])))
                f.writelines(pretty(density_record[i]) + '\n')
        return self.weight

    def test(self, test_envs):
        test_accs = []
        self.backend.backmodel.eval()
        self.backend.featureSelector.eval()
        for i in range(len(test_envs)):
            pred = self.backend.single_forward(test_envs[i][0].to(device))
            error = torch.sqrt(torch.mean((pred.reshape(test_envs[i][1].shape).to(device) - test_envs[i][1].to(device)) ** 2))
            test_accs.append(error.data)
        print(pretty(test_accs))
        self.backend.backmodel.train()
        self.backend.featureSelector.train()
        return test_accs

def combine_envs(envs):
    X = []
    y = []
    for env in envs:
        X.append(env[0])
        y.append(env[1])
    X = torch.cat(X, dim=0).to(device)
    y = torch.cat(y, dim=0).to(device)
    return X.reshape(-1, X.shape[1]), y.reshape(-1, 1)

def seed_torch(seed=2018):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class EmpiricalRiskMinimizer(object):
    def __init__(self, X, y, mask):
        x_all = X.cpu().numpy()
        y_all = y.cpu().numpy()
        self.mask = mask
        x_all = x_all[:, self.mask.cpu().numpy()]
        lr_model = LinearRegression(fit_intercept=False)
        lr_model.fit(x_all, y_all)
        w = lr_model.coef_
        self.w = torch.Tensor(w).to(device)

    def solution(self):
        return self.w

    def test(self, X, y):
        X = X.cpu().numpy()
        X = X[:, self.mask.cpu().numpy()]
        y = y.cpu().numpy()
        err = np.mean((X.dot(self.w.cpu().numpy().T) - y) ** 2.).item()
        return np.sqrt(err)

if __name__ == "__main__":
    all_weights = torch.zeros(10, dtype=torch.float32, device=device)
    average = 0.0
    std = 0.0
    seeds = 10
    average_error_list = torch.zeros(10, dtype=torch.float32, device=device)
    for seed in range(0, seeds):
        seed_torch(seed)
        print("---------------seed = %d------------------" % seed)
        environments, _ = Multi_env_selection_bias()
        X, y = combine_envs(environments)
        front_params = {'num_clusters': 3}
        back_params = {'input_dim': X.shape[1],
                       'output_dim': 1,
                       'sigma': 0.1,
                       'lam': 0.1,
                       'alpha': 1000.0,
                       'hard_sum': 10,
                       'overall_threshold': 0.20}
        whole_iters = 5
        model = HRM(front_params, back_params, X, y)
        result_weight = model.solve(whole_iters)
        all_weights += result_weight
        mask = torch.where(result_weight > back_params['overall_threshold'])[0]
        evaluate_model = EmpiricalRiskMinimizer(X, y, mask)
        testing_envs = generate_test()
        testing_errors = []
        for [X_test, y_test] in testing_envs:
            testing_errors.append(evaluate_model.test(X_test.to(device), y_test.to(device)))
        testing_errors = torch.Tensor(testing_errors).to(device)
        print(testing_errors)
        average += torch.mean(testing_errors) / seeds
        std += torch.std(testing_errors) / seeds
        average_error_list += testing_errors / seeds
        print(average_error_list)
    print(average)
    print(std)
    print(all_weights)
