"""
For learning a fair representation
"""
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.autograd import Function
from torch.nn import Sequential
from typing import Dict
import torch.nn as nn

from torch.utils.data import Dataset
import numpy as np

from ethicml.algorithms.algorithm import Algorithm


class RepresentationHSIC(Algorithm):

    def run(self, train: Dict[str, pd.DataFrame], test: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        torch.manual_seed(888)

        scaler = StandardScaler()

        scaler.fit(train['x'].values)

        class CustomDataset(Dataset):
            def __init__(self, data, scaler):
                self.x = np.array(scaler.transform(data['x'].values), dtype=np.float32)
                self.y = np.array(data['y'].replace(-1, 0).values, dtype=np.float32)
                self.s = np.array(data['s'].values, dtype=np.float32)
                self.num = data['y'].count().values[0]
                self.size = data['x'].shape[1]

            def __getitem__(self, index):
                return self.x[index], self.s[index], self.y[index]

            def __len__(self):
                return self.num

        data = CustomDataset(train, scaler)

        dataset_loader = torch.utils.data.DataLoader(dataset=data,
                                                     batch_size=500,
                                                     shuffle=False)

        test_data = CustomDataset(test, scaler)
        num = float(data.num)
        size = int(data.size)
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=500, shuffle=False)

        def quadratic_time_HSIC(data_first, data_second, sigma1, sigma2):
            XX = torch.mm(data_first, torch.t(data_first))
            if not torch.all(torch.eq(XX, torch.t(XX))):
                print(XX, torch.t(XX))
            assert torch.all(torch.lt(torch.abs(torch.add(XX, -torch.t(XX))), 1e-6))
            YY = torch.mm(data_second, torch.t(data_second))
            assert torch.all(torch.eq(YY, torch.t(YY)))
            X_sqnorms = torch.diagonal(XX, 0)
            Y_sqnorms = torch.diagonal(YY, 0)

            r = lambda x: x.unsqueeze(0)
            c = lambda x: x.unsqueeze(1)

            gamma = 1 / sigma1#(2 * sigma1 ** 2)
            gamma2 = 1 / sigma2#(2 * sigma2 ** 2)
            # use the second binomial formula
            # Kernel_XX = tf.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
            Kernel_XX = torch.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
            # Kernel_YY = tf.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))
            Kernel_YY = torch.exp(-gamma2 * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))

            n, d = data_first.shape
            H = torch.eye(n) - (1.0/n)*(torch.ones(n,n))

            Kernel_XX_mean = torch.mean(Kernel_XX, 0, keepdim=True)
            Kernel_YY_mean = torch.mean(Kernel_YY, 0, keepdim=True)

            HK = Kernel_XX - Kernel_XX_mean
            # print("Kerel XX", Kernel_XX)
            # print("-", -(H@Kernel_XX + Kernel_XX_mean))
            # print("kernel xx mean", Kernel_XX_mean)
            beep_boop_HK = H@Kernel_XX
            HL = Kernel_YY - Kernel_YY_mean
            beep_boop_HL = H@Kernel_YY

            # n = float(Kernel_YY.size()[0])
            HKf = HK / (n - 1)
            HLf = HL / (n - 1)

            # biased estimate
            hsic = torch.trace(torch.mm(torch.t(HKf), HLf))
            beep_boop_hsic = torch.trace(beep_boop_HK@beep_boop_HL)/(n-1)**2
            # print("HK", HKf)
            # print("HL", HLf)
            # print(hsic, beep_boop_hsic, torch.trace(torch.mm(HKf, HLf)))

            a_hsic = torch.trace(torch.mm(HKf, HLf))
            if torch.eq(hsic, a_hsic):
                print("ok")
            else:
                print("WHOAH")
            return hsic#torch.trace(torch.mm(HKf, HLf))

        class GradReverse(Function):
            @staticmethod
            def forward(ctx, x):
                return x.view_as(x)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output.neg()

        def grad_reverse(x):
            return GradReverse.apply(x)


        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(size, 40),
                    nn.Sigmoid(),
                    nn.Linear(40, 4),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                x = self.encoder(x)
                f = grad_reverse(x)
                return x, f

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Sequential(
                    nn.Linear(4, 10),
                    nn.Sigmoid(),
                    nn.Linear(10, 1),
                )

            def forward(self, x):
                x = self.decoder(x)
                return x

        class Model(nn.Module):
            def __init__(self, enc, dec):
                super().__init__()
                self.enc = enc
                self.dec = dec

            def forward(self, x):
                x = self.enc(x)[0]
                x = self.dec(x)
                return x

        enc = Encoder()
        dec = Decoder()
        model = Model(enc, dec)

        test_1 = torch.tensor([[7.,8.,9.,10.], [10.,11.,12.,13.]])
        test_2 = torch.tensor([[1.],[3.]])

        a = quadratic_time_HSIC(test_1, test_2, 1, 1)
        print(a, quadratic_time_HSIC(test_1, test_2, 1, 1))

        # assert quadratic_time_HSIC(test_1, test_2, 1, 1) == 0.9817

        loss_fn = torch.nn.BCEWithLogitsLoss()

        optimizer1 = torch.optim.Adam(model.parameters())
        optimizera = torch.optim.Adam(model.parameters())
        optimizer2 = torch.optim.Adam(enc.parameters())
        optimizer3 = torch.optim.Adam(dec.decoder.parameters())

        hsic_cost = 1
        for t in range(501):
            # if t % 50 == 0:
            #     hsic_cost *= 10
            if t % 1 == 0:
                for x, s, y in dataset_loader:
                    repr, r_i = enc(x)
                    y_pred = model(x)

                    loss1 = loss_fn(y_pred, y)
                    loss2 = quadratic_time_HSIC(repr, s, 0.4, 0.5)
                    loss3 = quadratic_time_HSIC(repr, y, 0.4, 0.5)

                    loss_a = torch.log(0.0001+loss2) + torch.log(loss3).neg()
                    # loss_a = hsic_cost*loss2

                    optimizer1.zero_grad()
                    loss_a.backward(retain_graph=True)
                    optimizer1.step()

            else:
                for x, s, y in dataset_loader:
                    repr, r_i = enc(x)
                    y_pred = model(x)

                    loss1 = quadratic_time_HSIC(repr, y, 0.4, 0.5).neg()
                    # loss2 = torch.abs(quadratic_time_HSIC(repr, s, 0.4, 0.5))

                    loss_b = loss1

                    optimizera.zero_grad()
                    loss_b.backward()
                    optimizera.step()

            # for x, s, y in test_loader:
            #     repr = enc(x)
            #     y_pred = model(x)
            #
            #     loss2 = quadratic_time_HSIC(repr, s, 0.2, 0.5)
            #
            #     optimizer2.zero_grad()
            #     loss2.backward(retain_graph=True)
            #     optimizer2.step()

            if t % 50 == 0:
                x,s,y = next(iter(dataset_loader))
                repr, r_i = enc(x)
                y_pred = model(x)
                loss1 = quadratic_time_HSIC(repr, y, 0.4, 0.5)
                loss2 = quadratic_time_HSIC(repr, s, 0.4, 0.5)
                loss3 = loss_fn(y_pred, y)

                print(t, loss3, loss1, hsic_cost*loss2, hsic_cost)


        train = []
        test = []

        for x, _, _ in dataset_loader:
            train += enc(x)[0].data.numpy().tolist()

        for x, _, _ in test_loader:
            test += enc(x)[0].data.numpy().tolist()

        # print(results)
        # def step(a):
        #     return 1 if a >= 0.5 else -1
        # results = [step(r) for r in results]
        return (pd.DataFrame(train), pd.DataFrame(test))

    def get_name(self) -> str:
        return "HSIC repr"



# ZZ = torch.mm(repr, torch.t(repr))
# SS = torch.mm(s, torch.t(s))
# Z_sqnorms = torch.diagonal(ZZ)
# S_sqnorms = torch.diagonal(SS)
# r = lambda x: torch.unsqueeze(x, 0)
# c = lambda x: torch.unsqueeze(x, 1)
# z_pair_dist = (-2 * ZZ + c(Z_sqnorms) + r(Z_sqnorms))
# s_pair_dist = (-2 * SS + c(S_sqnorms) + r(S_sqnorms))
# z_pair_dist = torch.relu(z_pair_dist)
# s_pair_dist = torch.relu(s_pair_dist)
# z_sq_dist = torch.sqrt(z_pair_dist)
# s_sq_dist = torch.sqrt(s_pair_dist)
# sigma1 = torch.median(z_sq_dist) + 0.00001
# sigma2 = torch.median(s_sq_dist) + 0.00001