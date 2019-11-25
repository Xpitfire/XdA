import torch


class XdA(torch.nn.Module):

    def __init__(self, shape, alpha=0.01, sampling_no=64):
        super(XdA, self).__init__()
        self.norm = torch.nn.LayerNorm(shape[1:], elementwise_affine=False)
        self.alpha = alpha
        self.param_shape = shape
        self.sampling_no = sampling_no
        self.init = [True for _ in range(shape[0])]
        self.phi = torch.nn.ParameterList()
        self.mu = []
        self.prediction_errors = 0
        for _ in range(shape[0]):
            self.phi.append(torch.nn.Parameter(torch.zeros(shape[1:]), requires_grad=True))
            self.mu.append(None)

    def _mask(self, x, t):
        return (x >= self.phi[t].expand_as(x)).float()

    def _check_init(self, x, t):
        if self.init[t]:
            self.phi[t].data = x.mean(0).data
            self.mu[t] = x.mean(0).data
            self.init[t] = False

    def _update_mu(self, x, t):
        if self.training:
            self.mu[t].data = ((1 - self.alpha) * self.mu[t].data + self.alpha * x.mean(0)).data

    def reg(self, t=0):
        return torch.sum(torch.abs(self.phi[t] - self.mu[t]))

    def forward(self, x, t=0):
        x = self.norm(x)
        self._check_init(x, t)
        self._update_mu(x, t)
        mask = self._mask(x, t)
        x *= mask
        return x


class XdAv2(torch.nn.Module):

    def __init__(self, shape, alpha=0.01, sampling_no=64):
        super(XdAv2, self).__init__()
        self.norm = torch.nn.LayerNorm(shape[1:], elementwise_affine=False)
        self.alpha = alpha
        self.param_shape = shape
        self.sampling_no = sampling_no
        self.init = [True for _ in range(shape[0])]
        self.phi = torch.nn.ParameterList()
        self.mu = []
        self.prediction_errors = 0
        self.zero = torch.zeros(1).cuda()
        for _ in range(shape[0]):
            self.phi.append(torch.nn.Parameter(torch.zeros(shape[1:]), requires_grad=True))
            self.mu.append(None)

    def _check_init(self, x, t):
        if self.init[t]:
            self.phi[t].data = x.mean(0).data
            self.mu[t] = x.mean(0).data
            self.init[t] = False

    def _update_mu(self, x, t):
        if self.training:
            self.mu[t].data = ((1 - self.alpha) * self.mu[t].data + self.alpha * x.mean(0)).data

    def reg(self, t=0):
        return 0

    def forward(self, x, t=0):
        x = self.norm(x)
        self._check_init(x, t)
        self._update_mu(x, t)
        x = torch.max(x - self.phi[t].expand_as(x), self.zero)
        return x


class RegCosXdA(torch.nn.Module):

    def __init__(self, shape, alpha=0.01, sampling_no=64):
        super(RegCosXdA, self).__init__()
        self.norm = torch.nn.LayerNorm(shape[1:], elementwise_affine=False)
        self.alpha = alpha
        self.param_shape = shape
        self.sampling_no = sampling_no
        self.init = [True for _ in range(shape[0])]
        self.phi = torch.nn.ParameterList()
        self.mu = []
        self.prediction_errors = 0
        for _ in range(shape[0]):
            self.phi.append(torch.nn.Parameter(torch.zeros(shape[1:]), requires_grad=True))
            self.mu.append(None)

    def _mask(self, x, t):
        return (x >= self.phi[t].expand_as(x)).float()

    def _check_init(self, x, t):
        if self.init[t]:
            self.phi[t].data = x.mean(0).data
            self.mu[t] = x.mean(0).data
            self.init[t] = False

    def _update_mu(self, x, t):
        if self.training:
            self.mu[t].data = ((1 - self.alpha) * self.mu[t].data + self.alpha * x.mean(0)).data

    def reg(self, t=0):
        return torch.nn.functional.cosine_similarity(self.phi[t].view(-1), self.mu[t].view(-1), 0)

    def forward(self, x, t=0):
        x = self.norm(x)
        self._check_init(x, t)
        self._update_mu(x, t)
        mask = self._mask(x, t)
        x *= mask
        return x


class AXdA(XdA):

    def __init__(self, shape, alpha=0.01, sampling_no=64):
        super(AXdA, self).__init__(shape, alpha, sampling_no)
        self.last_task = None

    def reg(self, t=None):
        return torch.sum(torch.abs(self.phi[self.last_task] - self.mu[self.last_task]))

    def task(self, x):
        min_err = None
        t = 0
        if len(self.mu) > 1:
            x_mean = x.mean(0)
            for i, v in enumerate(self.mu):
                if v is None:
                    break
                err = torch.sum(torch.abs(x_mean-v))
                if min_err is None or err < min_err:
                    min_err = err
                    t = i
        return t

    def forward(self, x, t=None):
        x = self.norm(x)
        if t is None:
            t = self.task(x)
        self.last_task = t
        super()._check_init(x, t)
        super()._update_mu(x, t)
        mask = super()._mask(x, t)
        x *= mask
        return x


class CosAXdA(XdA):

    def __init__(self, shape, alpha=0.01, sampling_no=64):
        super(CosAXdA, self).__init__(shape, alpha, sampling_no)
        self.last_task = None

    def reg(self, t=None):
        return torch.sum(torch.abs(self.phi[self.last_task] - self.mu[self.last_task]))

    def task(self, x):
        max_sim = None
        t = 0
        if len(self.mu) > 1:
            x_mean = x.mean(0).view(-1)
            for i, v in enumerate(self.mu):
                if v is None:
                    break
                sim = torch.nn.functional.cosine_similarity(x_mean, v.view(-1), 0)
                if max_sim is None or sim > max_sim:
                    max_sim = sim
                    t = i
        return t

    def forward(self, x, t=None):
        x = self.norm(x)
        if t is None:
            t = self.task(x)
        self.last_task = t
        super()._check_init(x, t)
        super()._update_mu(x, t)
        mask = super()._mask(x, t)
        x *= mask
        return x
