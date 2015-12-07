from __future__ import division, print_function

from itertools import chain, combinations
import numpy as np
from scipy.misc import logsumexp as lse
from scipy.special import expit
from matplotlib import pyplot as plt

B = 1
model_weights = [1, 0]


class Function(object):
    def sample(self, n):
        raise NotImplementedError

    def __call__(self, S):
        raise NotImplementedError

    @property
    def parameters(self):
        raise NotImplementedError

    def gradient(self, S):
        """Gradients of the log-likelihood wrt the parameters."""
        raise NotImplementedError

    def project_parameters(self):
        raise NotImplementedError


class ModularFun(Function):
    def __init__(self, V, s):
        self.V = V
        self.s = s

        self.logz = np.sum(np.log(1 + np.exp(self.s)))

    def __call__(self, A):
        return np.sum(self.s[A]) - self.logz

    @property
    def parameters(self):
        return [self.s]

    def gradient(self, A):
        grad = - expit(self.s)
        grad[A] += 1
        return [grad]

    def project_parameters(self):
        self.logz = np.sum(np.log(1 + np.exp(self.s)))

    def sample(self, n):
        # sample n samples
        probs = expit(self.s).reshape(len(self.V))
        data = []
        for _ in range(n):
            s = np.nonzero(np.random.rand(len(self.V)) <= probs)[0]
            if np.size(s) != 0:
                data.append(s.tolist())
        return data

         #data_new = []
         #for d in data:
         #    idcs = np.argwhere(d)
         #    data_new.append(np.reshape(idcs, len(idcs)))
         #return data_new

    def _estimate_LL(self, data):
        return sum(self(d) for d in data) / len(data)


class GibbsSampler(object):
    def __init__(self, f):
        self.f = f

    @staticmethod
    def _indicator_to_set(ind):
        tmp = np.argwhere(ind)
        vec = np.reshape(tmp, len(tmp))
        return vec

    def _set_to_indicator(self, vec):
        ind = np.zeros(len(self.f.V), dtype=np.int)
        ind[vec] = 1
        return ind

    def sample_mult(self, nr, burn_in=1000, step=50, old_samples=None):
        # FIXME: Use greedy to find initial state
        if old_samples is None:
            initial = np.array(np.random.rand(len(self.f.V)) > 0.5, dtype=np.int)
            initial = self._set_to_indicator(self.sample(initial=initial, nr_rounds=burn_in))

        samples = []
        for i in range(nr):
            if old_samples is not None:
                initial = old_samples[i]
            samples.append(self.sample(initial=initial, nr_rounds=step))
            if old_samples is None:
                initial = self._set_to_indicator(samples[-1])

        return samples

    def sample(self, nr_rounds=100, initial=None):
        V = self.f.V

        if initial is None:
            initial = np.array(np.random.rand(len(V)) > 0.5, dtype=np.int)

        A = initial
        for r in range(nr_rounds):
            rand = np.log(np.random.rand(len(V)))
            for i in range(len(V)):
                A[i] = 0
                A1 = GibbsSampler._indicator_to_set(A)
                A[i] = 1
                A2 = GibbsSampler._indicator_to_set(A)
                logProb1 = self.f(A1)
                logProb2 = self.f(A2)
                logNorm = lse([logProb1, logProb2])
                prob1 = logProb1 - logNorm
                prob2 = logProb2 - logNorm
                if rand[i] < prob1:
                    A[i] = 0
                else:
                    A[i] = 1

        Aset = self._indicator_to_set(A)

        return Aset



class DiversityFun(Function):
    def __init__(self, V, n_dim):
        self.V = V
        self.n_dim = n_dim

        self.tpe = 'all'  # 'max' ... max only

        self.utilities = np.zeros(len(V), dtype=np.double)  # Utilities.
        self.W = 1e-3 * np.random.rand(len(V), n_dim).astype(dtype=np.double)  # Weights matrix.
        self.n_logz = np.array([0.], dtype=np.double)  # The "normalizer".

    @property
    def parameters(self):
        return [self.utilities, self.W, self.n_logz]

    # @property
    # def submod_fn(self):
    #     fac_ln = FacilityLocation(self.W)
    #     modular = (self.utilities - np.sum(self.W, axis=1))
    #     fn = add_modular(fac_ln, dict(zip(fac_ln.elements, modular)))
    #     return fn

    # def submod_approx(self):
    #     submod_fn = self.submod_fn
    #     return lower_greedy(submod_fn).probabilities
    #     # return upper_supdif(submod_fn).probabilities

    def project_parameters(self):
        negInd = self.W < 0
        self.W[negInd] = 1e-3 * np.random.rand(np.sum(negInd))

    def __call__(self, S):
        if len(S) == 0:
            return - 1000 + self.n_logz[0]  # FIXME: Assign no weight.

        slc = self.W[S, :]
        if self.tpe == 'max':
            return (
                self.n_logz[0] +
                np.sum(self.utilities[S]) +
                np.sum((np.max(slc, axis=0)))
            )
        else:
            return (
                self.n_logz[0] +
                np.sum(self.utilities[S]) +
                np.sum((np.max(slc, axis=0) - np.sum(slc, axis=0)))
            )

    def all_singleton_adds(self, S):
        """
        Compute all function values resulting from adding a single
        value to S
        """
        if self.tpe == 'max':
            assert False  # not implemented for the max only model
        else:
            Wutilities = -np.sum(self.W, axis=1)
            val = self.n_logz[0]
            val += np.sum(self.utilities[S])
            val += np.sum(Wutilities[S])
            vals = val + np.zeros(len(self.V))

            # now add the gain of adding a single value
            vals += self.utilities
            vals += Wutilities
            tmp = np.max(self.W[S, :], axis=0)

            # memory intense
            tmp2 =  np.repeat(tmp.reshape((1, self.n_dim, 1)), len(self.V), axis=0)
            vals += np.sum(np.max(np.concatenate((self.W.reshape((len(self.V), self.n_dim, 1)), tmp2), axis=2), axis=2), axis=1)

            # CPU-intense
            # for i in range(len(self.V)):
            #     vals[i] += np.sum(np.max(np.vstack((tmp, self.W[i, :])), axis=0))

            vals[S] = 0  # no gain for things that were already there

        return vals


    def gradient(self, A):
        A = np.asarray(A)
        grad_util = np.zeros_like(self.utilities)
        grad_util[A] = 1

        grad_W = np.zeros_like(self.W)
        indices = list(np.argmax(self.W[A, :], axis=0))
        grad_W[A, :] -= 1
        grad_W[A[indices], range(self.n_dim)] += 1

        grad_n_logz = np.ones_like(self.n_logz)

        return [grad_util, grad_W, grad_n_logz]

    def _estimate_LL(self, data, k_max=5):
        logZ = self.logZ_approx(k_max)
        return sum(self(d) - logZ for d in data) / len(data)

    def logZ_fixed_k(self):
        """
        Approximates \sum_{A \subseteq V, |A| = k} for some given k
         http://stackoverflow.com/questions/10106193/sum-of-product-of-subsets
        """

        l = self.utilities - np.sum(self.W, axis=1)
        l = l.tolist()

        x = [0]
        for i in l:  # a * b * i
            x = [lse([a, b + i]) for a, b in zip(chain([-float('inf')],x), chain(x,[-float('inf')]))]

        # apply lbd
        x = [y + self.n_logz[0] + np.sum(np.max(self.W, axis=0)) for y, k in zip(x, range(len(self.V), 0, -1))]
        x.append(self([])) # append zero-th entry
        x = x[::-1] # reverse order
        return x

    def logZ_approx(self, kmax, eps=1e-6):
        # TODO: remove -- only for testing!!!
        V = self.V
        probsAll = []
        for k in range(kmax):
            probs = []
            for A in combinations(V, k):
                probs.append(self(list(A)))
            probsAll.append(lse(probs))

        probsBound = self.logZ_fixed_k()
        # print(probsAll)
        # print(probsBound)
        assert np.all(probsAll <= np.array(probsBound[:len(probsAll)]) + eps)

        # remainder = -float('inf') # TODO: fix
        remainder = lse(probsBound[len(probsAll):])
        #IPython.embed()
        print("** logZex=%f, remainder=%f" % (np.exp(lse(probsAll)), np.exp(remainder)))

        return lse([lse(probsAll), remainder])

    def _estimate_LL_exact(self, data):
        logZ = self.logZ_fast()
        return sum(self(d) - logZ for d in data) / len(data)

    def logZ_FacLoc(self, ind, order):
        """ Compute FacLoc using the given indices... """
        inc = set()
        for (d, i) in enumerate(ind):
            inc.add(order[i,d])

        # remove elements that should not be included
        rem = set()
        for (d, i) in enumerate(ind):
            #if i > 0:
            r = order[:i,d]
            rem = rem.union(set(r))

        # given = inc.difference(rem)
        given = inc

        #print("inc: ", inc)
        # print("rem: ", rem)
        if len(inc.intersection(rem)) > 0:
            return - float('inf')
        # print("giv: ", given)
        # print("com: ", set(range(N)).difference(rem).difference(given))

        W = self.W
        N = len(self.V)
        if self.tpe == 'max':
            U = self.utilities
        else:
            U = self.utilities - np.sum(W, axis=1)

        val = 0
        D = self.n_dim
        for d in range(D):
            val += np.max(W[list(given), d])
        val += sum(U[list(given)])  # np.prod(np.exp(U[list(given)]))
        rest = list(set(range(N)).difference(rem).difference(given))
        # if len(rest) > 0:
        val += np.sum(np.log1p(np.exp(U[rest]))) # np.prod(1 + np.exp(U[rest]))

        return val


    def logZ_fast(self):
        logZ = self([]) - self.n_logz[0]

        W = self.W
        N = len(self.V)
        D = self.n_dim

        order = np.argsort(W, axis=0)
        for d in range(D):
            order[:,d] = order[::-1,d]

        ind = np.zeros(D)
        for k in range(N ** D):
            # print("** ind: ", ind)
            logZ = lse([logZ, self.logZ_FacLoc(ind, order)])

            # increase indices
            ind[0] += 1
            for l in range(D):
                if ind[l] >= N:
                    if l + 1 < D:
                        ind[l+1] += 1
                    ind[l] = 0

        return logZ + self.n_logz[0]

    def _checkgradient(self, A, eps=1e-6):
        pars = self.parameters
        grad_analytic = self.gradient(A)
        grad_numeric = []
        for p in pars:
            grad_numeric.append(np.zeros_like(p))

        f0 = self(A)
        for j, p in enumerate(pars):
            for i in range(p.size):
                idx = np.unravel_index(i, p.shape)
                orig = p[idx]
                p[idx] -= eps
                f1 = self(A)
                p[idx] = orig
                grad_numeric[j][idx] = (f0 - f1) / eps

        for i in range(len(pars)):
            print("CHECKING PARS #%d" % i)
            print("> analytic")
            print(grad_analytic[i])
            print("> numeric")
            print(grad_numeric[i])

            err = np.linalg.norm(grad_analytic[i] - grad_numeric[i])
            assert err < 1e-3
            print("-" * 30)

        print("*** GRADIENT OK")


class NCE(object):
    def __init__(self, f_model, f_noise):
        self.f_model = f_model
        self.f_noise = f_noise

    def _h(self, data, nu):
        # Implements (5) from
        # http://www.jmlr.org/papers/volume13/gutmann12a/gutmann12a.pdf
        G = self.f_model(data) - self.f_noise(data)
        return expit(G - np.log(nu))

    def _log_h(self, data, nu, mul=1.):
        # Implements (5) from
        # http://www.jmlr.org/papers/volume13/gutmann12a/gutmann12a.pdf
        G = self.f_model(data) - self.f_noise(data)
        return - np.logaddexp(0, mul * (np.log(nu) - G))

    def _objective(self, s_model, s_noise):
        nu = len(s_noise) / len(s_model)
        objective = sum(self._log_h(data, nu) for data in s_model)
        objective += sum(self._log_h(data, nu, -1.) for data in s_noise)
        return objective

    def gradient_sample(self, label, data, nu, out=None):
        assert label in (0, 1)

        grads = self.f_model.gradient(data)  # Model gradient.
        G = self.f_model(data) - self.f_noise(data)  # log-lik difference.
        fact = label - expit(G - np.log(nu))
        for grad in grads:
            grad *= fact
        return grads

    def gradient(self, s_model, s_noise):
        nu = len(s_noise) / len(s_model)

        # initialize things
        grads = [np.zeros_like(param) for param in self.f_model.parameters]

        for i, data in enumerate(chain(s_model, s_noise)):
            label = 1 if i < len(s_model) else 0
            grads_term = self.gradient_sample(label, data, nu)
            for grad, grad_term in zip(grads, grads_term):
                grad += grad_term

        return grads

    def learn(self, s_model, s_noise, n_iter=1000, eta_0=3, compute_LL=False,
              plot=True):
        pars = self.f_model.parameters

        if plot:
            values = []
        for i in range(n_iter):
            eta = eta_0 / np.power(1 + i, 0.9)  # Step size.

            grad = self.gradient(s_model, s_noise)
            for j, p in enumerate(pars):
                p += eta * grad[j] / len(s_model)
            self.f_model.project_parameters()

            if i % 5 == 0 and compute_LL:
                print("      LL ~ %f" % (self.f_model._estimate_LL(s_model)))

            if plot:
                values.append(self._objective(s_model, s_noise))

            self._report(s_model, s_noise, i, grad, eta)

        if plot:
            plt.plot(values, 'bo--')
            plt.show()

    def learn_sgd(self, s_model, s_noise, n_iter=1000, eta_0=1e-1,
                  compute_LL=False, plot=True):
        nu = len(s_noise) / len(s_model)  # Fraction of noise to data samples.
        params = self.f_model.parameters

        data = s_model + s_noise
        labels = [1] * len(s_model) + [0] * len(s_noise)

        if compute_LL:
            print("      LL ~ ", self.f_model._estimate_LL(s_model))

        if plot:
            values = []
        for i in range(n_iter * len(data)):
            idx = np.random.randint(0, len(data))
            eta = eta_0 / np.power(1 + i, 0.1)
            grads = self.gradient_sample(labels[idx], data[idx], nu)
            for param, grad in zip(params, grads):
                param += eta * grad
            self.f_model.project_parameters()

            if i % len(data) == 0:
                pass  # self._report(s_model, s_noise, i, grads, eta)

            if i % (len(data) // 20) == 0 and plot:
                values.append(self._objective(s_model, s_noise))

            if i % (5 * len(data)) == 0 and compute_LL:
                print("      LL ~ %f" % (self.f_model._estimate_LL(s_model)))

            if i % 20 * len(data) == 0:
                pass  # self._checkgradient(s_model, s_noise)

        if plot:
            plt.plot(values, 'ro--')
            plt.show()

    def _report(self, s_model, s_noise, i, grads, eta):
        print("[%3d] obj=%f" % (i, self._objective(s_model, s_noise)))
        print("      ||grad||=%f (eta=%f)" % (
            np.sum([np.linalg.norm(x) for x in grads]), eta))

    def _checkgradient(self, s_model, s_noise, eps=1e-6):
        pars = self.f_model.parameters
        grad_analytic = self.gradient(s_model, s_noise)
        grad_numeric = []
        for p in pars:
            grad_numeric.append(np.zeros_like(p))

        f0 = self._objective(s_model, s_noise)
        for j, p in enumerate(pars):
            for i in range(p.size):
                idx = np.unravel_index(i, p.shape)
                orig = p[idx]
                p[idx] -= eps
                f1 = self._objective(s_model, s_noise)
                p[idx] = orig
                grad_numeric[j][idx] = (f0 - f1) / eps

        print("CHECKING GRADIENT OF NCE.")
        for i in range(len(pars)):
            print("CHECKING PARS #%d" % i)
            print("> analytic")
            print(grad_analytic[i])
            print("> numeric")
            print(grad_numeric[i])

            nrm = np.linalg.norm(grad_analytic[i])
            if nrm < 1e-12:
                nrm = 1
            err = np.linalg.norm(grad_analytic[i] - grad_numeric[i]) / nrm
            print("Error=%f" % err)
            assert err < 1e-2
            print("-" * 30)

        print("*** GRADIENT OK")
