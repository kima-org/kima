# -*- coding: utf-8 -*-

# Copyright 2019-2024 Jean-Baptiste Delisle
# Licensed under the EUPL-1.2 or later

import numpy as np
from spleaf import cov, term, fenrir

prec = 1e-12
n = 153
deltas = [-1e-6, -3.3e-8, 1e-8, 5.7e-7]
coef_num_err = 10
nharm = 5


def _grad1d(x, f, k, deltaxk0, deltafmin, deltafmax, maxiter, absolute, args):
  xb = x.copy()
  deltaxk = deltaxk0
  fx = f(x, *args)
  if absolute:
    scale = 1.0
  else:
    scale = 1.0 / np.mean(np.abs(fx))
  xb[k] = x[k] + deltaxk
  fxb = f(xb, *args)
  for _ in range(maxiter):
    df = scale * np.mean(np.abs(fxb - fx))
    if df < deltafmin:
      deltaxk *= 1.5
    elif df > deltafmax:
      deltaxk /= 2.0
    else:
      break
    xb[k] = x[k] + deltaxk
    fxb = f(xb, *args)
  return (fxb - fx) / deltaxk


def grad(
  x,
  f,
  deltax0=1e-8,
  deltafmin=1e-6,
  deltafmax=5e-6,
  maxiter=500,
  absolute=False,
  args=(),
):
  if isinstance(deltax0, float):
    deltax = np.full_like(x, deltax0)
  else:
    deltax = deltax0
  return np.array(
    [
      _grad1d(x, f, k, deltax[k], deltafmin, deltafmax, maxiter, absolute, args)
      for k in range(x.size)
    ]
  )


def _generate_random_C(seed=0):
  np.random.seed(seed)
  t = np.cumsum(10 ** np.random.uniform(-2, 1.5, n))
  series_index = [np.arange(0, n, 2), np.arange(1, n, 2)]
  sig_err = np.random.uniform(0.5, 1.5, n)

  P = np.random.uniform(10, 100)
  rho = P * 10 ** np.random.uniform(-0.3, 2)
  sig_rv_phot = np.random.uniform(-1.5, 1.5)
  sig_rv_cb = np.random.uniform(-1.5, 1.5)
  sig_phot = np.random.uniform(-1.5, 1.5)
  a180 = np.random.uniform(0, 1)
  bar_i = np.random.uniform(0, np.pi / 2 - np.pi / 20)
  delta = np.random.uniform(0, np.pi / 2 - np.pi / 20)
  ld_a1 = np.random.uniform(-1, 1)
  ld_a2 = np.random.uniform(-1, 1)
  br_center = np.random.uniform(-20, 20)
  br_a1 = np.random.uniform(-1, 1)
  br_a2 = np.random.uniform(-1, 1)

  return cov.Cov(
    t,
    err=term.Error(sig_err),
    fenrir=fenrir.SpotsSameLat(nharm=nharm, brightening=True).kernel(
      series_index,
      decay_kernel=term.Matern12Kernel(1, rho),
      P=P,
      sig_rv_phot=sig_rv_phot,
      sig_rv_cb=sig_rv_cb,
      sig_phot=sig_phot,
      a180=a180,
      bar_i=bar_i,
      delta=delta,
      ld_a1=ld_a1,
      ld_a2=ld_a2,
      br_center=br_center,
      br_a1=br_a1,
      br_a2=br_a2,
    ),
  )


def _test_method_grad(method):
  """
  Common code for testing chi2_grad, loglike_grad
  """
  C = _generate_random_C()
  y = np.random.normal(0.0, 5.0, C.n)

  kparam = [k for k, p in enumerate(C.param)]  # if 'ld' not in p and 'br' not in p]
  param = [p for p in C.param]  # if 'ld' not in p and 'br' not in p]

  # Analytical grad
  func = getattr(C, method)
  _ = func(y)
  f_grad_res, f_grad_param = getattr(C, method + '_grad')()
  f_grad_param = f_grad_param[kparam]

  # Numerical grad
  def func_param(x):
    C.set_param(x, param)
    return func(y)

  f_grad_res_num = []
  f_grad_param_num = []
  for delta0 in deltas:
    f_grad_res_num.append(grad(y, func, deltax0=delta0))
    f_grad_param_num.append(grad(C.get_param(param), func_param, deltax0=delta0))
  f_grad_param_num = np.array(f_grad_param_num)

  # Comparison
  err = np.abs(f_grad_res - np.mean(f_grad_res_num, axis=0))
  num_err = np.std(f_grad_res_num, axis=0)
  err = max(0.0, np.max(err - coef_num_err * num_err))
  assert err < prec, (
    '{}_grad (y) not working' ' at required precision ({} > {})'
  ).format(method, err, prec)

  err = np.abs(f_grad_param - np.mean(f_grad_param_num, axis=0))
  num_err = np.std(f_grad_param_num, axis=0)
  for p, e, en in zip(param, err, num_err):
    print(p, e, en, e - coef_num_err * en)

  err = max(0.0, np.max(err - coef_num_err * num_err))
  assert err < prec, (
    '{}_grad (param) not working' ' at required precision ({} > {})'
  ).format(method, err, prec)


def test_chi2_grad():
  _test_method_grad('chi2')


def test_loglike_grad():
  _test_method_grad('loglike')
