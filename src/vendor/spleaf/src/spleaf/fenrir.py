# -*- coding: utf-8 -*-

# Copyright 2019-2024 Jean-Baptiste Delisle
# Licensed under the EUPL-1.2 or later

__all__ = [
  'SpotsSameLat',
  'SpotsOppLatMag',
  'gaussian_weight',
  'symmetric_gaussian_weight',
  'SpotsLatDist',
  'SpotsLatDistMag',
  'SpotsLatDistInterpolator',
  'SpotsLatDistMagInterpolator',
  'MultiDist',
]

import pickle

import numpy as np

from . import term


class SpotsSameLat:
  r"""
  FENRIR model with spots/faculae appearing randomly on the star
  but always at the same latitude :math:`\delta`.

  Parameters
  ----------
  nharm : int
    Number of harmonics in the Fourier decomposition.
  brightening : bool
    Whether or not a brightening effect should be included in the model
    (for faculae or mixtures of faculae and spots).
  normalized : bool
    Whether or not the Fourier decomposition should be normalized.
  """

  def __init__(self, nharm, brightening=False, normalized=True):
    self._nharm = nharm
    self._nfreq = nharm + 1
    self._brightening = brightening
    self._normalized = normalized

    self._nfreq0 = 7 if brightening else 5

    self._j = np.arange(1, self._nfreq + self._nfreq0)
    self._Sct0 = np.empty(self._nfreq + self._nfreq0)
    k = np.arange(self._nfreq)
    n = np.arange(self._nfreq0)
    self._kpn = k[:, np.newaxis] + n[np.newaxis]
    self._kmn = np.abs(k[:, np.newaxis] - n[np.newaxis])

  def _compute_trigo(self):
    self._cosbi = np.cos(self._params['bar_i'])
    self._sinbi = np.sin(self._params['bar_i'])
    self._cosbi2 = self._cosbi * self._cosbi
    self._sinbi2 = self._sinbi * self._sinbi
    self._cos2bi = self._cosbi2 - self._sinbi2
    self._sin2bi = 2 * self._sinbi * self._cosbi
    self._tanbi = self._sinbi / self._cosbi

    self._cosd = np.cos(self._params['delta'])
    self._sind = np.sin(self._params['delta'])
    self._cosd2 = self._cosd * self._cosd
    self._sind2 = self._sind * self._sind
    self._cos2d = self._cosd2 - self._sind2
    self._sin2d = 2 * self._sind * self._cosd
    self._tand = self._sind / self._cosd

  def _rv_cb(self):
    alpha = np.zeros(3)
    alpha[0] = self._sinbi2 * self._sind2 + 0.5 * self._cosbi2 * self._cosd2  # const
    alpha[1] = 0.5 * self._sin2bi * self._sin2d  # cosphi
    alpha[2] = 0.5 * self._cosbi2 * self._cosd2  # cos2phi
    return alpha

  def _rv_cb_back(self, grad_alpha):
    # alpha[0] = self._sinbi2 * self._sind2 + 0.5 * self._cosbi2 * self._cosd2  # const
    self._grad_params['bar_i'] += (
      self._sinbi * self._cosbi * (2 * self._sind2 - self._cosd2) * grad_alpha[0]
    )
    self._grad_params['delta'] += (
      self._sind * self._cosd * (2 * self._sinbi2 - self._cosbi2) * grad_alpha[0]
    )
    # alpha[1] = 0.5 * self._sin2bi * self._sin2d  # cosphi
    self._grad_params['bar_i'] += self._cos2bi * self._sin2d * grad_alpha[1]
    self._grad_params['delta'] += self._cos2d * self._sin2bi * grad_alpha[1]
    # alpha[2] = 0.5 * self._cosbi2 * self._cosd2  # cos2phi
    self._grad_params['bar_i'] -= (
      self._sinbi * self._cosbi * self._cosd2 * grad_alpha[2]
    )
    self._grad_params['delta'] -= self._sind * self._cosd * self._cosbi2 * grad_alpha[2]

  def _rv_phot(self):
    beta = np.zeros(3)
    # Photometric / cos bar_i/2
    beta[1] = self._sinbi * self._sin2d  # sinphi
    beta[2] = self._cosbi * self._cosd2  # sin2phi
    return beta

  def _rv_phot_back(self, grad_beta):
    # beta[1] = self._sinbi * self._sin2d  # sinphi
    self._grad_params['bar_i'] += self._cosbi * self._sin2d * grad_beta[1]
    self._grad_params['delta'] += 2 * self._cos2d * self._sinbi * grad_beta[1]
    # beta[2] = self._cosbi * self._cosd2  # sin2phi
    self._grad_params['bar_i'] -= self._sinbi * self._cosd2 * grad_beta[2]
    self._grad_params['delta'] -= self._sin2d * self._cosbi * grad_beta[2]

  def _phot(self):
    alpha = np.zeros(2)
    alpha[0] = self._sinbi * self._sind  # const
    alpha[1] = self._cosbi * self._cosd  # cosphi
    return alpha

  def _phot_back(self, grad_alpha):
    # alpha[0] = self._sinbi * self._sind  # const
    self._grad_params['bar_i'] += self._cosbi * self._sind * grad_alpha[0]
    self._grad_params['delta'] += self._cosd * self._sinbi * grad_alpha[0]
    # alpha[1] = self._cosbi * self._cosd  # cosphi
    self._grad_params['bar_i'] -= self._sinbi * self._cosd * grad_alpha[1]
    self._grad_params['delta'] -= self._sind * self._cosbi * grad_alpha[1]

  def _apply_limb_dark(self, coefs, ld, is_alpha=True):
    degree_ld = len(ld)
    nfreq = coefs.shape[0]

    matrix_ld = np.zeros((degree_ld, nfreq + degree_ld - 1))
    matrix_ld[0, :nfreq] = coefs

    a = self._sinbi * self._sind
    b = self._cosbi * self._cosd
    hb = b * 0.5

    for k in range(1, degree_ld):
      matrix_ld[k, :-1] = hb * matrix_ld[k - 1, 1:]
      matrix_ld[k, 1:] += hb * matrix_ld[k - 1, :-1]
      if is_alpha:
        matrix_ld[k, 1] += hb * matrix_ld[k - 1, 0]  # Add cos (- omega*t) term
      else:
        matrix_ld[k, 0] = 0
      matrix_ld[k] += a * matrix_ld[k - 1]

    return np.sum(ld[:, np.newaxis] * matrix_ld, axis=0), matrix_ld

  def _apply_limb_dark_back(self, grad_new_coefs, coefs, ld, matrix_ld, is_alpha=True):
    degree_ld = len(ld)
    nfreq = coefs.shape[0]
    a = self._sinbi * self._sind
    b = self._cosbi * self._cosd
    hb = b * 0.5
    grad_a = 0
    grad_hb = 0
    # return np.sum(ld[:, np.newaxis] * matrix_ld, axis=0), matrix_ld
    grad_ld = np.sum(matrix_ld * grad_new_coefs[np.newaxis], axis=1)
    grad_matrix_ld = ld[:, np.newaxis] * grad_new_coefs[np.newaxis]

    for k in range(degree_ld - 1, 0, -1):
      # matrix_ld[k] += a * matrix_ld[k - 1]
      grad_a += np.sum(grad_matrix_ld[k] * matrix_ld[k - 1])
      grad_matrix_ld[k - 1] += a * grad_matrix_ld[k]
      if is_alpha:
        # matrix_ld[k, 1] += hb * matrix_ld[k - 1, 0]  # Add cos (- omega*t) term
        grad_hb += np.sum(matrix_ld[k - 1, 0] * grad_matrix_ld[k, 1])
        grad_matrix_ld[k - 1, 0] += hb * grad_matrix_ld[k, 1]
      else:
        grad_matrix_ld[k, 0] = 0
      # matrix_ld[k, 1:] += hb * matrix_ld[k - 1, :-1]
      grad_hb += np.sum(matrix_ld[k - 1, :-1] * grad_matrix_ld[k, 1:])
      grad_matrix_ld[k - 1, :-1] += hb * grad_matrix_ld[k, 1:]
      # matrix_ld[k, :-1] = hb * matrix_ld[k - 1, 1:]
      grad_hb += np.sum(matrix_ld[k - 1, 1:] * grad_matrix_ld[k, :-1])
      grad_matrix_ld[k - 1, 1:] += hb * grad_matrix_ld[k, :-1]

    # a = self._sinbi * self._sind
    # b = self._cosbi * self._cosd
    # hb = b * 0.5
    self._grad_params['bar_i'] += (
      self._cosbi * self._sind * grad_a - self._sinbi * self._cosd * grad_hb / 2
    )
    self._grad_params['delta'] += (
      self._cosd * self._sinbi * grad_a - self._sind * self._cosbi * grad_hb / 2
    )

    # matrix_ld[0, :nfreq] = coefs
    grad_coefs = grad_matrix_ld[0, :nfreq]

    return grad_coefs, grad_ld

  def compute_Fourier(
    self,
    sig_rv_phot,
    sig_rv_cb,
    sig_phot,
    a180,
    bar_i,
    delta,
    ld_a1,
    ld_a2,
    br_center=None,
    br_a1=None,
    br_a2=None,
  ):
    r"""
    Compute the Fourier decomposition of the kernel.

    Parameters
    ----------
    sig_rv_phot, sig_rv_cb, sig_phot : float
      Amplitudes of the three modeled effects
      (photometric and convective blueshift effects on radial-velocity,
      photometric effect)
    a180 : float
      Relative amplitude of the spots/faculae at the opposite longitude.
    bar_i : float
      Complementary inclination of the star (:math:`\bar{i} = \pi/2 - i`).
    delta : float
      Latitude of the spots/faculae.
    ld_a1, ld_a2 : float
      Quadratic limb-darkening coefficients.
    br_center, br_a1, br_a2 : float
      Brightening coefficients (ignored if the brightening is not activated).

    Returns
    -------
    ab : (2, nharm + 1, 2) ndarray
      Fourier coefficients, ordered such as ab[0, 2, 0] and ab[0, 2, 1]
      are the cosine coefficients of the second harmonics of the RV and photometric
      timeseries respectively, while ab[1] are the sine coefficients.
    """
    self._params = dict(
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
    )
    self._compute_trigo()

    # Visibility condition
    coswt0 = max(-self._tanbi * self._tand, -1)
    assert coswt0 < 1
    wt0 = np.arccos(coswt0)

    # Rv effect
    self._alpha_rv_cb = self._rv_cb()
    self._beta_rv_phot = self._rv_phot()
    # Photometric effect
    self._alpha_phot = self._phot()

    # Limb-darkening
    ld = np.array([1 - ld_a1 - ld_a2, ld_a1, ld_a2])
    self._alpha_rv_cb_ld, self._matrix_rv_cb_ld = self._apply_limb_dark(
      self._alpha_rv_cb, ld
    )
    self._beta_rv_phot_ld, self._matrix_rv_phot_ld = self._apply_limb_dark(
      self._beta_rv_phot, ld, is_alpha=False
    )
    self._alpha_phot_ld, self._matrix_phot_ld = self._apply_limb_dark(
      self._alpha_phot, ld
    )

    # Limb-brightening
    if self._brightening:
      br = np.array([br_center - br_a1 - br_a2, br_a1, br_a2])
      self._beta_rv_phot_br, self._matrix_rv_phot_br = self._apply_limb_dark(
        self._beta_rv_phot_ld, br, is_alpha=False
      )
      self._alpha_phot_br, self._matrix_phot_br = self._apply_limb_dark(
        self._alpha_phot_ld, br
      )
    else:
      self._beta_rv_phot_br = self._beta_rv_phot_ld.copy()
      self._alpha_phot_br = self._alpha_phot_ld.copy()

    # Visibility windowing
    self._Sct0[0] = wt0
    self._Sct0[1:] = np.sin(self._j * wt0) / self._j
    self._Sckpn = self._Sct0[self._kpn]
    self._Sckmn = self._Sct0[self._kmn]
    self._Scp = self._Sckpn + self._Sckmn
    self._Scm = self._Sckpn - self._Sckmn
    self._Scp[0] /= np.sqrt(2)

    self._unscaled_ab = np.zeros((2, self._nfreq, 2))
    self._unscaled_ab[0, :, 0] = (
      self._Scp[:, : self._alpha_rv_cb_ld.shape[0]] @ self._alpha_rv_cb_ld
    )
    self._unscaled_ab[1, :, 0] = self._Scm[:, 1:] @ self._beta_rv_phot_br[1:]
    self._unscaled_ab[0, :, 1] = self._Scp[:, :-1] @ self._alpha_phot_br

    # Spot at opposite longitude
    self._unscaled_ab_180 = self._unscaled_ab.copy()
    self._unscaled_ab_180[:, ::2] *= 1 + a180
    self._unscaled_ab_180[:, 1::2] *= 1 - a180

    # Normalized coefs
    self._norm_rv_cb = 1
    self._norm_rv_phot = 1
    self._norm_phot = 1
    if self._normalized:
      self._norm_rv_cb = np.sum(self._unscaled_ab_180[0, :, 0] ** 2)
      self._norm_rv_phot = np.sum(self._unscaled_ab_180[1, :, 0] ** 2)
      self._norm_phot = np.sum(self._unscaled_ab_180[0, :, 1] ** 2)

    self._ab = self._unscaled_ab_180.copy()
    self._ab[0, :, 0] *= sig_rv_cb / np.sqrt(self._norm_rv_cb)
    self._ab[1, :, 0] *= sig_rv_phot / np.sqrt(self._norm_rv_phot)
    self._ab[0, :, 1] *= sig_phot / np.sqrt(self._norm_phot)

    return self._ab

  def compute_Fourier_back(self, grad_ab):
    r"""
    Backward propagation of the gradient for :func:`compute_Fourier`.
    """

    self._grad_params = {key: 0 for key in self._params}

    # self._ab[0, :, 0] *= sig_rv_cb / np.sqrt(self._norm_rv_cb)
    abgab_rv_cb = np.sum(self._unscaled_ab_180[0, :, 0] * grad_ab[0, :, 0]) / np.sqrt(
      self._norm_rv_cb
    )
    self._grad_params['sig_rv_cb'] += abgab_rv_cb
    grad_norm_rv_cb = -0.5 * self._params['sig_rv_cb'] * abgab_rv_cb / self._norm_rv_cb
    grad_ab[0, :, 0] *= self._params['sig_rv_cb'] / np.sqrt(self._norm_rv_cb)
    # self._ab[1, :, 0] *= sig_rv_phot / np.sqrt(self._norm_rv_phot)
    abgab_rv_phot = np.sum(self._unscaled_ab_180[1, :, 0] * grad_ab[1, :, 0]) / np.sqrt(
      self._norm_rv_phot
    )
    self._grad_params['sig_rv_phot'] += abgab_rv_phot
    grad_norm_rv_phot = (
      -0.5 * self._params['sig_rv_phot'] * abgab_rv_phot / self._norm_rv_phot
    )
    grad_ab[1, :, 0] *= self._params['sig_rv_phot'] / np.sqrt(self._norm_rv_phot)
    # self._ab[0, :, 1] *= sig_phot / np.sqrt(self._norm_phot)
    abgab_phot = np.sum(self._unscaled_ab_180[0, :, 1] * grad_ab[0, :, 1]) / np.sqrt(
      self._norm_phot
    )
    self._grad_params['sig_phot'] += abgab_phot
    grad_norm_phot = -0.5 * self._params['sig_phot'] * abgab_phot / self._norm_phot
    grad_ab[0, :, 1] *= self._params['sig_phot'] / np.sqrt(self._norm_phot)

    if self._normalized:
      # self._norm_rv_cb = np.sum(self._unscaled_ab_180[0, :, 0] ** 2)
      grad_ab[0, :, 0] += 2 * grad_norm_rv_cb * self._unscaled_ab_180[0, :, 0]
      # self._norm_rv_phot = np.sum(self._unscaled_ab_180[1, :, 0] ** 2)
      grad_ab[1, :, 0] += 2 * grad_norm_rv_phot * self._unscaled_ab_180[1, :, 0]
      # self._norm_phot = np.sum(self._unscaled_ab_180[0, :, 1] ** 2)
      grad_ab[0, :, 1] += 2 * grad_norm_phot * self._unscaled_ab_180[0, :, 1]

    # self._unscaled_ab_180[:, ::2] *= 1 + a180
    # self._unscaled_ab_180[:, 1::2] *= 1 - a180
    self._grad_params['a180'] += np.sum(
      grad_ab[:, ::2] * self._unscaled_ab[:, ::2]
    ) - np.sum(grad_ab[:, 1::2] * self._unscaled_ab[:, 1::2])
    grad_ab[:, ::2] *= 1 + self._params['a180']
    grad_ab[:, 1::2] *= 1 - self._params['a180']

    # self._unscaled_ab[0, :, 0] = self._Scp[:, : self._alpha_rv_cb_ld.shape[0]] @ self._alpha_rv_cb_ld
    # self._unscaled_ab[1, :, 0] = self._Scm[:, 1:] @ self._beta_rv_phot_br[1:]
    # self._unscaled_ab[0, :, 1] = self._Scp[:, :-1] @ self._alpha_phot_br
    grad_Scp = np.zeros_like(self._Scp)
    grad_Scm = np.zeros_like(self._Scm)

    grad_Scp[:, : self._alpha_rv_cb_ld.shape[0]] += (
      grad_ab[0, :, 0, np.newaxis] * self._alpha_rv_cb_ld[np.newaxis]
    )
    grad_alpha_rv_cb = (
      self._Scp[:, : self._alpha_rv_cb_ld.shape[0]].T @ grad_ab[0, :, 0]
    )
    grad_Scm[:, 1:] += (
      grad_ab[1, :, 0, np.newaxis] * self._beta_rv_phot_br[np.newaxis, 1:]
    )
    grad_beta_rv_phot = np.zeros_like(self._beta_rv_phot_br)
    grad_beta_rv_phot[1:] = self._Scm[:, 1:].T @ grad_ab[1, :, 0]
    grad_Scp[:, :-1] += grad_ab[0, :, 1, np.newaxis] * self._alpha_phot_br[np.newaxis]
    grad_alpha_phot = self._Scp[:, :-1].T @ grad_ab[0, :, 1]

    # self._Scp[0] /= np.sqrt(2)
    grad_Scp[0] /= np.sqrt(2)
    # self._Scp = self._Sckpn + self._Sckmn
    # self._Scm = self._Sckpn - self._Sckmn
    grad_Sckpn = grad_Scp + grad_Scm
    grad_Sckmn = grad_Scp - grad_Scm
    # self._Sckpn = self._Sct0[self._kpn]
    # self._Sckmn = self._Sct0[self._kmn]
    grad_Sct0 = np.zeros_like(self._Sct0)
    for i, j in enumerate(self._kpn.flat):
      grad_Sct0[j] += grad_Sckpn.flat[i]
    for i, j in enumerate(self._kmn.flat):
      grad_Sct0[j] += grad_Sckmn.flat[i]
    # self._Sct0[0] = wt0
    # self._Sct0[1:] = np.sin(self._j * wt0) / self._j
    grad_wt0 = grad_Sct0[0] + np.sum(grad_Sct0[1:] * np.cos(self._j * self._Sct0[0]))

    # Limb-brightening
    if self._brightening:
      br = np.array(
        [
          self._params['br_center'] - self._params['br_a1'] - self._params['br_a2'],
          self._params['br_a1'],
          self._params['br_a2'],
        ]
      )
      # self._beta_rv_phot_br = self._apply_limb_dark(self._beta_rv_phot_ld, br, is_alpha=False)
      grad_beta_rv_phot, grad_br = self._apply_limb_dark_back(
        grad_beta_rv_phot,
        self._beta_rv_phot_ld,
        br,
        self._matrix_rv_phot_br,
        is_alpha=False,
      )
      # self._alpha_phot_br = self._apply_limb_dark(self._alpha_phot_ld, br)
      grad_alpha_phot, grad_br_tmp = self._apply_limb_dark_back(
        grad_alpha_phot, self._alpha_phot_ld, br, self._matrix_phot_br
      )
      grad_br += grad_br_tmp
      # br = np.array([br_center - br_a1 - br_a2, br_a1, br_a2])
      self._grad_params['br_center'] = grad_br[0]
      self._grad_params['br_a1'] = grad_br[1] - grad_br[0]
      self._grad_params['br_a2'] = grad_br[2] - grad_br[0]

    # Limb-darkening
    ld = np.array(
      [
        1 - self._params['ld_a1'] - self._params['ld_a2'],
        self._params['ld_a1'],
        self._params['ld_a2'],
      ]
    )
    # self._alpha_rv_cb_ld = self._apply_limb_dark(self._alpha_rv_cb, ld)
    grad_alpha_rv_cb, grad_ld = self._apply_limb_dark_back(
      grad_alpha_rv_cb, self._alpha_rv_cb, ld, self._matrix_rv_cb_ld
    )
    # self._beta_rv_phot_ld = self._apply_limb_dark(self._beta_rv_phot, ld, is_alpha=False)
    grad_beta_rv_phot, grad_ld_tmp = self._apply_limb_dark_back(
      grad_beta_rv_phot,
      self._beta_rv_phot,
      ld,
      self._matrix_rv_phot_ld,
      is_alpha=False,
    )
    grad_ld += grad_ld_tmp
    # self._alpha_phot_ld = self._apply_limb_dark(self._alpha_phot, ld)
    grad_alpha_phot, grad_ld_tmp = self._apply_limb_dark_back(
      grad_alpha_phot, self._alpha_phot, ld, self._matrix_phot_ld
    )
    grad_ld += grad_ld_tmp
    # ld = np.array([1 - ld_a1 - ld_a2, ld_a1, ld_a2])
    self._grad_params['ld_a1'] = grad_ld[1] - grad_ld[0]
    self._grad_params['ld_a2'] = grad_ld[2] - grad_ld[0]

    # self._alpha_rv_cb = self._rv_cb()
    self._rv_cb_back(grad_alpha_rv_cb)
    # self._beta_rv_phot = self._rv_phot()
    self._rv_phot_back(grad_beta_rv_phot)
    # self._alpha_phot = self._phot()
    self._phot_back(grad_alpha_phot)

    # coswt0 = max(-self._tanbi * self._tand, -1)
    # wt0 = np.arccos(coswt0)
    if self._tanbi * self._tand < 1:
      coswt0 = -self._tanbi * self._tand
      grad_coswt0 = -grad_wt0 / np.sqrt(1 - coswt0**2)
      self._grad_params['bar_i'] -= (1 + self._tanbi**2) * self._tand * grad_coswt0
      self._grad_params['delta'] -= (1 + self._tand**2) * self._tanbi * grad_coswt0

    return self._grad_params

  def _translate_param(self, **kwargs):
    r"""
    Helper function to perform a change of variables.
    Intended to be used with :class:`spleaf.term.TransformKernel`.
    """

    param = {key: kwargs.pop(key) for key in list(kwargs) if key.startswith('decay_')}
    param['fourier_P'] = kwargs.pop('P')

    ab = self.compute_Fourier(**kwargs)

    param['fourier_alpha'] = ab[0, ..., np.newaxis]
    param['fourier_beta'] = ab[1, ..., np.newaxis]

    return param

  def _translate_param_back(self, grad):
    r"""
    Backward propagation of the gradient for :func:`_translate_param`.
    """
    grad_param = grad.copy()
    grad_param['P'] = grad_param.pop('fourier_P')
    grad_ab = np.array(
      [grad_param.pop('fourier_alpha'), grad_param.pop('fourier_beta')]
    )
    grad_param.update(self.compute_Fourier_back(grad_ab[..., 0]))

    return grad_param

  def kernel(
    self,
    series_index,
    decay_kernel,
    P,
    sig_rv_phot,
    sig_rv_cb,
    sig_phot,
    a180,
    bar_i,
    delta,
    ld_a1,
    ld_a2,
    br_center=None,
    br_a1=None,
    br_a2=None,
  ):
    r"""
    Generate a S+LEAF kernel corresponding to this FENRIR model,
    to be included in a covariance matrix (:class:`spleaf.cov.Cov`).

    Parameters
    ----------
    series_index : list of ndarrays
      Indices corresponding to each original time series in the merged time series.
    decay_kernel : :class:`spleaf.term.Kernel`
      Kernel describing the (dis)appearing of spots/faculae.
    P : float
      Rotation period.
    sig_rv_phot, sig_rv_cb, sig_phot : float
      Amplitudes of the three modeled effects
      (photometric and convective blueshift effects on radial-velocity,
      photometric effect)
    a180 : float
      Relative amplitude of the spots/faculae at the opposite longitude.
    bar_i : float
      Complementary inclination of the star (:math:`\bar{i} = \pi/2 - i`).
    delta : float
      Latitude of the spots/faculae.
    ld_a1, ld_a2 : float
      Quadratic limb-darkening coefficients.
    br_center, br_a1, br_a2 : float
      Brightening coefficients (ignored if the brightening is not activated).

    Returns
    -------
    kernel : :class:`spleaf.term.Kernel`
      S+LEAF kernel corresponding to this FENRIR model.
    """
    kwargs = dict(
      P=P,
      sig_rv_phot=sig_rv_phot,
      sig_rv_cb=sig_rv_cb,
      sig_phot=sig_phot,
      a180=a180,
      bar_i=bar_i,
      delta=delta,
      ld_a1=ld_a1,
      ld_a2=ld_a2,
    )

    if self._brightening:
      kwargs.update(dict(br_center=br_center, br_a1=br_a1, br_a2=br_a2))

    kwargs.update(
      {f'decay_{key}': decay_kernel._get_param(key) for key in decay_kernel._param}
    )

    return term.TransformKernel(
      term.SimpleProductKernel(
        decay=decay_kernel,
        fourier=term.MultiFourierKernel(
          1,
          np.zeros((self._nfreq, 2, 1)),
          np.zeros((self._nfreq, 2, 1)),
          series_index=series_index,
          vectorize=True,
        ),
      ),
      self._translate_param,
      self._translate_param_back,
      **kwargs,
    )


class SpotsOppLatMag:
  r"""
  FENRIR model with spots/faculae appearing randomly on the star
  but always at the two same latitudes :math:`\pm\delta`,
  and including the effect of the magnetic cycle.

  Parameters
  ----------
  nharm : int
    Number of harmonics in the Fourier decomposition.
  brightening : bool
    Whether or not a brightening effect should be included in the model
    (for faculae or mixtures of faculae and spots).
  """

  def __init__(self, nharm, brightening=False):
    self.north = SpotsSameLat(nharm, brightening, normalized=False)
    self.south = SpotsSameLat(nharm, brightening, normalized=False)

  def compute_Fourier(
    self,
    sig_rv_phot,
    sig_rv_cb,
    sig_phot,
    mag_ratio,
    a180,
    bar_i,
    delta,
    ld_a1,
    ld_a2,
    br_center=None,
    br_a1=None,
    br_a2=None,
  ):
    r"""
    Compute the Fourier decomposition of the kernel.

    Parameters
    ----------
    sig_rv_phot, sig_rv_cb, sig_phot : float
      Amplitudes of the three modeled effects
      (photometric and convective blueshift effects on radial-velocity,
      photometric effect)
    mag_ratio : float
      Relative amplitude of the magnetic cycle related activity with respect to the
      rotation period related activity.
    a180 : float
      Relative amplitude of the spots/faculae at the opposite longitude.
    bar_i : float
      Complementary inclination of the star (:math:`\bar{i} = \pi/2 - i`).
    delta : float
      Latitude of the spots/faculae.
    ld_a1, ld_a2 : float
      Quadratic limb-darkening coefficients.
    br_center, br_a1, br_a2 : float
      Brightening coefficients (ignored if the brightening is not activated).
    **kwargs :
      Parameters to provide to the weight function.

    Returns
    -------
    per_ab : (2, nharm + 1, 2, 2) ndarray
      Fourier coefficients for the rotation period related effects,
      ordered such as ab[0, 2, 0] and ab[0, 2, 1]
      are the cosine coefficients of the second harmonics of the RV and photometric
      timeseries respectively, while ab[1] are the sine coefficients.
    mag_ab : (2, 1, 2, 1) ndarray
      Fourier coefficients for the magnetic cycle related effects.
    """
    self._params = dict(
      sig_rv_phot=sig_rv_phot,
      sig_rv_cb=sig_rv_cb,
      sig_phot=sig_phot,
      mag_ratio=mag_ratio,
      a180=a180,
      bar_i=bar_i,
      delta=delta,
      ld_a1=ld_a1,
      ld_a2=ld_a2,
      br_center=br_center,
      br_a1=br_a1,
      br_a2=br_a2,
    )

    self._per_ab = np.zeros((2, self.north._nfreq, 2, 2))
    try:
      self._per_ab[..., 0] = self.north.compute_Fourier(
        1,
        1,
        1,
        a180,
        bar_i,
        delta,
        ld_a1,
        ld_a2,
        br_center,
        br_a1,
        br_a2,
      )
      self._north_visible = True
    except AssertionError:
      self._north_visible = False
    try:
      self._per_ab[..., 1] = self.south.compute_Fourier(
        1,
        1,
        1,
        a180,
        bar_i,
        -delta,
        ld_a1,
        ld_a2,
        br_center,
        br_a1,
        br_a2,
      )
      self._south_visible = True
    except AssertionError:
      self._south_visible = False

    # Compute magnetic cycle effect
    self._mag_alpha = mag_ratio * np.sum(self._per_ab[0, 0], axis=1)

    # Scaling
    self._scaled_per_ab = self._per_ab.copy()
    self._scaled_mag_alpha = self._mag_alpha.copy()

    scale_cb = sig_rv_cb / np.sqrt(
      np.sum(self._scaled_per_ab[0, :, 0] ** 2) + self._scaled_mag_alpha[0] ** 2
    )
    self._scaled_per_ab[0, :, 0] *= scale_cb
    self._scaled_mag_alpha[0] *= scale_cb

    self._scaled_per_ab[1, :, 0] *= sig_rv_phot / np.sqrt(
      np.sum(self._scaled_per_ab[1, :, 0] ** 2)
    )

    scale_phot = sig_phot / np.sqrt(
      np.sum(self._scaled_per_ab[:, :, 1] ** 2) + self._scaled_mag_alpha[1] ** 2
    )
    self._scaled_per_ab[:, :, 1] *= scale_phot
    self._scaled_mag_alpha[1] *= scale_phot

    return (
      self._scaled_per_ab,
      np.array([[self._scaled_mag_alpha], [2 * [0]]])[..., np.newaxis],
    )

  def compute_Fourier_back(self, grad_per, grad_mag):
    r"""
    Backward propagation of the gradient for :func:`compute_Fourier`.
    """
    grad_params = {key: 0 for key in self._params}

    grad_mag = grad_mag[0, 0, :, 0]

    # Scaling
    # self._scaled_per_ab = self._per_ab.copy()
    # self._scaled_mag_alpha = self._mag_alpha.copy()

    # scale_cb = sig_rv_cb / np.sqrt(
    #   np.sum(self._scaled_per_ab[0, :, 0] ** 2) + self._scaled_mag_alpha[0] ** 2
    # )
    # self._scaled_per_ab[0, :, 0] *= scale_cb
    # self._scaled_mag_alpha[0] *= scale_cb
    cgradc = (
      np.sum(self._scaled_per_ab[0, :, 0] * grad_per[0, :, 0])
      + self._scaled_mag_alpha[0] * grad_mag[0]
    )
    grad_params['sig_rv_cb'] = cgradc / self._params['sig_rv_cb']
    s = np.sqrt(np.sum(self._per_ab[0, :, 0] ** 2) + self._mag_alpha[0] ** 2)
    grad_s = -cgradc / s
    grad_per[0, :, 0] *= self._params['sig_rv_cb'] / s
    grad_per[0, :, 0] += self._per_ab[0, :, 0] * grad_s / s
    grad_mag[0] *= self._params['sig_rv_cb'] / s
    grad_mag[0] += self._mag_alpha[0] * grad_s / s

    # self._scaled_per_ab[1, :, 0] *= sig_rv_phot / np.sqrt(
    #   np.sum(self._scaled_per_ab[1, :, 0] ** 2)
    # )
    cgradc = np.sum(self._scaled_per_ab[1, :, 0] * grad_per[1, :, 0])
    grad_params['sig_rv_phot'] = cgradc / self._params['sig_rv_phot']
    s = np.sqrt(np.sum(self._per_ab[1, :, 0] ** 2))
    grad_s = -cgradc / s
    grad_per[1, :, 0] *= self._params['sig_rv_phot'] / s
    grad_per[1, :, 0] += self._per_ab[1, :, 0] * grad_s / s

    # scale_phot = sig_phot / np.sqrt(
    #   np.sum(self._scaled_per_ab[:, :, 1] ** 2) + self._scaled_mag_alpha[1] ** 2
    # )
    # self._scaled_per_ab[:, :, 1] *= scale_phot
    # self._scaled_mag_alpha[1] *= scale_phot
    cgradc = (
      np.sum(self._scaled_per_ab[:, :, 1] * grad_per[:, :, 1])
      + self._scaled_mag_alpha[1] * grad_mag[1]
    )
    grad_params['sig_phot'] = cgradc / self._params['sig_phot']
    s = np.sqrt(np.sum(self._per_ab[:, :, 1] ** 2) + self._mag_alpha[1] ** 2)
    grad_s = -cgradc / s
    grad_per[:, :, 1] *= self._params['sig_phot'] / s
    grad_per[:, :, 1] += self._per_ab[:, :, 1] * grad_s / s
    grad_mag[1] *= self._params['sig_phot'] / s
    grad_mag[1] += self._mag_alpha[1] * grad_s / s

    # self._mag_alpha = mag_ratio * np.sum(self._per_ab[0, 0], axis=1)
    grad_params['mag_ratio'] = np.sum(np.sum(self._per_ab[0, 0], axis=1) * grad_mag)
    grad_per[0, 0] += self._params['mag_ratio'] * grad_mag[:, np.newaxis]

    lparam = ['a180', 'bar_i', 'ld_a1', 'ld_a2']
    if self.north._brightening:
      lparam += ['br_center', 'br_a1', 'br_a2']
    for k, (hemi, visib) in enumerate(
      [(self.north, self._north_visible), (self.south, self._south_visible)]
    ):
      if visib:
        grad_hemi = hemi.compute_Fourier_back(grad_per[..., k])
        grad_params['delta'] += (1 - 2 * k) * grad_hemi['delta']

        for key in lparam:
          grad_params[key] += grad_hemi[key]

    return grad_params

  def _translate_param(self, **kwargs):
    r"""
    Helper function to perform a change of variables.
    Intended to be used with :class:`spleaf.term.TransformKernel`.
    """
    param = {
      f'per_{key}': kwargs.pop(key) for key in list(kwargs) if key.startswith('decay_')
    }

    param.update(
      {key: kwargs.pop(key) for key in list(kwargs) if key.startswith('mag_decay_')}
    )

    param['per_fourier_P'] = kwargs.pop('P')

    per_ab, mag_ab = self.compute_Fourier(**kwargs)
    param['per_fourier_alpha'] = per_ab[0]
    param['per_fourier_beta'] = per_ab[1]
    param['mag_fourier_alpha'] = mag_ab[0]

    return param

  def _translate_param_back(self, grad):
    r"""
    Backward propagation of the gradient for :func:`_translate_param`.
    """
    grad_params = grad.copy()
    grad_params['P'] = grad_params.pop('per_fourier_P')
    grad_params.update(
      {
        key[4:]: grad_params.pop(key)
        for key in list(grad_params)
        if key.startswith('per_decay_')
      }
    )
    grad_per = np.array(
      [grad_params.pop('per_fourier_alpha'), grad_params.pop('per_fourier_beta')]
    )
    grad_mag = np.array([grad_params.pop('mag_fourier_alpha'), [2 * [[0]]]])
    grad_params.update(self.compute_Fourier_back(grad_per, grad_mag))

    return grad_params

  def kernel(
    self,
    series_index,
    decay_kernel,
    mag_decay_kernel,
    P,
    sig_rv_phot,
    sig_rv_cb,
    sig_phot,
    mag_ratio,
    a180,
    bar_i,
    delta,
    ld_a1,
    ld_a2,
    br_center=None,
    br_a1=None,
    br_a2=None,
  ):
    r"""
    Generate a S+LEAF kernel corresponding to this FENRIR model,
    to be included in a covariance matrix (:class:`spleaf.cov.Cov`).

    Parameters
    ----------
    series_index : list of ndarrays
      Indices corresponding to each original time series in the merged time series.
    decay_kernel : :class:`spleaf.term.Kernel`
      Kernel describing the (dis)appearing of spots/faculae.
    P : float
      Rotation period.
    sig_rv_phot, sig_rv_cb, sig_phot : float
      Amplitudes of the three modeled effects
      (photometric and convective blueshift effects on radial-velocity,
      photometric effect)
    mag_ratio : float
      Relative amplitude of the magnetic cycle related activity with respect to the
      rotation period related activity.
    a180 : float
      Relative amplitude of the spots/faculae at the opposite longitude.
    bar_i : float
      Complementary inclination of the star (:math:`\bar{i} = \pi/2 - i`).
    delta : float
      Latitude of the spots/faculae.
    ld_a1, ld_a2 : float
      Quadratic limb-darkening coefficients.
    br_center, br_a1, br_a2 : float
      Brightening coefficients (ignored if the brightening is not activated).

    Returns
    -------
    kernel : :class:`spleaf.term.Kernel`
      S+LEAF kernel corresponding to this FENRIR model.
    """
    kwargs = dict(
      P=P,
      sig_rv_phot=sig_rv_phot,
      sig_rv_cb=sig_rv_cb,
      sig_phot=sig_phot,
      mag_ratio=mag_ratio,
      a180=a180,
      bar_i=bar_i,
      delta=delta,
      ld_a1=ld_a1,
      ld_a2=ld_a2,
    )

    if self.north._brightening:
      kwargs.update(dict(br_center=br_center, br_a1=br_a1, br_a2=br_a2))

    kwargs.update(
      {f'decay_{key}': decay_kernel._get_param(key) for key in decay_kernel._param}
    )

    kwargs.update(
      {
        f'mag_decay_{key}': mag_decay_kernel._get_param(key)
        for key in mag_decay_kernel._param
      }
    )

    return term.TransformKernel(
      term.SimpleSumKernel(
        per=term.SimpleProductKernel(
          decay=decay_kernel,
          fourier=term.MultiFourierKernel(
            1,
            np.zeros((self.north._nfreq, 2, 2)),
            np.zeros((self.north._nfreq, 2, 2)),
            series_index=series_index,
            vectorize=True,
          ),
        ),
        mag=term.SimpleProductKernel(
          decay=mag_decay_kernel,
          fourier=term.MultiFourierKernel(
            None, np.zeros((1, 2, 1)), None, series_index=series_index, vectorize=True
          ),
        ),
      ),
      self._translate_param,
      self._translate_param_back,
      **kwargs,
    )


def gaussian_weight(delta, delta_mu, delta_sig):
  r"""
  Gaussian distribution of spots/faculae's latitudes.

  Parameters
  ----------
  delta : (n,) ndarray
    Latitudes at which to compute the distribution.
  delta_mu : float
    Center of the distribution.
  delta_sig : float
    Standard deviation of the distribution.

  Returns
  -------
  w : (n,) ndarray
    Weight for each latitude.
  """
  u = (delta - delta_mu) / delta_sig
  ew = -u * u / 2
  ew -= np.max(ew)
  return np.exp(ew)


def symmetric_gaussian_weight(delta, delta_mu, delta_sig):
  r"""
  Symmetric Gaussian distribution of spots/faculae's latitudes.

  Parameters
  ----------
  delta : (n,) ndarray
    Latitudes at which to compute the distribution.
  delta_mu : float
    Center of the distribution (on each hemisphere).
  delta_sig : float
    Standard deviation of the distribution.

  Returns
  -------
  w : (n,) ndarray
    Weight for each latitude.
  """
  u = np.array([(delta - delta_mu) / delta_sig, (delta + delta_mu) / delta_sig])
  ew = -u * u / 2
  ew -= np.max(ew)
  return np.sum(np.exp(ew), axis=0)


class SpotsLatDist:
  r"""
  FENRIR model with spots/faculae appearing randomly on the star,
  with the spots' latitude following an arbitrary distribution.

  The integrals on the latitude's distibution are performed numerically.

  Parameters
  ----------
  nharm : int
    Number of harmonics in the Fourier decomposition.
  ndelta : int
    Number of latitude values for the integrals.
  weight : function
    Weight function describing the spots' latitude distribution.
  brightening : bool
    Whether or not a brightening effect should be included in the model
    (for faculae or mixtures of faculae and spots).
  """

  def __init__(self, nharm, ndelta, weight, brightening=False):
    self._nharm = nharm
    self._nfreq = nharm + 1
    self._ndelta = ndelta
    self._weight = weight
    self._brightening = brightening
    self._spot = SpotsSameLat(nharm, brightening, normalized=False)
    self._u = (np.arange(ndelta) + 0.5) / (ndelta + 1)

  def compute_Fourier(
    self,
    sig_rv_phot,
    sig_rv_cb,
    sig_phot,
    a180,
    bar_i,
    ld_a1,
    ld_a2,
    br_center=None,
    br_a1=None,
    br_a2=None,
    **kwargs,
  ):
    r"""
    Compute the Fourier decomposition of the kernel.

    Parameters
    ----------
    sig_rv_phot, sig_rv_cb, sig_phot : float
      Amplitudes of the three modeled effects
      (photometric and convective blueshift effects on radial-velocity,
      photometric effect)
    a180 : float
      Relative amplitude of the spots/faculae at the opposite longitude.
    bar_i : float
      Complementary inclination of the star (:math:`\bar{i} = \pi/2 - i`).
    ld_a1, ld_a2 : float
      Quadratic limb-darkening coefficients.
    br_center, br_a1, br_a2 : float
      Brightening coefficients (ignored if the brightening is not activated).
    **kwargs :
      Parameters to provide to the weight function.

    Returns
    -------
    ab : (2, nharm + 1, 2, 2) ndarray
      Fourier coefficients, ordered such as ab[0, 2, 0] and ab[0, 2, 1]
      are the cosine coefficients of the second harmonics of the RV and photometric
      timeseries respectively, while ab[1] are the sine coefficients.
    """
    delta_min = max(bar_i, 0) - np.pi / 2
    delta_max = min(bar_i, 0) + np.pi / 2
    delta = delta_min + self._u * (delta_max - delta_min)

    w = self._weight(delta, **kwargs)
    sqw = np.sqrt(w)

    ab_delta = np.array(
      [
        self._spot.compute_Fourier(
          sqwk, sqwk, sqwk, a180, bar_i, dk, ld_a1, ld_a2, br_center, br_a1, br_a2
        )
        for dk, sqwk in zip(delta, sqw)
      ]
    )

    # ab_delta : ndelta x alpha/beta=2 x nfreq x nseries=2
    ab = np.zeros((2, self._nfreq, 2, 2))
    for kharm in range(self._nfreq):
      T = np.tensordot(ab_delta[:, :, kharm], ab_delta[:, :, kharm], axes=(0, 0))
      # T: alpha/beta x nseries x alpha/beta x nseries
      Tb = T[::-1, :, ::-1].copy()
      Tb[1] *= -1
      Tb[:, :, 1] *= -1
      Tf = (T + Tb).reshape(4, 4)
      eigval, eigvec = np.linalg.eigh(Tf)
      U = eigvec[:, ::2] * np.sqrt(np.maximum(0, eigval[::2]))
      ik = U[3] != 0
      Uik = U[:, ik]
      rho = np.sqrt(Uik[1] ** 2 + Uik[3] ** 2)
      U[:, ik] = np.array(
        [
          (Uik[0] * Uik[1] + Uik[2] * Uik[3]) / rho,
          rho,
          (Uik[1] * Uik[2] - Uik[0] * Uik[3]) / rho,
          0 * rho,
        ]
      )
      U[:, U[1] < 0] *= -1
      rho = np.sqrt(np.sum(U[1] ** 2))
      if rho > 0:
        a = U[1, 0] / rho
        b = U[1, 1] / rho
        U[:, 0], U[:, 1] = a * U[:, 0] + b * U[:, 1], b * U[:, 0] - a * U[:, 1]
      ab[:, kharm] = U.reshape((2, 2, 2))

    ab[0, :, 0] *= sig_rv_cb / np.sqrt(np.sum(ab[0, :, 0] ** 2))
    ab[1, :, 0] *= sig_rv_phot / np.sqrt(np.sum(ab[1, :, 0] ** 2))
    ab[:, :, 1] *= sig_phot / np.sqrt(np.sum(ab[:, :, 1] ** 2))

    return ab

  def _translate_param(self, **kwargs):
    r"""
    Helper function to perform a change of variables.
    Intended to be used with :class:`spleaf.term.TransformKernel`.
    """

    param = {key: kwargs.pop(key) for key in list(kwargs) if key.startswith('decay_')}
    param['fourier_P'] = kwargs.pop('P')

    ab = self.compute_Fourier(**kwargs)
    param['fourier_alpha'] = ab[0]
    param['fourier_beta'] = ab[1]

    return param

  def kernel(
    self,
    series_index,
    decay_kernel,
    P,
    sig_rv_phot,
    sig_rv_cb,
    sig_phot,
    a180,
    bar_i,
    ld_a1,
    ld_a2,
    br_center=None,
    br_a1=None,
    br_a2=None,
    **kwargs,
  ):
    r"""
    Generate a S+LEAF kernel corresponding to this FENRIR model,
    to be included in a covariance matrix (:class:`spleaf.cov.Cov`).

    Parameters
    ----------
    series_index : list of ndarrays
      Indices corresponding to each original time series in the merged time series.
    decay_kernel : :class:`spleaf.term.Kernel`
      Kernel describing the (dis)appearing of spots/faculae.
    P : float
      Rotation period.
    sig_rv_phot, sig_rv_cb, sig_phot : float
      Amplitudes of the three modeled effects
      (photometric and convective blueshift effects on radial-velocity,
      photometric effect)
    a180 : float
      Relative amplitude of the spots/faculae at the opposite longitude.
    bar_i : float
      Complementary inclination of the star (:math:`\bar{i} = \pi/2 - i`).
    ld_a1, ld_a2 : float
      Quadratic limb-darkening coefficients.
    br_center, br_a1, br_a2 : float
      Brightening coefficients (ignored if the brightening is not activated).
    **kwargs :
      Parameters to provide to the weight function.

    Returns
    -------
    kernel : :class:`spleaf.term.Kernel`
      S+LEAF kernel corresponding to this FENRIR model.
    """
    kwargs.update(
      dict(
        P=P,
        sig_rv_phot=sig_rv_phot,
        sig_rv_cb=sig_rv_cb,
        sig_phot=sig_phot,
        a180=a180,
        bar_i=bar_i,
        ld_a1=ld_a1,
        ld_a2=ld_a2,
      )
    )

    if self._brightening:
      kwargs.update(dict(br_center=br_center, br_a1=br_a1, br_a2=br_a2))

    kwargs.update(
      {f'decay_{key}': decay_kernel._get_param(key) for key in decay_kernel._param}
    )

    return term.TransformKernel(
      term.SimpleProductKernel(
        decay=decay_kernel,
        fourier=term.MultiFourierKernel(
          1,
          np.zeros((self._nfreq, 2, 2)),
          np.zeros((self._nfreq, 2, 2)),
          series_index=series_index,
          vectorize=True,
        ),
      ),
      self._translate_param,
      **kwargs,
    )


class SpotsLatDistMag:
  r"""
  FENRIR model with spots/faculae appearing randomly on the star,
  with the spots' latitude following an arbitrary distribution,
  and including the effect of the magnetic cycle.

  Parameters
  ----------
  nharm : int
    Number of harmonics in the Fourier decomposition.
  ndelta : int
    Number of latitude values for the integrals.
  weight : function
    Weight function describing the spots' latitude distribution.
  brightening : bool
    Whether or not a brightening effect should be included in the model
    (for faculae or mixtures of faculae and spots).
  normalized : bool
    Whether or not the Fourier decomposition should be normalized.
  """

  def __init__(self, nharm, ndelta, weight, brightening=False, normalized=True):
    self._nharm = nharm
    self._nfreq = nharm + 1
    self._ndelta = ndelta
    self._weight = weight
    self._brightening = brightening
    self._normalized = normalized
    self._spot = SpotsSameLat(nharm, brightening, normalized=False)
    self._u = (np.arange(ndelta) + 0.5) / (ndelta + 1)

  def compute_Fourier(
    self,
    sig_rv_phot,
    sig_rv_cb,
    sig_phot,
    mag_ratio,
    a180,
    bar_i,
    ld_a1,
    ld_a2,
    br_center=None,
    br_a1=None,
    br_a2=None,
    **kwargs,
  ):
    r"""
    Compute the Fourier decomposition of the kernel.

    Parameters
    ----------
    sig_rv_phot, sig_rv_cb, sig_phot : float
      Amplitudes of the three modeled effects
      (photometric and convective blueshift effects on radial-velocity,
      photometric effect)
    mag_ratio : float
      Relative amplitude of the magnetic cycle related activity with respect to the
      rotation period related activity.
    a180 : float
      Relative amplitude of the spots/faculae at the opposite longitude.
    bar_i : float
      Complementary inclination of the star (:math:`\bar{i} = \pi/2 - i`).
    ld_a1, ld_a2 : float
      Quadratic limb-darkening coefficients.
    br_center, br_a1, br_a2 : float
      Brightening coefficients (ignored if the brightening is not activated).
    **kwargs :
      Parameters to provide to the weight function.

    Returns
    -------
    per_ab : (2, nharm + 1, 2, 2) ndarray
      Fourier coefficients for the rotation period related effects,
      ordered such as ab[0, 2, 0] and ab[0, 2, 1]
      are the cosine coefficients of the second harmonics of the RV and photometric
      timeseries respectively, while ab[1] are the sine coefficients.
    mag_ab : (2, 1, 2, 1) ndarray
      Fourier coefficients for the magnetic cycle related effects.
    """
    delta_min = max(bar_i, 0) - np.pi / 2
    delta_max = min(bar_i, 0) + np.pi / 2
    delta = delta_min + self._u * (delta_max - delta_min)

    w = self._weight(delta, **kwargs)
    w /= np.sum(w)
    sqw = np.sqrt(w)

    ab_delta = np.array(
      [
        self._spot.compute_Fourier(
          sqwk, sqwk, sqwk, a180, bar_i, dk, ld_a1, ld_a2, br_center, br_a1, br_a2
        )
        for dk, sqwk in zip(delta, sqw)
      ]
    )
    # ab_delta : ndelta x alpha/beta=2 x nfreq x nseries=2

    mag_alpha = mag_ratio * np.sum(sqw[:, np.newaxis] * ab_delta[:, 0, 0], axis=0)

    per_ab = np.zeros((2, self._nfreq, 2, 2))
    for kharm in range(self._nfreq):
      T = np.tensordot(ab_delta[:, :, kharm], ab_delta[:, :, kharm], axes=(0, 0))
      # T: alpha/beta x nseries x alpha/beta x nseries
      Tb = T[::-1, :, ::-1].copy()
      Tb[1] *= -1
      Tb[:, :, 1] *= -1
      Tf = (T + Tb).reshape(4, 4)
      eigval, eigvec = np.linalg.eigh(Tf)
      U = eigvec[:, ::2] * np.sqrt(np.maximum(0, eigval[::2]))
      ik = U[3] != 0
      Uik = U[:, ik]
      rho = np.sqrt(Uik[1] ** 2 + Uik[3] ** 2)
      U[:, ik] = np.array(
        [
          (Uik[0] * Uik[1] + Uik[2] * Uik[3]) / rho,
          rho,
          (Uik[1] * Uik[2] - Uik[0] * Uik[3]) / rho,
          0 * rho,
        ]
      )
      U[:, U[1] < 0] *= -1
      rho = np.sqrt(np.sum(U[1] ** 2))
      if rho > 0:
        a = U[1, 0] / rho
        b = U[1, 1] / rho
        U[:, 0], U[:, 1] = a * U[:, 0] + b * U[:, 1], b * U[:, 0] - a * U[:, 1]
      per_ab[:, kharm] = U.reshape((2, 2, 2))

    if self._normalized:
      sig_rv_cb /= np.sqrt(np.sum(per_ab[0, :, 0] ** 2) + mag_alpha[0] ** 2)
      sig_rv_phot /= np.sqrt(np.sum(per_ab[1, :, 0] ** 2))
      sig_phot /= np.sqrt(np.sum(per_ab[:, :, 1] ** 2) + mag_alpha[1] ** 2)
    per_ab[0, :, 0] *= sig_rv_cb
    mag_alpha[0] *= sig_rv_cb
    per_ab[1, :, 0] *= sig_rv_phot
    per_ab[:, :, 1] *= sig_phot
    mag_alpha[1] *= sig_phot
    return (per_ab, np.array([[mag_alpha], [2 * [0]]]))

  def _translate_param(self, **kwargs):
    r"""
    Helper function to perform a change of variables.
    Intended to be used with :class:`spleaf.term.TransformKernel`.
    """
    param = {
      f'per_{key}': kwargs.pop(key) for key in list(kwargs) if key.startswith('decay_')
    }

    param.update(
      {key: kwargs.pop(key) for key in list(kwargs) if key.startswith('mag_decay_')}
    )

    param['per_fourier_P'] = kwargs.pop('P')

    per_ab, mag_ab = self.compute_Fourier(**kwargs)
    param['per_fourier_alpha'] = per_ab[0]
    param['per_fourier_beta'] = per_ab[1]
    param['mag_fourier_alpha'] = mag_ab[0]

    return param

  def kernel(
    self,
    series_index,
    decay_kernel,
    mag_decay_kernel,
    P,
    sig_rv_phot,
    sig_rv_cb,
    sig_phot,
    mag_ratio,
    a180,
    bar_i,
    ld_a1,
    ld_a2,
    br_center=None,
    br_a1=None,
    br_a2=None,
    **kwargs,
  ):
    r"""
    Generate a S+LEAF kernel corresponding to this FENRIR model,
    to be included in a covariance matrix (:class:`spleaf.cov.Cov`).

    Parameters
    ----------
    series_index : list of ndarrays
      Indices corresponding to each original time series in the merged time series.
    decay_kernel : :class:`spleaf.term.Kernel`
      Kernel describing the (dis)appearing of spots/faculae.
    mag_decay_kernel : :class:`spleaf.term.Kernel`
      Kernel describing the variation of spots/faculae appearing rate
      due to the magnetic cycle.
    P : float
      Rotation period.
    sig_rv_phot, sig_rv_cb, sig_phot : float
      Amplitudes of the three modeled effects
      (photometric and convective blueshift effects on radial-velocity,
      photometric effect)
    mag_ratio : float
      Relative amplitude of the magnetic cycle related activity with respect to the
      rotation period related activity.
    a180 : float
      Relative amplitude of the spots/faculae at the opposite longitude.
    bar_i : float
      Complementary inclination of the star (:math:`\bar{i} = \pi/2 - i`).
    ld_a1, ld_a2 : float
      Quadratic limb-darkening coefficients.
    br_center, br_a1, br_a2 : float
      Brightening coefficients (ignored if the brightening is not activated).
    **kwargs :
      Parameters to provide to the weight function.

    Returns
    -------
    kernel : :class:`spleaf.term.Kernel`
      S+LEAF kernel corresponding to this FENRIR model.
    """
    kwargs.update(
      dict(
        P=P,
        sig_rv_phot=sig_rv_phot,
        sig_rv_cb=sig_rv_cb,
        sig_phot=sig_phot,
        mag_ratio=mag_ratio,
        a180=a180,
        bar_i=bar_i,
        ld_a1=ld_a1,
        ld_a2=ld_a2,
      )
    )

    if self._brightening:
      kwargs.update(dict(br_center=br_center, br_a1=br_a1, br_a2=br_a2))

    kwargs.update(
      {f'decay_{key}': decay_kernel._get_param(key) for key in decay_kernel._param}
    )

    kwargs.update(
      {
        f'mag_decay_{key}': mag_decay_kernel._get_param(key)
        for key in mag_decay_kernel._param
      }
    )

    return term.TransformKernel(
      term.SimpleSumKernel(
        per=term.SimpleProductKernel(
          decay=decay_kernel,
          fourier=term.MultiFourierKernel(
            1,
            np.zeros((self._nfreq, 2, 2)),
            np.zeros((self._nfreq, 2, 2)),
            series_index=series_index,
            vectorize=True,
          ),
        ),
        mag=term.SimpleProductKernel(
          decay=mag_decay_kernel,
          fourier=term.MultiFourierKernel(
            None, np.zeros((1, 2, 1)), None, series_index=series_index, vectorize=True
          ),
        ),
      ),
      self._translate_param,
      **kwargs,
    )


class SpotsLatDistInterpolator:
  r"""
  FENRIR model interpolating the results of :class:`SpotsLatDist`,
  to improve computational efficiency.

  This class should be instanciated using either :func:`create`
  to generate the grid on which to interpolate,
  or :func:`load` to load a previously computed grid.
  """

  def __init__(self):
    self._nharm = None
    self._nfreq = None
    self._grid = None
    self._ndim = None
    self._listdim = None
    self._ab = None

  @staticmethod
  def create(basename, nharm, ndelta, weight, fixed, grid, brightening=False):
    r"""
    Compute the Fourier decomposition on a grid using :func:`SpotsLatDist.compute_Fourier`.

    Parameters
    ----------
    basename : str
      Basename for the files used to store the grid.
    nharm : int
      Number of harmonics in the Fourier decomposition.
    ndelta : int
      Number of latitude values for the integrals.
    weight : function
      Weight function describing the spots' latitude distribution.
    fixed : dict
      Dictionary of fixed parameters and their values,
      to be passed to :func:`SpotsLatDist.compute_Fourier`.
    grid : dict
      Dictionary associating to each explored parameters
      the ndarray of values to be taken by this parameters
      to generate the grid.
    brightening : bool
      Whether or not a brightening effect should be included in the model
      (for faculae or mixtures of faculae and spots).
    """
    spi = SpotsLatDistInterpolator()
    spi._grid = grid
    spi._nharm = nharm
    spi._nfreq = nharm + 1
    with open(basename + '_grid.pkl', 'wb') as f:
      pickle.dump((spi._grid, spi._nharm), f)
    spi._ndim = len(spi._grid)
    spi._listdim = np.arange(spi._ndim)
    sp = SpotsLatDist(nharm, ndelta, weight, brightening)
    gshape = tuple(spi._grid[key].size for key in spi._grid)
    spi._ab = np.memmap(
      basename + '_mmap.dat',
      dtype=float,
      mode='w+',
      shape=gshape + (2, spi._nfreq, 2, 2),
    )
    ntot = np.prod(gshape)
    for count, inds in enumerate(np.ndindex(gshape)):
      if count % (ntot // 1000) == 0:
        print(f'\r{count/ntot*100:.1f}%', end=' ')
      kwargs = {key: spi._grid[key][ik] for key, ik in zip(spi._grid, inds)}
      kwargs.update(fixed)
      spi._ab[inds] = sp.compute_Fourier(1, 1, 1, 0, **kwargs)
    spi._ab.flush()
    print('\r100%')
    print('done')
    return spi

  @staticmethod
  def load(basename):
    r"""
    Load a previously computed grid.

    Parameters
    ----------
    basename : str
      Basename for the files used to store the grid.
    """
    spi = SpotsLatDistInterpolator()
    with open(basename + '_grid.pkl', 'rb') as f:
      spi._grid, spi._nharm = pickle.load(f)
    spi._nfreq = spi._nharm + 1
    spi._ndim = len(spi._grid)
    spi._listdim = np.arange(spi._ndim)
    gshape = tuple(g.size for g in spi._grid.values())
    spi._ab = np.memmap(
      basename + '_mmap.dat',
      dtype=float,
      mode='r',
      shape=gshape + (2, spi._nfreq, 2, 2),
    )
    return spi

  def compute_Fourier(self, sig_rv_phot, sig_rv_cb, sig_phot, a180, **kwargs):
    r"""
    Compute the Fourier decomposition of the kernel.

    Parameters
    ----------
    sig_rv_phot, sig_rv_cb, sig_phot : float
      Amplitudes of the three modeled effects
      (photometric and convective blueshift effects on radial-velocity,
      photometric effect)
    a180 : float
      Relative amplitude of the spots/faculae at the opposite longitude.
    **kwargs :
      Parameters at which to compute the interpolation.

    Returns
    -------
    ab : (2, nharm + 1, 2, 2) ndarray
      Fourier coefficients, ordered such as ab[0, 2, 0] and ab[0, 2, 1]
      are the cosine coefficients of the second harmonics of the RV and photometric
      timeseries respectively, while ab[1] are the sine coefficients.
    """
    self._params = dict(
      sig_rv_phot=sig_rv_phot, sig_rv_cb=sig_rv_cb, sig_phot=sig_phot, a180=a180
    )
    self._params.update(kwargs)
    self._inds = [
      max(
        1,
        min(
          self._grid[key].size - 1,
          np.searchsorted(self._grid[key], kwargs[key], 'right'),
        ),
      )
      for key in self._grid
    ]
    self._w = np.ones(self._ndim * [2])
    for k, key in enumerate(self._grid):
      g = self._grid[key]
      i = self._inds[k]
      x = kwargs[key]
      self._w[k * (slice(None),) + (0,)] *= g[i] - x
      self._w[k * (slice(None),) + (1,)] *= x - g[i - 1]
    self._unscaled_ab = np.tensordot(
      self._ab[tuple(slice(i - 1, i + 1) for i in self._inds)],
      self._w,
      axes=(self._listdim, self._listdim),
    )
    # Spot at opposite longitude
    self._unscaled_ab_180 = self._unscaled_ab.copy()
    self._unscaled_ab_180[:, ::2] *= 1 + a180
    self._unscaled_ab_180[:, 1::2] *= 1 - a180
    # Scaling
    self._scaled_ab = self._unscaled_ab_180.copy()
    self._scaled_ab[0, :, 0] *= sig_rv_cb / np.sqrt(
      np.sum(self._scaled_ab[0, :, 0] ** 2)
    )
    self._scaled_ab[1, :, 0] *= sig_rv_phot / np.sqrt(
      np.sum(self._scaled_ab[1, :, 0] ** 2)
    )
    self._scaled_ab[:, :, 1] *= sig_phot / np.sqrt(
      np.sum(self._scaled_ab[:, :, 1] ** 2)
    )
    return self._scaled_ab

  def compute_Fourier_back(self, grad):
    r"""
    Backward propagation of the gradient for :func:`compute_Fourier`.
    """
    grad_params = dict()

    # Scaling
    # self._scaled_ab = self._unscaled_ab_180.copy()
    # self._scaled_ab[0, :,
    #   0] *= sig_rv_cb / np.sqrt(np.sum(self._scaled_ab[0, :, 0]**2))
    cgradc = np.sum(self._scaled_ab[0, :, 0] * grad[0, :, 0])
    grad_params['sig_rv_cb'] = cgradc / self._params['sig_rv_cb']
    s = np.sqrt(np.sum(self._unscaled_ab_180[0, :, 0] ** 2))
    grad_s = -cgradc / s
    grad[0, :, 0] *= self._params['sig_rv_cb'] / s
    grad[0, :, 0] += self._unscaled_ab_180[0, :, 0] * grad_s / s

    # self._scaled_ab[1, :,
    #   0] *= sig_rv_phot / np.sqrt(np.sum(self._scaled_ab[1, :, 0]**2))
    cgradc = np.sum(self._scaled_ab[1, :, 0] * grad[1, :, 0])
    grad_params['sig_rv_phot'] = cgradc / self._params['sig_rv_phot']
    s = np.sqrt(np.sum(self._unscaled_ab_180[1, :, 0] ** 2))
    grad_s = -cgradc / s
    grad[1, :, 0] *= self._params['sig_rv_phot'] / s
    grad[1, :, 0] += self._unscaled_ab_180[1, :, 0] * grad_s / s

    # self._scaled_ab[:, :,
    #   1] *= sig_phot / np.sqrt(np.sum(self._scaled_ab[:, :, 1]**2))
    cgradc = np.sum(self._scaled_ab[:, :, 1] * grad[:, :, 1])
    grad_params['sig_phot'] = cgradc / self._params['sig_phot']
    s = np.sqrt(np.sum(self._unscaled_ab_180[:, :, 1] ** 2))
    grad_s = -cgradc / s
    grad[:, :, 1] *= self._params['sig_phot'] / s
    grad[:, :, 1] += self._unscaled_ab_180[:, :, 1] * grad_s / s

    # Spot at opposite longitude
    # self._unscaled_ab_180 = self._unscaled_ab.copy()
    # self._unscaled_ab_180[:, ::2] *= 1 + a180
    # self._unscaled_ab_180[:, 1::2] *= 1 - a180
    grad_params['a180'] = np.sum(self._unscaled_ab[:, ::2] * grad[:, ::2]) - np.sum(
      self._unscaled_ab[:, 1::2] * grad[:, 1::2]
    )
    grad[:, ::2] *= 1 + self._params['a180']
    grad[:, 1::2] *= 1 - self._params['a180']

    # self._unscaled_ab = np.tensordot(self._ab[tuple(
    #   slice(i - 1, i + 1) for i in self._inds)],
    #   self._w,
    #   axes=(self._listdim, self._listdim))
    grad_w = np.tensordot(
      grad,
      self._ab[tuple(slice(i - 1, i + 1) for i in self._inds)],
      axes=([0, 1, 2, 3], [-4, -3, -2, -1]),
    )
    # self._w = np.ones(self._ndim * [2])
    # for k, key in enumerate(self._grid):
    #   g = self._grid[key]
    #   i = self._inds[k]
    #   x = kwargs[key]
    #   self._w[k * (slice(None), ) + (0, )] *= g[i] - x
    #   self._w[k * (slice(None), ) + (1, )] *= x - g[i - 1]
    for k, key in enumerate(self._grid):
      dw = np.ones(self._ndim * [2])
      dw[k * (slice(None),) + (0,)] = -1
      for l, ley in enumerate(self._grid):
        if l == k:
          continue
        g = self._grid[ley]
        i = self._inds[l]
        x = self._params[ley]
        dw[l * (slice(None),) + (0,)] *= g[i] - x
        dw[l * (slice(None),) + (1,)] *= x - g[i - 1]
      grad_params[key] = np.sum(grad_w * dw)

    return grad_params

  def _translate_param(self, **kwargs):
    r"""
    Helper function to perform a change of variables.
    Intended to be used with :class:`spleaf.term.TransformKernel`.
    """
    param = {key: kwargs.pop(key) for key in list(kwargs) if key.startswith('decay_')}
    param['fourier_P'] = kwargs.pop('P')

    ab = self.compute_Fourier(**kwargs)
    param['fourier_alpha'] = ab[0]
    param['fourier_beta'] = ab[1]

    return param

  def _translate_param_back(self, grad):
    r"""
    Backward propagation of the gradient for :func:`_translate_param`.
    """
    grad_param = grad.copy()
    grad_param['P'] = grad_param.pop('fourier_P')
    grad_ab = np.array(
      [grad_param.pop('fourier_alpha'), grad_param.pop('fourier_beta')]
    )
    grad_param.update(self.compute_Fourier_back(grad_ab))

    return grad_param

  def kernel(
    self,
    series_index,
    decay_kernel,
    P,
    sig_rv_phot,
    sig_rv_cb,
    sig_phot,
    a180,
    **kwargs,
  ):
    r"""
    Generate a S+LEAF kernel corresponding to this FENRIR model,
    to be included in a covariance matrix (:class:`spleaf.cov.Cov`).

    Parameters
    ----------
    series_index : list of ndarrays
      Indices corresponding to each original time series in the merged time series.
    decay_kernel : :class:`spleaf.term.Kernel`
      Kernel describing the (dis)appearing of spots/faculae.
    P : float
      Rotation period.
    sig_rv_phot, sig_rv_cb, sig_phot : float
      Amplitudes of the three modeled effects
      (photometric and convective blueshift effects on radial-velocity,
      photometric effect)
    a180 : float
      Relative amplitude of the spots/faculae at the opposite longitude.
    **kwargs :
      Parameters at which to compute the interpolation.

    Returns
    -------
    kernel : :class:`spleaf.term.Kernel`
      S+LEAF kernel corresponding to this FENRIR model.
    """
    kwargs.update(
      dict(
        P=P, sig_rv_phot=sig_rv_phot, sig_rv_cb=sig_rv_cb, sig_phot=sig_phot, a180=a180
      )
    )

    kwargs.update(
      {f'decay_{key}': decay_kernel._get_param(key) for key in decay_kernel._param}
    )

    return term.TransformKernel(
      term.SimpleProductKernel(
        decay=decay_kernel,
        fourier=term.MultiFourierKernel(
          1,
          np.zeros((self._nfreq, 2, 2)),
          np.zeros((self._nfreq, 2, 2)),
          series_index=series_index,
          vectorize=True,
        ),
      ),
      self._translate_param,
      self._translate_param_back,
      **kwargs,
    )


class SpotsLatDistMagInterpolator:
  r"""
  FENRIR model interpolating the results of :class:`SpotsLatDistMag`,
  to improve computational efficiency.

  This class should be instanciated using either :func:`create`
  to generate the grid on which to interpolate,
  or :func:`load` to load a previously computed grid.
  """

  def __init__(self):
    self._nharm = None
    self._nfreq = None
    self._grid = None
    self._ndim = None
    self._listdim = None
    self._per_ab = None
    self._mag_alpha = None

  @staticmethod
  def create(basename, nharm, ndelta, weight, fixed, grid, brightening=False):
    r"""
    Compute the Fourier decomposition on a grid using :func:`SpotsLatDistMag.compute_Fourier`.

    Parameters
    ----------
    basename : str
      Basename for the files used to store the grid.
    nharm : int
      Number of harmonics in the Fourier decomposition.
    ndelta : int
      Number of latitude values for the integrals.
    weight : function
      Weight function describing the spots' latitude distribution.
    fixed : dict
      Dictionary of fixed parameters and their values,
      to be passed to :func:`SpotsLatDistMag.compute_Fourier`.
    grid : dict
      Dictionary associating to each explored parameters
      the ndarray of values to be taken by this parameters
      to generate the grid.
    brightening : bool
      Whether or not a brightening effect should be included in the model
      (for faculae or mixtures of faculae and spots).
    """
    spi = SpotsLatDistMagInterpolator()
    spi._grid = grid
    spi._nharm = nharm
    spi._nfreq = nharm + 1
    with open(basename + '_grid.pkl', 'wb') as f:
      pickle.dump((spi._grid, spi._nharm), f)
    spi._ndim = len(spi._grid)
    spi._listdim = np.arange(spi._ndim)
    sp = SpotsLatDistMag(nharm, ndelta, weight, brightening, normalized=False)
    gshape = tuple(spi._grid[key].size for key in spi._grid)
    spi._per_ab = np.memmap(
      basename + '_per_mmap.dat',
      dtype=float,
      mode='w+',
      shape=gshape + (2, spi._nfreq, 2, 2),
    )
    spi._mag_alpha = np.memmap(
      basename + '_mag_mmap.dat', dtype=float, mode='w+', shape=gshape + (2,)
    )
    ntot = np.prod(gshape)
    for count, inds in enumerate(np.ndindex(gshape)):
      if count % (ntot // 1000) == 0:
        print(f'\r{count/ntot*100:.1f}%', end=' ')
      kwargs = {key: spi._grid[key][ik] for key, ik in zip(spi._grid, inds)}
      kwargs.update(fixed)
      spi._per_ab[inds], mc = sp.compute_Fourier(1, 1, 1, 1, 0, **kwargs)
      spi._mag_alpha[inds] = mc[0, 0]
    spi._per_ab.flush()
    spi._mag_alpha.flush()
    print('\r100%')
    print('done')
    return spi

  @staticmethod
  def load(basename):
    r"""
    Load a previously computed grid.

    Parameters
    ----------
    basename : str
      Basename for the files used to store the grid.
    """
    spi = SpotsLatDistMagInterpolator()
    with open(basename + '_grid.pkl', 'rb') as f:
      spi._grid, spi._nharm = pickle.load(f)
    spi._nfreq = spi._nharm + 1
    spi._ndim = len(spi._grid)
    spi._listdim = np.arange(spi._ndim)
    gshape = tuple(g.size for g in spi._grid.values())
    spi._per_ab = np.memmap(
      basename + '_per_mmap.dat',
      dtype=float,
      mode='r',
      shape=gshape + (2, spi._nfreq, 2, 2),
    )
    spi._mag_alpha = np.memmap(
      basename + '_mag_mmap.dat', dtype=float, mode='r', shape=gshape + (2,)
    )
    return spi

  def compute_Fourier(
    self, sig_rv_phot, sig_rv_cb, sig_phot, mag_ratio, a180, **kwargs
  ):
    r"""
    Compute the Fourier decomposition of the kernel.

    Parameters
    ----------
    sig_rv_phot, sig_rv_cb, sig_phot : float
      Amplitudes of the three modeled effects
      (photometric and convective blueshift effects on radial-velocity,
      photometric effect)
    mag_ratio : float
      Relative amplitude of the magnetic cycle related activity with respect to the
      rotation period related activity.
    a180 : float
      Relative amplitude of the spots/faculae at the opposite longitude.
    **kwargs :
      Parameters at which to compute the interpolation.

    Returns
    -------
    per_ab : (2, nharm + 1, 2, 2) ndarray
      Fourier coefficients for the rotation period related effects,
      ordered such as ab[0, 2, 0] and ab[0, 2, 1]
      are the cosine coefficients of the second harmonics of the RV and photometric
      timeseries respectively, while ab[1] are the sine coefficients.
    mag_ab : (2, 1, 2, 1) ndarray
      Fourier coefficients for the magnetic cycle related effects.
    """
    self._params = dict(
      sig_rv_phot=sig_rv_phot,
      sig_rv_cb=sig_rv_cb,
      sig_phot=sig_phot,
      mag_ratio=mag_ratio,
      a180=a180,
    )
    self._params.update(kwargs)
    self._inds = [
      max(
        1,
        min(
          self._grid[key].size - 1,
          np.searchsorted(self._grid[key], kwargs[key], 'right'),
        ),
      )
      for key in self._grid
    ]
    self._w = np.ones(self._ndim * [2])
    for k, key in enumerate(self._grid):
      g = self._grid[key]
      i = self._inds[k]
      x = kwargs[key]
      self._w[k * (slice(None),) + (0,)] *= g[i] - x
      self._w[k * (slice(None),) + (1,)] *= x - g[i - 1]
    self._unscaled_per_ab = np.tensordot(
      self._per_ab[tuple(slice(i - 1, i + 1) for i in self._inds)],
      self._w,
      axes=(self._listdim, self._listdim),
    )
    self._unscaled_mag_alpha = np.tensordot(
      self._mag_alpha[tuple(slice(i - 1, i + 1) for i in self._inds)],
      self._w,
      axes=(self._listdim, self._listdim),
    )
    # Spot at opposite longitude
    self._unscaled_per_ab_180 = self._unscaled_per_ab.copy()
    self._unscaled_mag_alpha_180 = self._unscaled_mag_alpha.copy()
    self._unscaled_per_ab_180[:, ::2] *= 1 + a180
    self._unscaled_per_ab_180[:, 1::2] *= 1 - a180
    self._unscaled_mag_alpha_180 *= mag_ratio * (1 + a180)

    # Scaling
    self._scaled_per_ab = self._unscaled_per_ab_180.copy()
    self._scaled_mag_alpha = self._unscaled_mag_alpha_180.copy()

    scale_cb = sig_rv_cb / np.sqrt(
      np.sum(self._scaled_per_ab[0, :, 0] ** 2) + self._scaled_mag_alpha[0] ** 2
    )
    self._scaled_per_ab[0, :, 0] *= scale_cb
    self._scaled_mag_alpha[0] *= scale_cb

    self._scaled_per_ab[1, :, 0] *= sig_rv_phot / np.sqrt(
      np.sum(self._scaled_per_ab[1, :, 0] ** 2)
    )

    scale_phot = sig_phot / np.sqrt(
      np.sum(self._scaled_per_ab[:, :, 1] ** 2) + self._scaled_mag_alpha[1] ** 2
    )
    self._scaled_per_ab[:, :, 1] *= scale_phot
    self._scaled_mag_alpha[1] *= scale_phot

    return (
      self._scaled_per_ab,
      np.array([[self._scaled_mag_alpha], [2 * [0]]])[..., np.newaxis],
    )

  def compute_Fourier_back(self, grad_per, grad_mag):
    r"""
    Backward propagation of the gradient for :func:`compute_Fourier`.
    """
    grad_params = dict()

    grad_mag = grad_mag[0, 0, :, 0]

    # Scaling
    # self._scaled_per_ab = self._unscaled_per_ab_180.copy()
    # self._scaled_mag_alpha = self._unscaled_mag_alpha_180.copy()

    # scale_cb = sig_rv_cb / np.sqrt(
    #   np.sum(self._scaled_per_ab[0, :, 0]**2) +
    #   self._scaled_mag_alpha[0]**2)
    # self._scaled_per_ab[0, :, 0] *= scale_cb
    # self._scaled_mag_alpha[0] *= scale_cb
    cgradc = (
      np.sum(self._scaled_per_ab[0, :, 0] * grad_per[0, :, 0])
      + self._scaled_mag_alpha[0] * grad_mag[0]
    )
    grad_params['sig_rv_cb'] = cgradc / self._params['sig_rv_cb']
    s = np.sqrt(
      np.sum(self._unscaled_per_ab_180[0, :, 0] ** 2)
      + self._unscaled_mag_alpha_180[0] ** 2
    )
    grad_s = -cgradc / s
    grad_per[0, :, 0] *= self._params['sig_rv_cb'] / s
    grad_per[0, :, 0] += self._unscaled_per_ab_180[0, :, 0] * grad_s / s
    grad_mag[0] *= self._params['sig_rv_cb'] / s
    grad_mag[0] += self._unscaled_mag_alpha_180[0] * grad_s / s

    # self._scaled_per_ab[1, :,
    #   0] *= sig_rv_phot / np.sqrt(np.sum(self._scaled_per_ab[1, :, 0]**2))
    cgradc = np.sum(self._scaled_per_ab[1, :, 0] * grad_per[1, :, 0])
    grad_params['sig_rv_phot'] = cgradc / self._params['sig_rv_phot']
    s = np.sqrt(np.sum(self._unscaled_per_ab_180[1, :, 0] ** 2))
    grad_s = -cgradc / s
    grad_per[1, :, 0] *= self._params['sig_rv_phot'] / s
    grad_per[1, :, 0] += self._unscaled_per_ab_180[1, :, 0] * grad_s / s

    # scale_phot = sig_phot / np.sqrt(
    #   np.sum(self._scaled_per_ab[:,:,1]**2) + self._scaled_mag_alpha[1]**2)
    # self._scaled_per_ab[:,:,1] *= scale_phot
    # self._scaled_mag_alpha[1] *= scale_phot
    cgradc = (
      np.sum(self._scaled_per_ab[:, :, 1] * grad_per[:, :, 1])
      + self._scaled_mag_alpha[1] * grad_mag[1]
    )
    grad_params['sig_phot'] = cgradc / self._params['sig_phot']
    s = np.sqrt(
      np.sum(self._unscaled_per_ab_180[:, :, 1] ** 2)
      + self._unscaled_mag_alpha_180[1] ** 2
    )
    grad_s = -cgradc / s
    grad_per[:, :, 1] *= self._params['sig_phot'] / s
    grad_per[:, :, 1] += self._unscaled_per_ab_180[:, :, 1] * grad_s / s
    grad_mag[1] *= self._params['sig_phot'] / s
    grad_mag[1] += self._unscaled_mag_alpha_180[1] * grad_s / s

    # Spot at opposite longitude
    # self._unscaled_per_ab_180 = self._unscaled_per_ab.copy()
    # self._unscaled_mag_alpha_180 = self._unscaled_mag_alpha.copy()
    # self._unscaled_per_ab_180[:, ::2] *= 1 + a180
    # self._unscaled_per_ab_180[:, 1::2] *= 1 - a180
    # self._unscaled_mag_alpha_180 *= mag_ratio * (1 + a180)
    smag = np.sum(self._unscaled_mag_alpha * grad_mag)
    grad_params['a180'] = (
      np.sum(self._unscaled_per_ab[:, ::2] * grad_per[:, ::2])
      + self._params['mag_ratio'] * smag
      - np.sum(self._unscaled_per_ab[:, 1::2] * grad_per[:, 1::2])
    )
    grad_params['mag_ratio'] = (1 + self._params['a180']) * smag
    grad_per[:, ::2] *= 1 + self._params['a180']
    grad_per[:, 1::2] *= 1 - self._params['a180']
    grad_mag *= self._params['mag_ratio'] * (1 + self._params['a180'])

    # self._unscaled_per_ab = np.tensordot(self._per_ab[tuple(
    #   slice(i - 1, i + 1) for i in self._inds)],
    #   self._w,
    #   axes=(self._listdim, self._listdim))
    # self._unscaled_mag_alpha = np.tensordot(self._mag_alpha[tuple(
    #   slice(i - 1, i + 1) for i in self._inds)],
    #   self._w,
    #   axes=(self._listdim, self._listdim))
    grad_w = np.tensordot(
      grad_per,
      self._per_ab[tuple(slice(i - 1, i + 1) for i in self._inds)],
      axes=([0, 1, 2, 3], [-4, -3, -2, -1]),
    ) + np.tensordot(
      grad_mag,
      self._mag_alpha[tuple(slice(i - 1, i + 1) for i in self._inds)],
      axes=(0, -1),
    )

    # self._w = np.ones(self._ndim * [2])
    # for k, key in enumerate(self._grid):
    #   g = self._grid[key]
    #   i = self._inds[k]
    #   x = kwargs[key]
    #   self._w[k * (slice(None), ) + (0, )] *= g[i] - x
    #   self._w[k * (slice(None), ) + (1, )] *= x - g[i - 1]
    for k, key in enumerate(self._grid):
      dw = np.ones(self._ndim * [2])
      dw[k * (slice(None),) + (0,)] = -1
      for l, ley in enumerate(self._grid):
        if l == k:
          continue
        g = self._grid[ley]
        i = self._inds[l]
        x = self._params[ley]
        dw[l * (slice(None),) + (0,)] *= g[i] - x
        dw[l * (slice(None),) + (1,)] *= x - g[i - 1]
      grad_params[key] = np.sum(grad_w * dw)

    return grad_params

  def _translate_param(self, **kwargs):
    r"""
    Helper function to perform a change of variables.
    Intended to be used with :class:`spleaf.term.TransformKernel`.
    """
    param = {
      f'per_{key}': kwargs.pop(key) for key in list(kwargs) if key.startswith('decay_')
    }

    param.update(
      {key: kwargs.pop(key) for key in list(kwargs) if key.startswith('mag_decay_')}
    )

    param['per_fourier_P'] = kwargs.pop('P')

    per_ab, mag_ab = self.compute_Fourier(**kwargs)
    param['per_fourier_alpha'] = per_ab[0]
    param['per_fourier_beta'] = per_ab[1]
    param['mag_fourier_alpha'] = mag_ab[0]

    return param

  def _translate_param_back(self, grad):
    r"""
    Backward propagation of the gradient for :func:`_translate_param`.
    """
    grad_param = grad.copy()
    grad_param['P'] = grad_param.pop('per_fourier_P')
    grad_param.update(
      {
        key[4:]: grad_param.pop(key)
        for key in list(grad_param)
        if key.startswith('per_decay_')
      }
    )
    grad_per = np.array(
      [grad_param.pop('per_fourier_alpha'), grad_param.pop('per_fourier_beta')]
    )
    grad_mag = np.array([grad_param.pop('mag_fourier_alpha'), [2 * [[0]]]])
    grad_param.update(self.compute_Fourier_back(grad_per, grad_mag))

    return grad_param

  def kernel(
    self,
    series_index,
    decay_kernel,
    mag_decay_kernel,
    P,
    sig_rv_phot,
    sig_rv_cb,
    sig_phot,
    mag_ratio,
    a180,
    **kwargs,
  ):
    r"""
    Generate a S+LEAF kernel corresponding to this FENRIR model,
    to be included in a covariance matrix (:class:`spleaf.cov.Cov`).

    Parameters
    ----------
    series_index : list of ndarrays
      Indices corresponding to each original time series in the merged time series.
    decay_kernel : :class:`spleaf.term.Kernel`
      Kernel describing the (dis)appearing of spots/faculae.
    mag_decay_kernel : :class:`spleaf.term.Kernel`
      Kernel describing the variation of spots/faculae appearing rate
      due to the magnetic cycle.
    P : float
      Rotation period.
    sig_rv_phot, sig_rv_cb, sig_phot : float
      Amplitudes of the three modeled effects
      (photometric and convective blueshift effects on radial-velocity,
      photometric effect)
    mag_ratio : float
      Relative amplitude of the magnetic cycle related activity with respect to the
      rotation period related activity.
    a180 : float
      Relative amplitude of the spots/faculae at the opposite longitude.
    bar_i : float
      Complementary inclination of the star (:math:`\bar{i} = \pi/2 - i`).
    ld_a1, ld_a2 : float
      Quadratic limb-darkening coefficients.
    br_center, br_a1, br_a2 : float
      Brightening coefficients (ignored if the brightening is not activated).
    **kwargs :
      Parameters at which to compute the interpolation.

    Returns
    -------
    kernel : :class:`spleaf.term.Kernel`
      S+LEAF kernel corresponding to this FENRIR model.
    """
    kwargs.update(
      dict(
        P=P,
        sig_rv_phot=sig_rv_phot,
        sig_rv_cb=sig_rv_cb,
        sig_phot=sig_phot,
        mag_ratio=mag_ratio,
        a180=a180,
      )
    )

    kwargs.update(
      {f'decay_{key}': decay_kernel._get_param(key) for key in decay_kernel._param}
    )

    kwargs.update(
      {
        f'mag_decay_{key}': mag_decay_kernel._get_param(key)
        for key in mag_decay_kernel._param
      }
    )

    return term.TransformKernel(
      term.SimpleSumKernel(
        per=term.SimpleProductKernel(
          decay=decay_kernel,
          fourier=term.MultiFourierKernel(
            1,
            np.zeros((self._nfreq, 2, 2)),
            np.zeros((self._nfreq, 2, 2)),
            series_index=series_index,
            vectorize=True,
          ),
        ),
        mag=term.SimpleProductKernel(
          decay=mag_decay_kernel,
          fourier=term.MultiFourierKernel(
            None, np.zeros((1, 2, 1)), None, series_index=series_index, vectorize=True
          ),
        ),
      ),
      self._translate_param,
      self._translate_param_back,
      **kwargs,
    )


class MultiDist:
  r"""
  FENRIR model mixing several distributions of spots/faculae,
  with some shared parameters (e.g. the star's inclination).

  Parameters
  ----------
  **kwargs:
    Spots/faculae distributions (indexed by their names).
  """

  def __init__(self, **kwargs):
    self._shared = []
    self._dists = kwargs

  def _translate_param(self, **kwargs):
    r"""
    Helper function to perform a change of variables.
    Intended to be used with :class:`spleaf.term.TransformKernel`.
    """
    param = kwargs.copy()
    for par in self._shared:
      value = param.pop(par)
      for dist in self._dists:
        param[f'{dist}_{par}'] = value

    return param

  def _translate_param_back(self, grad):
    r"""
    Backward propagation of the gradient for :func:`_translate_param`.
    """
    grad_params = grad.copy()
    for par in self._shared:
      grad_params[par] = np.sum(
        [grad_params.pop(f'{dist}_{par}') for dist in self._dists]
      )
    return grad_params

  def kernel(self, series_index, shared_params, **kwargs):
    r"""
    Generate a S+LEAF kernel corresponding to this FENRIR model,
    to be included in a covariance matrix (:class:`spleaf.cov.Cov`).

    Parameters
    ----------
    series_index : list of ndarrays
      Indices corresponding to each original time series in the merged time series.
    shared_params: dict
      Parameters that are shared between all spot distributions
      (e.g. the inclination :math:`\bar{i}`)
    **kwargs:
      Dictionaries of specific parameters for each spot distribution.
    """
    self._shared = list(shared_params.keys())

    kernels = {
      dist: self._dists[dist].kernel(series_index, **shared_params, **kwargs[dist])
      for dist in self._dists
    }
    params = shared_params.copy()
    for dist in self._dists:
      params.update(
        {
          f'{dist}_{par}': kernels[dist]._get_param(par)
          for par in kernels[dist]._param
          if par not in self._shared
        }
      )

    return term.TransformKernel(
      term.SimpleSumKernel(**kernels),
      self._translate_param,
      self._translate_param_back,
      **params,
    )
