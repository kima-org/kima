# -*- coding: utf-8 -*-

# Copyright 2019-2023 Jean-Baptiste Delisle
#
# This file is part of spleaf.
#
# spleaf is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# spleaf is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with spleaf.  If not, see <http://www.gnu.org/licenses/>.

__all__ = [
  'SpotsSameLat', 'gaussian_weight', 'symmetric_gaussian_weight',
  'SpotsLatDist', 'SpotsLatDistMag', 'SpotsLatDistInterpolator',
  'SpotsLatDistMagInterpolator'
]

import numpy as np
from . import term
import pickle


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
    self._bar_i = 0
    self._delta = 0

  def _rv_cb(self):
    alpha = np.zeros(3)
    alpha[0] = np.sin(self._bar_i)**2 * np.sin(self._delta)**2 + 0.5 * np.cos(
      self._bar_i)**2 * np.cos(self._delta)**2  # const
    alpha[1] = 0.5 * np.sin(2 * self._bar_i) * np.sin(
      2 * self._delta)  # cosphi
    alpha[2] = 0.5 * np.cos(self._bar_i)**2 * np.cos(self._delta)**2  # cos2phi
    return (alpha)

  def _rv_phot(self):
    beta = np.zeros(3)
    # Photometric / cos bar_i/2
    beta[1] = np.sin(self._bar_i) * np.sin(2 * self._delta)  # sinphi
    beta[2] = np.cos(self._bar_i) * np.cos(self._delta)**2  # sin2phi
    return (beta)

  def _phot(self):
    alpha = np.zeros(2)
    alpha[0] = np.sin(self._bar_i) * np.sin(self._delta)  # const
    alpha[1] = np.cos(self._bar_i) * np.cos(self._delta)  # cosphi
    return (alpha)

  def _apply_limb_dark(self, coefs, ld, is_alpha=True):
    degree_ld = len(ld)
    nfreq = coefs.shape[0]

    coefs_ld = np.zeros((degree_ld, nfreq + degree_ld - 1))
    coefs_ld[0, :nfreq] = coefs

    a = np.sin(self._bar_i) * np.sin(self._delta)
    b = np.cos(self._bar_i) * np.cos(self._delta)
    hb = b * 0.5

    for k in range(1, degree_ld):
      coefs_ld[k, :-1] = hb * coefs_ld[k - 1, 1:]
      coefs_ld[k, 1:] += hb * coefs_ld[k - 1, :-1]
      if is_alpha:
        coefs_ld[k, 1] += hb * coefs_ld[k - 1, 0]  # Add cos (- omega*t) term
      else:
        coefs_ld[k, 0] = 0
      coefs_ld[k] += a * coefs_ld[k - 1]

    return (np.sum(ld[:, np.newaxis] * coefs_ld, axis=0))

  def compute_Fourier(self,
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
    br_a2=None):
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

    self._bar_i = bar_i
    self._delta = delta
    ld = np.array([1 - ld_a1 - ld_a2, ld_a1, ld_a2])
    if self._brightening:
      br = np.array([br_center - br_a1 - br_a2, br_a1, br_a2])

    coswt0 = max(-np.tan(self._bar_i) * np.tan(self._delta), -1)
    assert coswt0 < 1

    wt0 = np.arccos(coswt0)

    self._Sct0[0] = wt0
    self._Sct0[1:] = np.sin(self._j * wt0) / self._j
    Sckpn = self._Sct0[self._kpn]
    Sckmn = self._Sct0[self._kmn]
    Scp = Sckpn + Sckmn
    Scm = Sckpn - Sckmn
    Scp[0] /= np.sqrt(2)

    # Rv effect
    alpha_rv_cb = self._apply_limb_dark(self._rv_cb(), ld)
    beta_rv_phot = self._apply_limb_dark(self._rv_phot(), ld, is_alpha=False)
    if self._brightening:
      beta_rv_phot = self._apply_limb_dark(beta_rv_phot, br, is_alpha=False)

    # Photometric effect
    alpha_phot = self._apply_limb_dark(self._phot(), ld)
    if self._brightening:
      alpha_phot = self._apply_limb_dark(alpha_phot, br)

    ab = np.zeros((2, self._nfreq, 2))
    ab[0, :, 0] = Scp[:, :alpha_rv_cb.shape[0]] @ alpha_rv_cb
    ab[1, :, 0] = Scm[:, 1:] @ beta_rv_phot[1:]
    ab[0, :, 1] = Scp[:, :-1] @ alpha_phot

    # Spot at opposite longitude
    ab[:, ::2] *= 1 + a180
    ab[:, 1::2] *= 1 - a180

    # Normalized coefs
    if self._normalized:
      sig_rv_cb /= np.sqrt(np.sum(ab[0, :, 0]**2))
      sig_rv_phot /= np.sqrt(np.sum(ab[1, :, 0]**2))
      sig_phot /= np.sqrt(np.sum(ab[0, :, 1]**2))

    ab[0, :, 0] *= sig_rv_cb
    ab[1, :, 0] *= sig_rv_phot
    ab[0, :, 1] *= sig_phot

    return (ab)

  def _translate_param(self, **kwargs):
    r"""
    Helper function to perform a change of variables.
    Intended to be used with :class:`spleaf.term.TransformKernel`.
    """

    param = {
      key: kwargs.pop(key)
      for key in list(kwargs) if key.startswith('decay_')
    }
    param['fourier_P'] = kwargs.pop('P')

    ab = self.compute_Fourier(**kwargs)
    param['fourier_alpha'] = ab[0, ..., np.newaxis]
    param['fourier_beta'] = ab[1, ..., np.newaxis]

    return (param)

  def kernel(self,
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
    br_a2=None):
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
    kwargs = dict(P=P,
      sig_rv_phot=sig_rv_phot,
      sig_rv_cb=sig_rv_cb,
      sig_phot=sig_phot,
      a180=a180,
      bar_i=bar_i,
      delta=delta,
      ld_a1=ld_a1,
      ld_a2=ld_a2)

    if self._brightening:
      kwargs.update(dict(br_center=br_center, br_a1=br_a1, br_a2=br_a2))

    kwargs.update({
      f'decay_{key}': decay_kernel._get_param(key)
      for key in decay_kernel._param
    })

    return (term.TransformKernel(
      term.SimpleProductKernel(decay=decay_kernel,
      fourier=term.MultiFourierKernel(1,
      np.zeros((self._nfreq, 2, 1)),
      np.zeros((self._nfreq, 2, 1)),
      series_index=series_index,
      vectorize=True)), self._translate_param, **kwargs))


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
  return (np.exp(ew))


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
  u = np.array([(delta - delta_mu) / delta_sig,
    (delta + delta_mu) / delta_sig])
  ew = -u * u / 2
  ew -= np.max(ew)
  return (np.sum(np.exp(ew), axis=0))


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

  def compute_Fourier(self,
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
    **kwargs):
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

    ab_delta = np.array([
      self._spot.compute_Fourier(sqwk, sqwk, sqwk, a180, bar_i, dk, ld_a1,
      ld_a2, br_center, br_a1, br_a2) for dk, sqwk in zip(delta, sqw)
    ])

    # ab_delta : ndelta x alpha/beta=2 x nfreq x nseries=2
    ab = np.zeros((2, self._nfreq, 2, 2))
    for kharm in range(self._nfreq):
      T = np.tensordot(ab_delta[:, :, kharm],
        ab_delta[:, :, kharm],
        axes=(0, 0))
      # T: alpha/beta x nseries x alpha/beta x nseries
      Tb = T[::-1, :, ::-1].copy()
      Tb[1] *= -1
      Tb[:, :, 1] *= -1
      Tf = (T + Tb).reshape(4, 4)
      eigval, eigvec = np.linalg.eigh(Tf)
      U = eigvec[:, ::2] * np.sqrt(np.maximum(0, eigval[::2]))
      ik = U[3] != 0
      Uik = U[:, ik]
      rho = np.sqrt(Uik[1]**2 + Uik[3]**2)
      U[:, ik] = np.array([(Uik[0] * Uik[1] + Uik[2] * Uik[3]) / rho, rho,
        (Uik[1] * Uik[2] - Uik[0] * Uik[3]) / rho, 0 * rho])
      U[:, U[1] < 0] *= -1
      rho = np.sqrt(np.sum(U[1]**2))
      if rho > 0:
        a = U[1, 0] / rho
        b = U[1, 1] / rho
        U[:, 0], U[:, 1] = a * U[:, 0] + b * U[:, 1], b * U[:, 0] - a * U[:, 1]
      ab[:, kharm] = U.reshape((2, 2, 2))

    ab[0, :, 0] *= sig_rv_cb / np.sqrt(np.sum(ab[0, :, 0]**2))
    ab[1, :, 0] *= sig_rv_phot / np.sqrt(np.sum(ab[1, :, 0]**2))
    ab[:, :, 1] *= sig_phot / np.sqrt(np.sum(ab[:, :, 1]**2))

    return (ab)

  def _translate_param(self, **kwargs):
    r"""
    Helper function to perform a change of variables.
    Intended to be used with :class:`spleaf.term.TransformKernel`.
    """

    param = {
      key: kwargs.pop(key)
      for key in list(kwargs) if key.startswith('decay_')
    }
    param['fourier_P'] = kwargs.pop('P')

    ab = self.compute_Fourier(**kwargs)
    param['fourier_alpha'] = ab[0]
    param['fourier_beta'] = ab[1]

    return (param)

  def kernel(self,
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
    **kwargs):
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
      dict(P=P,
      sig_rv_phot=sig_rv_phot,
      sig_rv_cb=sig_rv_cb,
      sig_phot=sig_phot,
      a180=a180,
      bar_i=bar_i,
      ld_a1=ld_a1,
      ld_a2=ld_a2))

    if self._brightening:
      kwargs.update(dict(br_center=br_center, br_a1=br_a1, br_a2=br_a2))

    kwargs.update({
      f'decay_{key}': decay_kernel._get_param(key)
      for key in decay_kernel._param
    })

    return (term.TransformKernel(
      term.SimpleProductKernel(decay=decay_kernel,
      fourier=term.MultiFourierKernel(1,
      np.zeros((self._nfreq, 2, 2)),
      np.zeros((self._nfreq, 2, 2)),
      series_index=series_index,
      vectorize=True)), self._translate_param, **kwargs))


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

  def __init__(self, nharm, ndelta, weight, brightening=False,
    normalized=True):
    self._nharm = nharm
    self._nfreq = nharm + 1
    self._ndelta = ndelta
    self._weight = weight
    self._brightening = brightening
    self._normalized = normalized
    self._spot = SpotsSameLat(nharm, brightening, normalized=False)
    self._u = (np.arange(ndelta) + 0.5) / (ndelta + 1)

  def compute_Fourier(self,
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
    **kwargs):
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
    delta_min = max(bar_i, 0) - np.pi / 2
    delta_max = min(bar_i, 0) + np.pi / 2
    delta = delta_min + self._u * (delta_max - delta_min)

    w = self._weight(delta, **kwargs)
    w /= np.sum(w)
    sqw = np.sqrt(w)

    ab_delta = np.array([
      self._spot.compute_Fourier(sqwk, sqwk, sqwk, a180, bar_i, dk, ld_a1,
      ld_a2, br_center, br_a1, br_a2) for dk, sqwk in zip(delta, sqw)
    ])
    # ab_delta : ndelta x alpha/beta=2 x nfreq x nseries=2

    mag_alpha = mag_ratio * np.sum(sqw[:, np.newaxis] * ab_delta[:, 0, 0],
      axis=0)

    per_ab = np.zeros((2, self._nfreq, 2, 2))
    for kharm in range(self._nfreq):
      T = np.tensordot(ab_delta[:, :, kharm],
        ab_delta[:, :, kharm],
        axes=(0, 0))
      # T: alpha/beta x nseries x alpha/beta x nseries
      Tb = T[::-1, :, ::-1].copy()
      Tb[1] *= -1
      Tb[:, :, 1] *= -1
      Tf = (T + Tb).reshape(4, 4)
      eigval, eigvec = np.linalg.eigh(Tf)
      U = eigvec[:, ::2] * np.sqrt(np.maximum(0, eigval[::2]))
      ik = U[3] != 0
      Uik = U[:, ik]
      rho = np.sqrt(Uik[1]**2 + Uik[3]**2)
      U[:, ik] = np.array([(Uik[0] * Uik[1] + Uik[2] * Uik[3]) / rho, rho,
        (Uik[1] * Uik[2] - Uik[0] * Uik[3]) / rho, 0 * rho])
      U[:, U[1] < 0] *= -1
      rho = np.sqrt(np.sum(U[1]**2))
      if rho > 0:
        a = U[1, 0] / rho
        b = U[1, 1] / rho
        U[:, 0], U[:, 1] = a * U[:, 0] + b * U[:, 1], b * U[:, 0] - a * U[:, 1]
      per_ab[:, kharm] = U.reshape((2, 2, 2))

    if self._normalized:
      sig_rv_cb /= np.sqrt(np.sum(per_ab[0, :, 0]**2) + mag_alpha[0]**2)
      sig_rv_phot /= np.sqrt(np.sum(per_ab[1, :, 0]**2))
      sig_phot /= np.sqrt(np.sum(per_ab[:, :, 1]**2) + mag_alpha[1]**2)
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
      f'per_{key}': kwargs.pop(key)
      for key in list(kwargs) if key.startswith('decay_')
    }

    param.update({
      key: kwargs.pop(key)
      for key in list(kwargs) if key.startswith('mag_decay_')
    })

    param['per_fourier_P'] = kwargs.pop('P')

    per_ab, mag_ab = self.compute_Fourier(**kwargs)
    param['per_fourier_alpha'] = per_ab[0]
    param['per_fourier_beta'] = per_ab[1]
    param['mag_fourier_alpha'] = mag_ab[0]

    return (param)

  def kernel(self,
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
    **kwargs):
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
    kernel : :class:`spleaf.term.Kernel`
      S+LEAF kernel corresponding to this FENRIR model.
    """
    kwargs.update(
      dict(P=P,
      sig_rv_phot=sig_rv_phot,
      sig_rv_cb=sig_rv_cb,
      sig_phot=sig_phot,
      mag_ratio=mag_ratio,
      a180=a180,
      bar_i=bar_i,
      ld_a1=ld_a1,
      ld_a2=ld_a2))

    if self._brightening:
      kwargs.update(dict(br_center=br_center, br_a1=br_a1, br_a2=br_a2))

    kwargs.update({
      f'decay_{key}': decay_kernel._get_param(key)
      for key in decay_kernel._param
    })

    kwargs.update({
      f'mag_decay_{key}': mag_decay_kernel._get_param(key)
      for key in mag_decay_kernel._param
    })

    return (term.TransformKernel(
      term.SimpleSumKernel(per=term.SimpleProductKernel(decay=decay_kernel,
      fourier=term.MultiFourierKernel(1,
      np.zeros((self._nfreq, 2, 2)),
      np.zeros((self._nfreq, 2, 2)),
      series_index=series_index,
      vectorize=True)),
      mag=term.SimpleProductKernel(decay=mag_decay_kernel,
      fourier=term.MultiFourierKernel(None,
      np.zeros((1, 2, 1)),
      None,
      series_index=series_index,
      vectorize=True))), self._translate_param, **kwargs))


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
    spi._ab = np.memmap(basename + '_mmap.dat',
      dtype=float,
      mode='w+',
      shape=gshape + (2, spi._nfreq, 2, 2))
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
    return (spi)

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
    spi._ab = np.memmap(basename + '_mmap.dat',
      dtype=float,
      mode='r',
      shape=gshape + (2, spi._nfreq, 2, 2))
    return (spi)

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
    self._params = dict(sig_rv_phot=sig_rv_phot,
      sig_rv_cb=sig_rv_cb,
      sig_phot=sig_phot,
      a180=a180)
    self._params.update(kwargs)
    self._inds = [
      max(
      1,
      min(self._grid[key].size - 1,
      np.searchsorted(self._grid[key], kwargs[key], 'right')))
      for key in self._grid
    ]
    self._w = np.ones(self._ndim * [2])
    for k, key in enumerate(self._grid):
      g = self._grid[key]
      i = self._inds[k]
      x = kwargs[key]
      self._w[k * (slice(None), ) + (0, )] *= g[i] - x
      self._w[k * (slice(None), ) + (1, )] *= x - g[i - 1]
    self._unscaled_ab = np.tensordot(self._ab[tuple(
      slice(i - 1, i + 1) for i in self._inds)],
      self._w,
      axes=(self._listdim, self._listdim))
    # Spot at opposite longitude
    self._unscaled_ab_180 = self._unscaled_ab.copy()
    self._unscaled_ab_180[:, ::2] *= 1 + a180
    self._unscaled_ab_180[:, 1::2] *= 1 - a180
    # Scaling
    self._scaled_ab = self._unscaled_ab_180.copy()
    self._scaled_ab[0, :,
      0] *= sig_rv_cb / np.sqrt(np.sum(self._scaled_ab[0, :, 0]**2))
    self._scaled_ab[1, :,
      0] *= sig_rv_phot / np.sqrt(np.sum(self._scaled_ab[1, :, 0]**2))
    self._scaled_ab[:, :,
      1] *= sig_phot / np.sqrt(np.sum(self._scaled_ab[:, :, 1]**2))
    return (self._scaled_ab)

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
    s = np.sqrt(np.sum(self._unscaled_ab_180[0, :, 0]**2))
    grad_s = -cgradc / s
    grad[0, :, 0] *= self._params['sig_rv_cb'] / s
    grad[0, :, 0] += self._unscaled_ab_180[0, :, 0] * grad_s / s

    # self._scaled_ab[1, :,
    #   0] *= sig_rv_phot / np.sqrt(np.sum(self._scaled_ab[1, :, 0]**2))
    cgradc = np.sum(self._scaled_ab[1, :, 0] * grad[1, :, 0])
    grad_params['sig_rv_phot'] = cgradc / self._params['sig_rv_phot']
    s = np.sqrt(np.sum(self._unscaled_ab_180[1, :, 0]**2))
    grad_s = -cgradc / s
    grad[1, :, 0] *= self._params['sig_rv_phot'] / s
    grad[1, :, 0] += self._unscaled_ab_180[1, :, 0] * grad_s / s

    # self._scaled_ab[:, :,
    #   1] *= sig_phot / np.sqrt(np.sum(self._scaled_ab[:, :, 1]**2))
    cgradc = np.sum(self._scaled_ab[:, :, 1] * grad[:, :, 1])
    grad_params['sig_phot'] = cgradc / self._params['sig_phot']
    s = np.sqrt(np.sum(self._unscaled_ab_180[:, :, 1]**2))
    grad_s = -cgradc / s
    grad[:, :, 1] *= self._params['sig_phot'] / s
    grad[:, :, 1] += self._unscaled_ab_180[:, :, 1] * grad_s / s

    # Spot at opposite longitude
    # self._unscaled_ab_180 = self._unscaled_ab.copy()
    # self._unscaled_ab_180[:, ::2] *= 1 + a180
    # self._unscaled_ab_180[:, 1::2] *= 1 - a180
    grad_params['a180'] = np.sum(self._unscaled_ab[:, ::2] *
      grad[:, ::2]) - np.sum(self._unscaled_ab[:, 1::2] * grad[:, 1::2])
    grad[:, ::2] *= 1 + self._params['a180']
    grad[:, 1::2] *= 1 - self._params['a180']

    # self._unscaled_ab = np.tensordot(self._ab[tuple(
    #   slice(i - 1, i + 1) for i in self._inds)],
    #   self._w,
    #   axes=(self._listdim, self._listdim))
    grad_w = np.tensordot(grad,
      self._ab[tuple(slice(i - 1, i + 1) for i in self._inds)],
      axes=([0, 1, 2, 3], [-4, -3, -2, -1]))
    # self._w = np.ones(self._ndim * [2])
    # for k, key in enumerate(self._grid):
    #   g = self._grid[key]
    #   i = self._inds[k]
    #   x = kwargs[key]
    #   self._w[k * (slice(None), ) + (0, )] *= g[i] - x
    #   self._w[k * (slice(None), ) + (1, )] *= x - g[i - 1]
    for k, key in enumerate(self._grid):
      dw = np.ones(self._ndim * [2])
      dw[k * (slice(None), ) + (0, )] = -1
      for l, ley in enumerate(self._grid):
        if l == k:
          continue
        g = self._grid[ley]
        i = self._inds[l]
        x = self._params[ley]
        dw[l * (slice(None), ) + (0, )] *= g[i] - x
        dw[l * (slice(None), ) + (1, )] *= x - g[i - 1]
      grad_params[key] = np.sum(grad_w * dw)

    return (grad_params)

  def _translate_param(self, **kwargs):
    r"""
    Helper function to perform a change of variables.
    Intended to be used with :class:`spleaf.term.TransformKernel`.
    """
    param = {
      key: kwargs.pop(key)
      for key in list(kwargs) if key.startswith('decay_')
    }
    param['fourier_P'] = kwargs.pop('P')

    ab = self.compute_Fourier(**kwargs)
    param['fourier_alpha'] = ab[0]
    param['fourier_beta'] = ab[1]

    return (param)

  def _translate_param_back(self, grad):
    r"""
    Backward propagation of the gradient for :func:`_translate_param`.
    """
    grad_param = grad.copy()
    grad_param['P'] = grad_param.pop('fourier_P')
    grad_ab = np.array(
      [grad_param.pop('fourier_alpha'),
      grad_param.pop('fourier_beta')])
    grad_param.update(self.compute_Fourier_back(grad_ab))

    return (grad_param)

  def kernel(self, series_index, decay_kernel, P, sig_rv_phot, sig_rv_cb,
    sig_phot, a180, **kwargs):
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
      dict(P=P,
      sig_rv_phot=sig_rv_phot,
      sig_rv_cb=sig_rv_cb,
      sig_phot=sig_phot,
      a180=a180))

    kwargs.update({
      f'decay_{key}': decay_kernel._get_param(key)
      for key in decay_kernel._param
    })

    return (term.TransformKernel(
      term.SimpleProductKernel(decay=decay_kernel,
      fourier=term.MultiFourierKernel(1,
      np.zeros((self._nfreq, 2, 2)),
      np.zeros((self._nfreq, 2, 2)),
      series_index=series_index,
      vectorize=True)), self._translate_param, self._translate_param_back,
      **kwargs))


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
    spi._per_ab = np.memmap(basename + '_per_mmap.dat',
      dtype=float,
      mode='w+',
      shape=gshape + (2, spi._nfreq, 2, 2))
    spi._mag_alpha = np.memmap(basename + '_mag_mmap.dat',
      dtype=float,
      mode='w+',
      shape=gshape + (2, ))
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
    return (spi)

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
    spi._per_ab = np.memmap(basename + '_per_mmap.dat',
      dtype=float,
      mode='r',
      shape=gshape + (2, spi._nfreq, 2, 2))
    spi._mag_alpha = np.memmap(basename + '_mag_mmap.dat',
      dtype=float,
      mode='r',
      shape=gshape + (2, ))
    return (spi)

  def compute_Fourier(
      self, sig_rv_phot, sig_rv_cb, sig_phot, mag_ratio, a180, **kwargs):
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
    self._params = dict(sig_rv_phot=sig_rv_phot,
      sig_rv_cb=sig_rv_cb,
      sig_phot=sig_phot,
      mag_ratio=mag_ratio,
      a180=a180)
    self._params.update(kwargs)
    self._inds = [
      max(
      1,
      min(self._grid[key].size - 1,
      np.searchsorted(self._grid[key], kwargs[key], 'right')))
      for key in self._grid
    ]
    self._w = np.ones(self._ndim * [2])
    for k, key in enumerate(self._grid):
      g = self._grid[key]
      i = self._inds[k]
      x = kwargs[key]
      self._w[k * (slice(None), ) + (0, )] *= g[i] - x
      self._w[k * (slice(None), ) + (1, )] *= x - g[i - 1]
    self._unscaled_per_ab = np.tensordot(self._per_ab[tuple(
      slice(i - 1, i + 1) for i in self._inds)],
      self._w,
      axes=(self._listdim, self._listdim))
    self._unscaled_mag_alpha = np.tensordot(self._mag_alpha[tuple(
      slice(i - 1, i + 1) for i in self._inds)],
      self._w,
      axes=(self._listdim, self._listdim))
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
      np.sum(self._scaled_per_ab[0, :, 0]**2) + self._scaled_mag_alpha[0]**2)
    self._scaled_per_ab[0, :, 0] *= scale_cb
    self._scaled_mag_alpha[0] *= scale_cb

    self._scaled_per_ab[1, :,
      0] *= sig_rv_phot / np.sqrt(np.sum(self._scaled_per_ab[1, :, 0]**2))

    scale_phot = sig_phot / np.sqrt(
      np.sum(self._scaled_per_ab[:, :, 1]**2) + self._scaled_mag_alpha[1]**2)
    self._scaled_per_ab[:, :, 1] *= scale_phot
    self._scaled_mag_alpha[1] *= scale_phot

    return (self._scaled_per_ab, np.array([[self._scaled_mag_alpha],
      [2 * [0]]])[..., np.newaxis])

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
    cgradc = np.sum(self._scaled_per_ab[0, :, 0] *
      grad_per[0, :, 0]) + self._scaled_mag_alpha[0] * grad_mag[0]
    grad_params['sig_rv_cb'] = cgradc / self._params['sig_rv_cb']
    s = np.sqrt(
      np.sum(self._unscaled_per_ab_180[0, :, 0]**2) +
      self._unscaled_mag_alpha_180[0]**2)
    grad_s = -cgradc / s
    grad_per[0, :, 0] *= self._params['sig_rv_cb'] / s
    grad_per[0, :, 0] += self._unscaled_per_ab_180[0, :, 0] * grad_s / s
    grad_mag[0] *= self._params['sig_rv_cb'] / s
    grad_mag[0] += self._unscaled_mag_alpha_180[0] * grad_s / s

    # self._scaled_per_ab[1, :,
    #   0] *= sig_rv_phot / np.sqrt(np.sum(self._scaled_per_ab[1, :, 0]**2))
    cgradc = np.sum(self._scaled_per_ab[1, :, 0] * grad_per[1, :, 0])
    grad_params['sig_rv_phot'] = cgradc / self._params['sig_rv_phot']
    s = np.sqrt(np.sum(self._unscaled_per_ab_180[1, :, 0]**2))
    grad_s = -cgradc / s
    grad_per[1, :, 0] *= self._params['sig_rv_phot'] / s
    grad_per[1, :, 0] += self._unscaled_per_ab_180[1, :, 0] * grad_s / s

    # scale_phot = sig_phot / np.sqrt(
    #   np.sum(self._scaled_per_ab[:,:,1]**2) + self._scaled_mag_alpha[1]**2)
    # self._scaled_per_ab[:,:,1] *= scale_phot
    # self._scaled_mag_alpha[1] *= scale_phot
    cgradc = np.sum(self._scaled_per_ab[:, :, 1] *
      grad_per[:, :, 1]) + self._scaled_mag_alpha[1] * grad_mag[1]
    grad_params['sig_phot'] = cgradc / self._params['sig_phot']
    s = np.sqrt(
      np.sum(self._unscaled_per_ab_180[:, :, 1]**2) +
      self._unscaled_mag_alpha_180[1]**2)
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
    grad_params['a180'] = np.sum(self._unscaled_per_ab[:, ::2] *
      grad_per[:, ::2]) + self._params['mag_ratio'] * smag - np.sum(
      self._unscaled_per_ab[:, 1::2] * grad_per[:, 1::2])
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
    grad_w = np.tensordot(grad_per,
      self._per_ab[tuple(slice(i - 1, i + 1) for i in self._inds)],
      axes=([0, 1, 2, 3], [-4, -3, -2, -1])) + np.tensordot(grad_mag,
      self._mag_alpha[tuple(slice(i - 1, i + 1) for i in self._inds)],
      axes=(0, -1))

    # self._w = np.ones(self._ndim * [2])
    # for k, key in enumerate(self._grid):
    #   g = self._grid[key]
    #   i = self._inds[k]
    #   x = kwargs[key]
    #   self._w[k * (slice(None), ) + (0, )] *= g[i] - x
    #   self._w[k * (slice(None), ) + (1, )] *= x - g[i - 1]
    for k, key in enumerate(self._grid):
      dw = np.ones(self._ndim * [2])
      dw[k * (slice(None), ) + (0, )] = -1
      for l, ley in enumerate(self._grid):
        if l == k:
          continue
        g = self._grid[ley]
        i = self._inds[l]
        x = self._params[ley]
        dw[l * (slice(None), ) + (0, )] *= g[i] - x
        dw[l * (slice(None), ) + (1, )] *= x - g[i - 1]
      grad_params[key] = np.sum(grad_w * dw)

    return (grad_params)

  def _translate_param(self, **kwargs):
    r"""
    Helper function to perform a change of variables.
    Intended to be used with :class:`spleaf.term.TransformKernel`.
    """
    param = {
      f'per_{key}': kwargs.pop(key)
      for key in list(kwargs) if key.startswith('decay_')
    }

    param.update({
      key: kwargs.pop(key)
      for key in list(kwargs) if key.startswith('mag_decay_')
    })

    param['per_fourier_P'] = kwargs.pop('P')

    per_ab, mag_ab = self.compute_Fourier(**kwargs)
    param['per_fourier_alpha'] = per_ab[0]
    param['per_fourier_beta'] = per_ab[1]
    param['mag_fourier_alpha'] = mag_ab[0]

    return (param)

  def _translate_param_back(self, grad):
    r"""
    Backward propagation of the gradient for :func:`_translate_param`.
    """
    grad_param = grad.copy()
    grad_param['P'] = grad_param.pop('per_fourier_P')
    grad_param.update({
      key[4:]: grad_param.pop(key)
      for key in list(grad_param) if key.startswith('per_decay_')
    })
    grad_per = np.array([
      grad_param.pop('per_fourier_alpha'),
      grad_param.pop('per_fourier_beta')
    ])
    grad_mag = np.array([grad_param.pop('mag_fourier_alpha'), [2 * [[0]]]])
    grad_param.update(self.compute_Fourier_back(grad_per, grad_mag))

    return (grad_param)

  def kernel(self, series_index, decay_kernel, mag_decay_kernel, P,
    sig_rv_phot, sig_rv_cb, sig_phot, mag_ratio, a180, **kwargs):
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
    delta : float
      Latitude of the spots/faculae.
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
      dict(P=P,
      sig_rv_phot=sig_rv_phot,
      sig_rv_cb=sig_rv_cb,
      sig_phot=sig_phot,
      mag_ratio=mag_ratio,
      a180=a180))

    kwargs.update({
      f'decay_{key}': decay_kernel._get_param(key)
      for key in decay_kernel._param
    })

    kwargs.update({
      f'mag_decay_{key}': mag_decay_kernel._get_param(key)
      for key in mag_decay_kernel._param
    })

    return (term.TransformKernel(
      term.SimpleSumKernel(per=term.SimpleProductKernel(decay=decay_kernel,
      fourier=term.MultiFourierKernel(1,
      np.zeros((self._nfreq, 2, 2)),
      np.zeros((self._nfreq, 2, 2)),
      series_index=series_index,
      vectorize=True)),
      mag=term.SimpleProductKernel(decay=mag_decay_kernel,
      fourier=term.MultiFourierKernel(None,
      np.zeros((1, 2, 1)),
      None,
      series_index=series_index,
      vectorize=True))), self._translate_param, self._translate_param_back,
      **kwargs))
