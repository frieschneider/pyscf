#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
ROHF-DODS: "different orbitals for different spins" hand-off for post-HF
methods, built from a converged ROHF wave function.

The ROHF SCF iteration is left completely untouched -- different valid
choices of coupling coefficients in the Roothaan effective Fock converge to
the same density and energy. Form I and Form II (Plakhutin-style canonical
coupling coefficients, see J. Chem. Phys. 125, 204110 (2006)) only enter
once, after convergence, to build the alpha and beta canonical orbitals:

    alpha orbitals  from Form I coefficients (Koopmans-consistent for
                    ionization out of closed/open orbitals)
    beta orbitals   from Form II coefficients (Koopmans-consistent for
                    electron attachment into open/virtual orbitals)

For each channel, the closed, open, and virtual blocks are diagonalized
separately -- exactly as :func:`rohf.canonicalize` already does for plain
ROHF -- except each block uses its own Form-specific mixture of the true
alpha/beta Fock matrices (focka, fockb). Blocks are never merged, so no
cross-block (closed-open, open-virtual, closed-virtual) matrix element is
ever needed. The result is packaged as a plain :class:`pyscf.scf.uhf.UHF`
object, ready to be handed to post-HF methods (MP2, CCSD, ...) the same way
`mf.to_uhf()` already does for plain ROHF.
'''

from functools import reduce
import collections
import numpy
import scipy.linalg
from pyscf.scf import addons
from pyscf.scf import hf
from pyscf.scf import rohf


# Coupling coefficients for the closed(c)/open(o)/virtual(v) "self" Fock
# matrices, read as
#     fock_cc = Acc * focka + Bcc * fockb
#     fock_oo = Aoo * focka + Boo * fockb
#     fock_vv = Avv * focka + Bvv * fockb
# (each pair need not sum to 1). Each of the three blocks is diagonalized
# independently (see _canonicalize_by_shell) -- there is no off-diagonal
# coupling block, since closed/open/virtual are never merged into a single
# diagonalization the way rohf.get_roothaan_fock's single matrix does.
RohfCouplingCoeffs = collections.namedtuple(
    'RohfCouplingCoeffs', ['Acc', 'Aoo', 'Avv', 'Bcc', 'Boo', 'Bvv'])


def _form1_coeffs(nopen):
    '''Form I coupling coefficients (Koopmans-consistent for ionization out
    of closed/open orbitals), used to build the alpha-channel orbitals in
    :func:`semi_canonicalize_dods`.
    '''
    S = nopen / 2
    return RohfCouplingCoeffs(
        Acc=(2 * S + 1)/(2 * S),
        Aoo=1.0,
        Avv=1.0,
        Bcc=-1/(2 * S),
        Boo=0.0,
        Bvv=0.0)


def _form2_coeffs(nopen):
    '''Form II coupling coefficients (Koopmans-consistent for electron
    attachment into open/virtual orbitals), used to build the beta-channel
    orbitals in :func:`semi_canonicalize_dods`.
    '''
    S = nopen / 2
    return RohfCouplingCoeffs(
        Acc=0.0,
        Aoo=0.0,
        Avv=-1/(2 * S),
        Bcc=1.0,
        Boo=1.0,
        Bvv=(2 * S + 1)/(2 * S))


def get_coupling_coeffs(mf, form):
    '''Return the Form I ('I') or Form II ('II') coupling coefficients for
    mf's open-shell structure. Cached on mf so repeated calls (e.g. from
    both alpha- and beta-channel construction) don't recompute.
    '''
    if form not in ('I', 'II'):
        raise ValueError(f"form must be 'I' or 'II', got {form!r}")
    cache = mf._coupling_coeffs
    if form not in cache:
        nopen = mf.nelec[0] - mf.nelec[1]
        if nopen == 0:
            raise ValueError(
                'ROHF-DODS requires an open-shell reference (S > 0); '
                f'got a closed-shell system (nelec={mf.nelec}).')
        if form == 'I':
            cache[form] = _form1_coeffs(nopen)
        else:
            cache[form] = _form2_coeffs(nopen)
    return cache[form]


def _shell_focks(focka, fockb, coeffs):
    '''Build the closed/open/virtual "self" Fock matrices from a
    RohfCouplingCoeffs.
    '''
    fock_cc = coeffs.Acc * focka + coeffs.Bcc * fockb
    fock_oo = coeffs.Aoo * focka + coeffs.Boo * fockb
    fock_vv = coeffs.Avv * focka + coeffs.Bvv * fockb
    return fock_cc, fock_oo, fock_vv


def build_spin_focks(mf):
    '''Build the closed/open/virtual Fock matrices used for the alpha
    (Form I) and beta (Form II) semicanonicalizations.

    Falls back to plain (focka, focka, focka) / (fockb, fockb, fockb) as
    long as the Form I/II coefficients are not implemented yet, so the
    rest of the pipeline (semicanonicalization, UHF packaging) stays
    runnable and testable end-to-end in the meantime.
    '''
    fock = mf.get_fock()
    focka, fockb = fock.focka, fock.fockb

    try:
        coeffs_I = get_coupling_coeffs(mf, 'I')
        coeffs_II = get_coupling_coeffs(mf, 'II')
    except NotImplementedError:
        return (focka, focka, focka), (fockb, fockb, fockb)

    return (_shell_focks(focka, fockb, coeffs_I),
            _shell_focks(focka, fockb, coeffs_II))


def _canonicalize_by_shell(mo_coeff, mo_occ, fock_cc, fock_oo, fock_vv):
    '''Diagonalize a (possibly different) effective Fock within each of the
    closed, open, and virtual blocks separately, without changing
    occupancy. Mirrors :func:`hf.canonicalize`, but takes one Fock matrix
    per block instead of a single shared one.
    '''
    coreidx = mo_occ == 2
    openidx = mo_occ == 1
    viridx = mo_occ == 0
    mo = numpy.empty_like(mo_coeff)
    mo_e = numpy.empty(mo_occ.size)
    for idx, fock in ((coreidx, fock_cc), (openidx, fock_oo), (viridx, fock_vv)):
        if numpy.count_nonzero(idx) > 0:
            orb = mo_coeff[:, idx]
            f1 = reduce(numpy.dot, (orb.conj().T, fock, orb))
            e, c = scipy.linalg.eigh(f1)
            mo[:, idx] = numpy.dot(orb, c)
            mo_e[idx] = e
    mo = hf._adjust_phase_(mo)
    return mo_e, mo


def semi_canonicalize_dods(mf):
    '''Diagonalize the closed/open/virtual blocks separately for the alpha
    channel (Form I coefficients) and the beta channel (Form II
    coefficients).

    Returns:
        (mo_coeff_a, mo_coeff_b, mo_energy_a, mo_energy_b)
    '''
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    (focka_cc, focka_oo, focka_vv), (fockb_cc, fockb_oo, fockb_vv) = build_spin_focks(mf)

    mo_energy_a, mo_coeff_a = _canonicalize_by_shell(
        mo_coeff, mo_occ, focka_cc, focka_oo, focka_vv)
    mo_energy_b, mo_coeff_b = _canonicalize_by_shell(
        mo_coeff, mo_occ, fockb_cc, fockb_oo, fockb_vv)
    return mo_coeff_a, mo_coeff_b, mo_energy_a, mo_energy_b


def to_uhf_dods(mf):
    '''Build a genuine UHF instance with distinct alpha/beta mo_coeff:
    alpha canonicalized with Form I, beta canonicalized with Form II.
    '''
    mo_coeff_a, mo_coeff_b, mo_energy_a, mo_energy_b = semi_canonicalize_dods(mf)
    mf1 = addons.convert_to_uhf(mf)
    mf1.mo_coeff = (mo_coeff_a, mo_coeff_b)
    mf1.mo_energy = (mo_energy_a, mo_energy_b)
    mf1.mo_occ = numpy.array((mf.mo_occ > 0, mf.mo_occ == 2), dtype=numpy.double)
    mf1.converged = mf.converged
    mf1.e_tot = mf.e_tot
    mf1._rohf = mf
    return mf1


class ROHFDoDS(rohf.ROHF):
    '''ROHF with a DODS (different orbitals for different spins) hand-off
    for post-HF methods, using Plakhutin-style Form I (alpha) / Form II
    (beta) canonical coupling coefficients.
    See Plakhutin, Davidson J. Chem. Phys. 140, 014102 (2014).

    The SCF iteration is identical to plain ROHF -- Form I/II only enter in
    :meth:`to_uhf_dods`. :meth:`run` calls it automatically and returns the
    UHF-like object directly; the converged ROHF-shaped object (needed for
    ROHF-specific methods like `spin_square()` or `nuc_grad_method()`, which
    assume a 1D `mo_occ`) stays reachable via the returned object's `._rohf`
    attribute. `.kernel()` is left returning the plain ROHF energy, since
    generic pyscf machinery (e.g. `newton_ah`, `as_scanner`) calls `.kernel()`
    on SCF objects assuming that return type.
    '''

    def __init__(self, mol):
        rohf.ROHF.__init__(self, mol)
        self._coupling_coeffs = {}

    get_coupling_coeffs = get_coupling_coeffs
    build_spin_focks = build_spin_focks
    semi_canonicalize_dods = semi_canonicalize_dods
    to_uhf_dods = to_uhf_dods

    def run(self, *args, **kwargs):
        rohf.ROHF.run(self, *args, **kwargs)
        mf1 = self.to_uhf_dods()
        return mf1
