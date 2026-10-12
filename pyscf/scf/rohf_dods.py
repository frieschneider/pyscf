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
from pyscf.scf import uhf


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


def _form1_coeffs_canonical(nopen):
    '''Form I coupling coefficients (Koopmans-consistent for ionization out
    of closed/open orbitals), "canonical" variant, used to build the
    alpha-channel orbitals in :func:`semi_canonicalize_dods`.
    '''
    S = nopen / 2
    return RohfCouplingCoeffs(
        Acc=(2 * S + 1)/(2 * S),
        Aoo=1.0,
        Avv=1.0,
        Bcc=-1/(2 * S),
        Boo=0.0,
        Bvv=0.0)


def _form2_coeffs_canonical(nopen):
    '''Form II coupling coefficients (Koopmans-consistent for electron
    attachment into open/virtual orbitals), "canonical" variant, used to
    build the beta-channel orbitals in :func:`semi_canonicalize_dods`.
    '''
    S = nopen / 2
    return RohfCouplingCoeffs(
        Acc=0.0,
        Aoo=0.0,
        Avv=-1/(2 * S),
        Bcc=1.0,
        Boo=1.0,
        Bvv=(2 * S + 1)/(2 * S))


def _form1_coeffs_semicanonical(nopen):
    '''Form I coupling coefficients, "semicanonical" variant.
    '''
    return RohfCouplingCoeffs(
        Acc=1.0,
        Aoo=1.0,
        Avv=1.0,
        Bcc=0.0,
        Boo=0.0,
        Bvv=0.0)


def _form2_coeffs_semicanonical(nopen):
    '''Form II coupling coefficients, "semicanonical" variant.

    '''
    return RohfCouplingCoeffs(
        Acc=0.0,
        Aoo=0.0,
        Avv=0.0,
        Bcc=1.0,
        Boo=1.0,
        Bvv=1.0)


# (form, variant) -> coefficient function
_COEFF_FUNCS = {
    ('I', 'canonical'): _form1_coeffs_canonical,
    ('II', 'canonical'): _form2_coeffs_canonical,
    ('I', 'semicanonical'): _form1_coeffs_semicanonical,
    ('II', 'semicanonical'): _form2_coeffs_semicanonical,
}


def get_coupling_coeffs(mf, form, variant):
    '''Return the Form I ('I') or Form II ('II') coupling coefficients, for
    the given variant ('canonical' or 'semicanonical'), for mf's open-shell
    structure. Cached on mf so repeated calls (e.g. from both alpha- and
    beta-channel construction) don't recompute.
    '''
    if form not in ('I', 'II'):
        raise ValueError(f"form must be 'I' or 'II', got {form!r}")
    if variant not in ('canonical', 'semicanonical'):
        raise ValueError(
            f"variant must be 'canonical' or 'semicanonical', got {variant!r}")
    key = (form, variant)
    cache = mf._coupling_coeffs
    if key not in cache:
        nopen = mf.nelec[0] - mf.nelec[1]
        if nopen <= 0:
            raise ValueError(
                'ROHF-DODS requires nelec[0] > nelec[1] (S > 0); '
                f'got nelec={mf.nelec}.')
        cache[key] = _COEFF_FUNCS[key](nopen)
    return cache[key]


def _shell_focks(focka, fockb, coeffs):
    '''Build the closed/open/virtual "self" Fock matrices from a
    RohfCouplingCoeffs.
    '''
    fock_cc = coeffs.Acc * focka + coeffs.Bcc * fockb
    fock_oo = coeffs.Aoo * focka + coeffs.Boo * fockb
    fock_vv = coeffs.Avv * focka + coeffs.Bvv * fockb
    return fock_cc, fock_oo, fock_vv


def build_spin_focks(mf, variant):
    '''Build the closed/open/virtual Fock matrices used for the alpha
    (Form I) and beta (Form II) semicanonicalizations, for the given
    variant ('canonical' or 'semicanonical').
    '''
    fock = mf.get_fock()
    focka, fockb = fock.focka, fock.fockb
    coeffs_I = get_coupling_coeffs(mf, 'I', variant)
    coeffs_II = get_coupling_coeffs(mf, 'II', variant)
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


def _canonicalize_merged(mo_coeff, mo_occ, occ_mask, vir_mask, fock):
    '''Diagonalize one Fock within occ_mask and vir_mask separately (2-way
    split -- occ_mask may merge more than one of closed/open/virtual),
    without changing occupancy.
    '''
    mo = numpy.empty_like(mo_coeff)
    mo_e = numpy.empty(mo_occ.size)
    for idx in (occ_mask, vir_mask):
        if numpy.count_nonzero(idx) > 0:
            orb = mo_coeff[:, idx]
            f1 = reduce(numpy.dot, (orb.conj().T, fock, orb))
            e, c = scipy.linalg.eigh(f1)
            mo[:, idx] = numpy.dot(orb, c)
            mo_e[idx] = e
    mo = hf._adjust_phase_(mo)
    return mo_e, mo


def semi_canonicalize_dods(mf, variant):
    '''Build the alpha- and beta-channel canonical orbitals, for the given
    variant ('canonical' or 'semicanonical').

    'canonical': closed, open, and virtual diagonalized as three separate
    blocks (Form-specific mixture of focka/fockb per block).

    'semicanonical': alpha diagonalized as (closed+open merged) / virtual
    -- the standard alpha-occ/alpha-vir split; beta as closed /
    (open+virtual merged) -- the standard beta-occ/beta-vir split. This
    requires the semicanonical coefficients to give the same Fock for all
    three blocks within each channel (pure focka for alpha, pure fockb for
    beta), since a single matrix is diagonalized over each merged block.

    Returns:
        (mo_coeff_a, mo_coeff_b, mo_energy_a, mo_energy_b)
    '''
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    (focka_cc, focka_oo, focka_vv), (fockb_cc, fockb_oo, fockb_vv) = \
        build_spin_focks(mf, variant)

    if variant == 'semicanonical':
        coreidx = mo_occ == 2
        openidx = mo_occ == 1
        viridx = mo_occ == 0
        mo_energy_a, mo_coeff_a = _canonicalize_merged(
            mo_coeff, mo_occ, coreidx | openidx, viridx, focka_cc)
        mo_energy_b, mo_coeff_b = _canonicalize_merged(
            mo_coeff, mo_occ, coreidx, openidx | viridx, fockb_cc)
    else:
        mo_energy_a, mo_coeff_a = _canonicalize_by_shell(
            mo_coeff, mo_occ, focka_cc, focka_oo, focka_vv)
        mo_energy_b, mo_coeff_b = _canonicalize_by_shell(
            mo_coeff, mo_occ, fockb_cc, fockb_oo, fockb_vv)
    return mo_coeff_a, mo_coeff_b, mo_energy_a, mo_energy_b


def to_uhf_dods(mf, variant='canonical'):
    '''Build a genuine UHF instance with distinct alpha/beta mo_coeff:
    alpha canonicalized with Form I, beta canonicalized with Form II, using
    the given variant ('canonical' or 'semicanonical') of coefficients.
    '''
    mo_coeff_a, mo_coeff_b, mo_energy_a, mo_energy_b = \
        semi_canonicalize_dods(mf, variant)
    mf1 = addons.convert_to_uhf(mf)
    mf1.__class__ = uhf.UHF  # strip ROHFDoDS methods kept by convert_to_uhf's class merge
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

    def run(self, *args, variant='canonical', **kwargs):
        rohf.ROHF.run(self, *args, **kwargs)
        mf1 = self.to_uhf_dods(variant=variant)
        return mf1
