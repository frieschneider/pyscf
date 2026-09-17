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

import unittest
import numpy
from pyscf import gto, scf, mp, cc
from pyscf.scf import rohf_dods


def setUpModule():
    global mol
    mol = gto.M(
        verbose = 0,
        atom = 'O 0 0 0; O 0 0 1.2',
        basis = 'sto-3g',
        spin = 2,
    )

def tearDownModule():
    global mol
    del mol


class KnownValues(unittest.TestCase):
    def test_scf_matches_plain_rohf(self):
        # Form I/II must not affect the SCF iteration itself. mo_coeff
        # itself is not a safe invariant to compare here: O2 has a
        # degenerate open shell, so independent SCF runs -- even two plain
        # scf.ROHF(mol) calls -- can converge to different but physically
        # equivalent rotations within that degenerate subspace. The energy
        # and density matrix are the rotation-invariant quantities.
        #
        # Uses .kernel() directly (not .run()), since .run() now returns
        # the UHF-like object -- .kernel() is left returning the plain ROHF
        # result so this ROHF-shaped comparison still works.
        mf_ref = scf.ROHF(mol).run(conv_tol=1e-11)

        mf = rohf_dods.ROHFDoDS(mol)
        mf.conv_tol = 1e-11
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, mf_ref.e_tot, 9)
        dm = numpy.asarray(mf.make_rdm1())
        dm_ref = numpy.asarray(mf_ref.make_rdm1())
        self.assertTrue(numpy.allclose(dm, dm_ref, atol=1e-8))

    def test_run_returns_uhf_dods_directly(self):
        # .run() should hand back the UHF-like object directly, with no
        # separate .to_uhf_dods() call needed, and keep the converged
        # ROHF-shaped object reachable via ._rohf.
        mf1 = rohf_dods.ROHFDoDS(mol).run(conv_tol=1e-11)

        self.assertTrue(isinstance(mf1, scf.uhf.UHF))
        self.assertTrue(hasattr(mf1, '_rohf'))
        # ROHF-specific methods (needing a 1D mo_occ) still work via the backref.
        ss, mult = mf1._rohf.spin_square()
        self.assertAlmostEqual(mult, mol.spin + 1, 9)

        neleca, nelecb = mol.nelec
        self.assertAlmostEqual(mf1.mo_occ[0].sum(), neleca, 9)
        self.assertAlmostEqual(mf1.mo_occ[1].sum(), nelecb, 9)

        # The plumbing must run to completion with the real Form I/II
        # coefficients; this does not check that the resulting energies are
        # physically meaningful (see Tier 3 in the module docstring / PR
        # discussion for that).
        mp.MP2(mf1).kernel()
        cc.CCSD(mf1).kernel()

    def test_canonicalize_by_shell_matches_hf_canonicalize(self):
        # With a single Fock shared by all three blocks (e.g. pure focka:
        # Acc=Aoo=Avv=1, Bcc=Boo=Bvv=0), _canonicalize_by_shell must match
        # hf.canonicalize(mf, mo_coeff, mo_occ, fock=focka) exactly, since
        # both diagonalize the same closed/open/virtual blocks separately.
        from pyscf.scf import hf
        mf = scf.ROHF(mol).run(conv_tol=1e-11)
        fock = mf.get_fock()
        focka = fock.focka

        ref_e, ref_mo = hf.canonicalize(mf, mf.mo_coeff, mf.mo_occ, fock=focka)
        got_e, got_mo = rohf_dods._canonicalize_by_shell(
            mf.mo_coeff, mf.mo_occ, focka, focka, focka)

        self.assertTrue(numpy.allclose(ref_e, got_e, atol=1e-12))
        self.assertTrue(numpy.allclose(ref_mo, got_mo, atol=1e-12))

    # ---- Tier 2: invariants that must hold for ANY valid coupling
    # coefficients, since semicanonicalization only rotates orbitals within
    # the closed/open/virtual blocks -- it must never change the underlying
    # ROHF density or total energy. Failing here points at a bug in the
    # diagonalization/projector machinery, independent of whether the
    # Form I/II *coefficients* themselves are physically correct (Tier 3).

    def test_dods_orbitals_are_orthonormal(self):
        mf1 = rohf_dods.ROHFDoDS(mol).run(conv_tol=1e-11)
        s = mf1._rohf.get_ovlp()
        for c in mf1.mo_coeff:
            self.assertTrue(numpy.allclose(
                c.conj().T.dot(s).dot(c), numpy.eye(c.shape[1]), atol=1e-8))

    def test_dods_density_matches_rohf(self):
        mf1 = rohf_dods.ROHFDoDS(mol).run(conv_tol=1e-11)

        dma_ref, dmb_ref = mf1._rohf.make_rdm1()
        dma_new, dmb_new = mf1.make_rdm1()
        self.assertTrue(numpy.allclose(dma_ref, dma_new, atol=1e-8))
        self.assertTrue(numpy.allclose(dmb_ref, dmb_new, atol=1e-8))

    def test_dods_energy_recomputed_matches_rohf(self):
        mf1 = rohf_dods.ROHFDoDS(mol).run(conv_tol=1e-11)

        # Recompute the energy from mf1's own orbitals/density from
        # scratch -- don't just trust the e_tot that to_uhf_dods() copied
        # over from the ROHF object.
        dm_new = mf1.make_rdm1()
        e_recomputed = mf1.energy_tot(dm=dm_new)
        self.assertAlmostEqual(e_recomputed, mf1._rohf.e_tot, 8)


if __name__ == '__main__':
    print('Full tests for ROHF-DODS')
    unittest.main()
