/**
 * GauXC Copyright (c) 2020-2024, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy).
 *
 * (c) 2024-2025, Microsoft Corporation
 *
 * All rights reserved.
 *
 * See LICENSE.txt for details
 */
#include "ut_common.hpp"
#include "catch2/catch.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <gauxc/external/cube.hpp>
#include <gauxc/orbital_evaluator.hpp>
#include <gauxc/xc_integrator/local_work_driver.hpp>
#include <gauxc/gauxc_config.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef GAUXC_HAS_HDF5
#include <highfive/H5File.hpp>
#endif
#ifdef GAUXC_HAS_MPI
#include <mpi.h>
#endif

#include "standards.hpp"

// Reference implementation lives in src/, behind the public header surface.
#include "xc_integrator/local_work_driver/host/local_host_work_driver.hpp"

using namespace GauXC;

namespace {

/// Unique path for a test output file, inside the build-tree scratch
/// directory configured by CMake.
std::string make_temp_path(const char* suffix) {
  static int counter = 0;
  return std::string(GAUXC_TEST_TMP_PATH) + "/gauxc_test_" +
         std::to_string(++counter) + suffix;
}

OrbitalEvaluator make_evaluator(const BasisSet<double>& basis, double tol) {
  return OrbitalEvaluatorFactory::make_orbital_evaluator(ExecutionSpace::Host,
                                                         basis, tol);
}

/// AO collocation over every shell, i.e. the unscreened reference.
std::vector<double> reference_collocation(const BasisSet<double>& basis,
                                          int64_t npts, const double* pts) {
  const int32_t nbf = basis.nbf();
  std::vector<double> ao(static_cast<size_t>(nbf) * npts);
  auto drv = LocalWorkDriverFactory::make_local_work_driver(
      ExecutionSpace::Host, "Reference");
  auto* host_drv = dynamic_cast<LocalHostWorkDriver*>(drv.get());
  REQUIRE(host_drv != nullptr);
  std::vector<int32_t> shell_list(basis.size());
  for (size_t i = 0; i < shell_list.size(); ++i)
    shell_list[i] = static_cast<int32_t>(i);
  host_drv->eval_collocation(static_cast<size_t>(npts),
                             static_cast<size_t>(basis.nshells()),
                             static_cast<size_t>(nbf), pts, basis,
                             shell_list.data(), ao.data());
  return ao;
}

/// Build a deterministic PRNG-based set of points within a small bounding box
/// around the molecule. Avoids putting samples too close to nuclei to keep
/// values numerically well-behaved.
std::vector<double> make_random_points(int64_t npts, unsigned seed = 1234u) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> u(-2.5, 2.5);
  std::vector<double> pts(static_cast<size_t>(npts) * 3);
  for (int64_t p = 0; p < npts; ++p) {
    pts[3 * p + 0] = u(gen);
    pts[3 * p + 1] = u(gen);
    pts[3 * p + 2] = u(gen);
  }
  return pts;
}

std::vector<double> make_random_vector(size_t n, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> v(n);
  for (auto& x : v) x = u(gen);
  return v;
}

}  // namespace

TEST_CASE("OrbitalEvaluator / Water cc-pVDZ matches eval_collocation",
          "[orbital_evaluator]") {
  auto mol = make_water();
  auto basis = make_ccpvdz(mol, SphericalType(true));
  for (auto& sh : basis) sh.set_shell_tolerance(1e-12);

  const int32_t nbf = basis.nbf();
  REQUIRE(nbf > 0);

  const int64_t npts = 137;  // not a multiple of any batch size
  const auto pts = make_random_points(npts);

  // Reference: AO collocation directly via the LocalHostWorkDriver.
  const auto ao_ref = reference_collocation(basis, npts, pts.data());

  auto eval = make_evaluator(basis, 1e-12);
  REQUIRE(eval.nbf() == nbf);

  SECTION("eval_orbital with one-hot coefficient reproduces single AO column") {
    std::vector<double> C(nbf, 0.0);
    std::vector<double> out(static_cast<size_t>(npts), 0.0);
    for (int32_t mu : {0, nbf / 3, nbf / 2, nbf - 1}) {
      std::fill(C.begin(), C.end(), 0.0);
      C[static_cast<size_t>(mu)] = 1.0;
      std::fill(out.begin(), out.end(), 0.0);
      eval.eval_orbital(npts, pts.data(), C.data(), out.data());
      for (int64_t p = 0; p < npts; ++p) {
        const double ref =
            ao_ref[static_cast<size_t>(p) * nbf + static_cast<size_t>(mu)];
        CHECK(out[static_cast<size_t>(p)] == Approx(ref).margin(1e-12));
      }
    }
  }

  SECTION("eval_orbitals with random C matches AO ^T @ C") {
    const int32_t nmo = 4;
    const auto C = make_random_vector(static_cast<size_t>(nbf) * nmo, 99u);

    std::vector<double> out(static_cast<size_t>(npts) * nmo, 0.0);
    eval.eval_orbitals(npts, pts.data(), nmo, C.data(), nbf, out.data(), npts);

    for (int32_t j = 0; j < nmo; ++j) {
      for (int64_t p = 0; p < npts; ++p) {
        double ref = 0.0;
        for (int32_t mu = 0; mu < nbf; ++mu) {
          ref += C[static_cast<size_t>(j) * nbf + mu] *
                 ao_ref[static_cast<size_t>(p) * nbf + mu];
        }
        const double got =
            out[static_cast<size_t>(j) * npts + static_cast<size_t>(p)];
        CHECK(got == Approx(ref).margin(1e-10));
      }
    }
  }

  SECTION("eval_density with identity D equals sum of squared AO values") {
    std::vector<double> D(static_cast<size_t>(nbf) * nbf, 0.0);
    for (int32_t mu = 0; mu < nbf; ++mu) {
      D[static_cast<size_t>(mu) * nbf + mu] = 1.0;
    }
    std::vector<double> out(static_cast<size_t>(npts), 0.0);
    eval.eval_density(npts, pts.data(), D.data(), nbf, out.data());

    for (int64_t p = 0; p < npts; ++p) {
      double ref = 0.0;
      for (int32_t mu = 0; mu < nbf; ++mu) {
        const double a = ao_ref[static_cast<size_t>(p) * nbf + mu];
        ref += a * a;
      }
      CHECK(out[static_cast<size_t>(p)] == Approx(ref).margin(1e-10));
    }
  }

  SECTION("eval_density with rank-1 D = c c^T equals (c.AO)^2") {
    const auto c = make_random_vector(static_cast<size_t>(nbf), 7u);

    std::vector<double> D(static_cast<size_t>(nbf) * nbf, 0.0);
    for (int32_t mu = 0; mu < nbf; ++mu) {
      for (int32_t nu = 0; nu < nbf; ++nu) {
        D[static_cast<size_t>(mu) * nbf + nu] = c[mu] * c[nu];
      }
    }
    std::vector<double> out(static_cast<size_t>(npts), 0.0);
    eval.eval_density(npts, pts.data(), D.data(), nbf, out.data());

    std::vector<double> orb(static_cast<size_t>(npts), 0.0);
    eval.eval_orbital(npts, pts.data(), c.data(), orb.data());

    for (int64_t p = 0; p < npts; ++p) {
      const double ref = orb[static_cast<size_t>(p)] * orb[static_cast<size_t>(p)];
      CHECK(out[static_cast<size_t>(p)] == Approx(ref).margin(1e-10));
    }
  }

  SECTION("invalid leading dimensions throw") {
    std::vector<double> C(nbf, 0.0);
    std::vector<double> out(static_cast<size_t>(npts), 0.0);
    CHECK_THROWS(eval.eval_orbitals(npts, pts.data(), 1, C.data(), nbf - 1,
                                    out.data(), npts));
    CHECK_THROWS(eval.eval_orbitals(npts, pts.data(), 1, C.data(), nbf,
                                    out.data(), npts - 1));
    // ldo is narrowed to int for the BLAS call; oversized values must throw
    // rather than silently truncate.
    const size_t huge_ldo =
        static_cast<size_t>(std::numeric_limits<int>::max()) + 1;
    CHECK_THROWS(eval.eval_orbitals(npts, pts.data(), 1, C.data(), nbf,
                                    out.data(), huge_ldo));
  }
}

TEST_CASE("CubeGrid eval overloads match pointer-based eval",
          "[orbital_evaluator]") {
  auto mol = make_water();
  auto basis = make_ccpvdz(mol, SphericalType(true));
  for (auto& sh : basis) sh.set_shell_tolerance(1e-12);
  const int32_t nbf = basis.nbf();
  auto eval = make_evaluator(basis, 1e-12);

  // 6x7x8 fits in a single batch; 20x20x20 = 8000 points spreads over many
  // batches at any thread count, exercising the trailing partial batch and
  // the dynamic schedule.
  const std::vector<CubeGrid> grids = {CubeGrid::from_molecule(mol, 6, 7, 8),
                                       CubeGrid::from_molecule(mol, 20, 20, 20)};

  const auto C = make_random_vector(static_cast<size_t>(nbf), 42u);

  SECTION("eval_orbital(grid) matches eval_orbital(npts, points)") {
    for (const auto& grid : grids) {
      const int64_t npts = grid.num_points();
      const auto pts = grid.points();

      std::vector<double> ref(static_cast<size_t>(npts));
      eval.eval_orbital(npts, pts.data(), C.data(), ref.data());

      std::vector<double> out(static_cast<size_t>(npts));
      eval.eval_orbital(grid, C.data(), out.data());

      for (int64_t p = 0; p < npts; ++p) {
        CHECK(out[p] == Approx(ref[p]).margin(1e-12));
      }
    }
  }

  SECTION("eval_density(grid) matches eval_density(npts, points)") {
    // Identity density.
    std::vector<double> D(static_cast<size_t>(nbf) * nbf, 0.0);
    for (int32_t i = 0; i < nbf; ++i) D[i * nbf + i] = 1.0;

    for (const auto& grid : grids) {
      const int64_t npts = grid.num_points();
      const auto pts = grid.points();

      std::vector<double> ref(static_cast<size_t>(npts));
      eval.eval_density(npts, pts.data(), D.data(), nbf, ref.data());

      std::vector<double> out(static_cast<size_t>(npts));
      eval.eval_density(grid, D.data(), nbf, out.data());

      for (int64_t p = 0; p < npts; ++p) {
        CHECK(out[p] == Approx(ref[p]).margin(1e-12));
      }
    }
  }
}

TEST_CASE("OrbitalEvaluator shell screening", "[orbital_evaluator]") {
  // Two well-separated centres of different elements: batches near one centre
  // screen out every shell of the other, so 0 < nbe < nbf is reached, and
  // batches in the gap screen out everything (nbe == 0).
  Molecule mol;
  mol.emplace_back(AtomicNumber(8), 0.0, 0.0, 0.0);
  mol.emplace_back(AtomicNumber(1), 20.0, 0.0, 0.0);

  auto basis = make_ccpvdz(mol, SphericalType(true));
  constexpr double shell_tol = 1e-10;
  for (auto& sh : basis) sh.set_shell_tolerance(shell_tol);
  const int32_t nbf = basis.nbf();
  auto eval = make_evaluator(basis, shell_tol);

  const auto C = make_random_vector(static_cast<size_t>(nbf), 2024u);
  std::vector<double> D(static_cast<size_t>(nbf) * nbf, 0.0);
  for (int32_t i = 0; i < nbf; ++i) D[i * nbf + i] = 1.0;

  SECTION("a grid far from every atom evaluates to exactly zero") {
    CubeGrid far_grid;
    far_grid.origin = {100.0, 100.0, 100.0};
    far_grid.spacing = {0.5, 0.5, 0.5};
    far_grid.nx = far_grid.ny = far_grid.nz = 12;
    const int64_t npts = far_grid.num_points();

    std::vector<double> orb(static_cast<size_t>(npts), 1.0);
    eval.eval_orbital(far_grid, C.data(), orb.data());
    for (int64_t p = 0; p < npts; ++p) CHECK(orb[p] == 0.0);

    std::vector<double> rho(static_cast<size_t>(npts), 1.0);
    eval.eval_density(far_grid, D.data(), nbf, rho.data());
    for (int64_t p = 0; p < npts; ++p) CHECK(rho[p] == 0.0);
  }

  SECTION("partial screening agrees with the unscreened reference") {
    // Grid elongated along x (the slow axis), so a batch of consecutive
    // points is a narrow x-slab: near one centre, near the other, or in the
    // empty gap between them.
    CubeGrid grid;
    grid.origin = {-4.0, -4.0, -4.0};
    grid.spacing = {0.8, 2.0, 2.0};
    grid.nx = 40;
    grid.ny = 4;
    grid.nz = 4;
    const int64_t npts = grid.num_points();
    const auto pts = grid.points();
    const auto ao_ref = reference_collocation(basis, npts, pts.data());

    std::vector<double> orb(static_cast<size_t>(npts));
    eval.eval_orbital(grid, C.data(), orb.data());

    std::vector<double> rho(static_cast<size_t>(npts));
    eval.eval_density(grid, D.data(), nbf, rho.data());

    // The grid overload walks spatial tiles while the pointer overload takes
    // contiguous index ranges, so the two screen against different bounding
    // boxes. They agree to within the shell tolerance, not bitwise.
    std::vector<double> orb_pts(static_cast<size_t>(npts));
    eval.eval_orbital(npts, pts.data(), C.data(), orb_pts.data());
    std::vector<double> rho_pts(static_cast<size_t>(npts));
    eval.eval_density(npts, pts.data(), D.data(), nbf, rho_pts.data());

    bool any_nonzero = false;
    for (int64_t p = 0; p < npts; ++p) {
      CHECK(orb[p] == Approx(orb_pts[p]).margin(shell_tol));
      CHECK(rho[p] == Approx(rho_pts[p]).margin(shell_tol));

      double orb_ref = 0.0, rho_ref = 0.0;
      for (int32_t mu = 0; mu < nbf; ++mu) {
        const double a = ao_ref[static_cast<size_t>(p) * nbf + mu];
        orb_ref += C[static_cast<size_t>(mu)] * a;
        rho_ref += a * a;
      }
      if (std::fabs(orb_ref) > 1e-3) any_nonzero = true;
      // Screening discards shells whose magnitude is below the shell
      // tolerance across the batch bbox, so agreement is at that level.
      CHECK(orb[p] == Approx(orb_ref).margin(1e-9));
      CHECK(rho[p] == Approx(rho_ref).margin(1e-9));
    }
    CHECK(any_nonzero);
  }
}

TEST_CASE("OrbitalEvaluator tiled grid traversal matches the reference",
          "[orbital_evaluator]") {
  // Grids spanning well beyond a cutoff radius along z are walked in spatial
  // tiles rather than contiguous index ranges. The 32 Bohr z extent here is
  // several times the largest cc-pVDZ cutoff radius (13.2 Bohr), so the tiled
  // path stays selected; the two centres are far enough apart that screening
  // is active within it.
  Molecule mol;
  mol.emplace_back(AtomicNumber(8), 0.0, 0.0, 0.0);
  mol.emplace_back(AtomicNumber(1), 0.0, 0.0, 20.0);

  constexpr double shell_tol = 1e-10;
  auto basis = make_ccpvdz(mol, SphericalType(true));
  for (auto& sh : basis) sh.set_shell_tolerance(shell_tol);
  const int32_t nbf = basis.nbf();
  auto eval = make_evaluator(basis, shell_tol);

  CubeGrid grid;
  grid.origin = {-4.0, -4.0, -6.0};
  grid.spacing = {2.0, 2.0, 0.8};
  grid.nx = 4;
  grid.ny = 4;
  grid.nz = 40;
  const int64_t npts = grid.num_points();
  const auto pts = grid.points();
  const auto ao_ref = reference_collocation(basis, npts, pts.data());

  const auto C = make_random_vector(static_cast<size_t>(nbf), 8675u);
  std::vector<double> D(static_cast<size_t>(nbf) * nbf, 0.0);
  for (int32_t i = 0; i < nbf; ++i) D[i * nbf + i] = 1.0;

  std::vector<double> orb(static_cast<size_t>(npts));
  std::vector<double> rho(static_cast<size_t>(npts));
  eval.eval_orbital(grid, C.data(), orb.data());
  eval.eval_density(grid, D.data(), nbf, rho.data());

  // The pointer overload always uses contiguous batches, so it cross-checks
  // the tiled result against a different decomposition.
  std::vector<double> orb_pts(static_cast<size_t>(npts));
  std::vector<double> rho_pts(static_cast<size_t>(npts));
  eval.eval_orbital(npts, pts.data(), C.data(), orb_pts.data());
  eval.eval_density(npts, pts.data(), D.data(), nbf, rho_pts.data());

  bool any_nonzero = false;
  for (int64_t p = 0; p < npts; ++p) {
    double orb_ref = 0.0, rho_ref = 0.0;
    for (int32_t mu = 0; mu < nbf; ++mu) {
      const double a = ao_ref[static_cast<size_t>(p) * nbf + mu];
      orb_ref += C[static_cast<size_t>(mu)] * a;
      rho_ref += a * a;
    }
    if (std::fabs(orb_ref) > 1e-3) any_nonzero = true;
    CHECK(orb[p] == Approx(orb_ref).margin(1e-9));
    CHECK(rho[p] == Approx(rho_ref).margin(1e-9));
    CHECK(orb[p] == Approx(orb_pts[p]).margin(shell_tol));
    CHECK(rho[p] == Approx(rho_pts[p]).margin(shell_tol));
  }
  CHECK(any_nonzero);
}

TEST_CASE("OrbitalEvaluator scatters multiple orbitals into a padded output",
          "[orbital_evaluator]") {
  // A tiled batch is not contiguous in the output, so its result is staged and
  // scattered column by column. nmo > 1 with a padded ldo is the only
  // combination that exercises both strides of that scatter. Padding entries
  // hold sentinels: C must never be read past nbf, out never written past npts.
  Molecule mol;
  mol.emplace_back(AtomicNumber(8), 0.0, 0.0, 0.0);
  mol.emplace_back(AtomicNumber(1), 0.0, 0.0, 20.0);

  constexpr double shell_tol = 1e-10;
  auto basis = make_ccpvdz(mol, SphericalType(true));
  for (auto& sh : basis) sh.set_shell_tolerance(shell_tol);
  const int32_t nbf = basis.nbf();
  auto eval = make_evaluator(basis, shell_tol);

  CubeGrid grid;
  grid.origin = {-4.0, -4.0, -6.0};
  grid.spacing = {2.0, 2.0, 0.8};
  grid.nx = 4;
  grid.ny = 4;
  grid.nz = 40;
  const int64_t npts = grid.num_points();
  const auto pts = grid.points();
  const auto ao_ref = reference_collocation(basis, npts, pts.data());

  constexpr int32_t nmo = 3;
  constexpr double sentinel = -12345.0;
  const size_t ldc = static_cast<size_t>(nbf) + 7;
  const size_t ldo = static_cast<size_t>(npts) + 5;

  const auto Craw = make_random_vector(static_cast<size_t>(nbf) * nmo, 99u);
  std::vector<double> C(ldc * nmo, sentinel);
  for (int32_t j = 0; j < nmo; ++j)
    for (int32_t mu = 0; mu < nbf; ++mu)
      C[j * ldc + mu] = Craw[static_cast<size_t>(j) * nbf + mu];

  std::vector<double> out(ldo * nmo, sentinel);
  eval.eval_orbitals(grid, nmo, C.data(), ldc, out.data(), ldo);

  // The pointer overload always batches contiguously, so it checks the tiled
  // scatter against a different decomposition of the same grid.
  std::vector<double> out_pts(ldo * nmo, sentinel);
  eval.eval_orbitals(npts, pts.data(), nmo, C.data(), ldc, out_pts.data(), ldo);

  for (int32_t j = 0; j < nmo; ++j) {
    for (int64_t p = 0; p < npts; ++p) {
      double ref = 0.0;
      for (int32_t mu = 0; mu < nbf; ++mu)
        ref += Craw[static_cast<size_t>(j) * nbf + mu] *
               ao_ref[static_cast<size_t>(p) * nbf + mu];
      const size_t k = static_cast<size_t>(j) * ldo + static_cast<size_t>(p);
      CHECK(out[k] == Approx(ref).margin(1e-9));
      CHECK(out[k] == Approx(out_pts[k]).margin(shell_tol));
    }
    for (size_t p = static_cast<size_t>(npts); p < ldo; ++p) {
      CHECK(out[static_cast<size_t>(j) * ldo + p] == sentinel);
      CHECK(out_pts[static_cast<size_t>(j) * ldo + p] == sentinel);
    }
  }
}

TEST_CASE("OrbitalEvaluator handles single-plane grids", "[orbital_evaluator]") {
  // An axis of one point gets zero spacing from from_molecule, which the
  // tile-shaping code must treat as unconstrained rather than divide by.
  auto mol = make_water();
  constexpr double shell_tol = 1e-10;
  auto basis = make_ccpvdz(mol, SphericalType(true));
  for (auto& sh : basis) sh.set_shell_tolerance(shell_tol);
  const int32_t nbf = basis.nbf();
  auto eval = make_evaluator(basis, shell_tol);

  const auto C = make_random_vector(static_cast<size_t>(nbf), 4242u);
  std::vector<double> D(static_cast<size_t>(nbf) * nbf, 0.0);
  for (int32_t i = 0; i < nbf; ++i) D[static_cast<size_t>(i) * nbf + i] = 1.0;

  const std::vector<CubeGrid> grids = {CubeGrid::from_molecule(mol, 1, 9, 9),
                                       CubeGrid::from_molecule(mol, 9, 1, 9),
                                       CubeGrid::from_molecule(mol, 9, 9, 1)};

  for (const auto& grid : grids) {
    const int64_t npts = grid.num_points();
    const auto pts = grid.points();
    const auto ao = reference_collocation(basis, npts, pts.data());

    std::vector<double> orb(static_cast<size_t>(npts));
    std::vector<double> rho(static_cast<size_t>(npts));
    eval.eval_orbital(grid, C.data(), orb.data());
    eval.eval_density(grid, D.data(), nbf, rho.data());

    for (int64_t p = 0; p < npts; ++p) {
      double orb_ref = 0.0, rho_ref = 0.0;
      for (int32_t mu = 0; mu < nbf; ++mu) {
        const double a = ao[static_cast<size_t>(p) * nbf + mu];
        orb_ref += C[static_cast<size_t>(mu)] * a;
        rho_ref += a * a;
      }
      CHECK(orb[p] == Approx(orb_ref).margin(1e-9));
      CHECK(rho[p] == Approx(rho_ref).margin(1e-9));
    }
  }
}

TEST_CASE("OrbitalEvaluator screens a batch lying inside one grid row",
          "[orbital_evaluator]") {
  // With iz fastest, a contiguous batch shorter than one row is boxed by its
  // own z sub-range rather than the whole axis. Three things have to line up
  // for that box to matter. The row must outlast a batch, so nz exceeds the
  // batch size. The grid must stay off the tiled path, so the z extent sits
  // inside the tiling threshold of twice the median cutoff radius, derived
  // here from the basis rather than hard-coded. And the molecule must sit at
  // the far end of z, so that shrinking the box moves the face nearest the
  // shells and changes what survives screening.
  auto mol = make_water();

  constexpr double shell_tol = 1e-10;
  auto basis = make_ccpvdz(mol, SphericalType(true));
  for (auto& sh : basis) sh.set_shell_tolerance(shell_tol);
  const int32_t nbf = basis.nbf();

  std::vector<double> radii;
  radii.reserve(basis.size());
  for (const auto& sh : basis) radii.push_back(sh.cutoff_radius());
  const size_t mid = radii.size() / 2;
  std::nth_element(radii.begin(), radii.begin() + mid, radii.end());
  const double median_radius = radii[mid];

  const double z_extent = 1.8 * median_radius;

  CubeGrid grid;
  grid.nx = 1;
  grid.ny = 3;
  grid.nz = 3000;
  grid.spacing = {1.0, 1.0, z_extent / static_cast<double>(grid.nz)};
  // Offset in x so no point lands on a nucleus; the row runs from z_extent
  // below the molecule up to 1 Bohr short of it.
  grid.origin = {0.3, 0.0, -(z_extent + 1.0)};

  const int64_t npts = grid.num_points();
  const auto pts = grid.points();
  const auto ao_ref = reference_collocation(basis, npts, pts.data());

  const auto C = make_random_vector(static_cast<size_t>(nbf), 4099u);
  std::vector<double> D(static_cast<size_t>(nbf) * nbf, 0.0);
  for (int32_t i = 0; i < nbf; ++i) D[i * nbf + i] = 1.0;

  std::vector<double> orb(static_cast<size_t>(npts));
  std::vector<double> rho(static_cast<size_t>(npts));
  std::vector<double> orb_pts(static_cast<size_t>(npts));
  std::vector<double> rho_pts(static_cast<size_t>(npts));

#ifdef _OPENMP
  const int saved_threads = omp_get_max_threads();
  // Serial so the batch is npts/4 rather than thread-count dependent, which
  // puts three quarters of a row in one batch on any machine.
  omp_set_num_threads(1);
#endif
  auto eval = make_evaluator(basis, shell_tol);
  eval.eval_orbital(grid, C.data(), orb.data());
  eval.eval_density(grid, D.data(), nbf, rho.data());
  // The pointer overload boxes the points it is handed instead of deriving a
  // box from grid indices, so it checks the index arithmetic independently.
  eval.eval_orbital(npts, pts.data(), C.data(), orb_pts.data());
  eval.eval_density(npts, pts.data(), D.data(), nbf, rho_pts.data());
#ifdef _OPENMP
  omp_set_num_threads(saved_threads);
#endif

  for (int64_t p = 0; p < npts; ++p) {
    double orb_ref = 0.0, rho_ref = 0.0;
    for (int32_t mu = 0; mu < nbf; ++mu) {
      const double a = ao_ref[static_cast<size_t>(p) * nbf + mu];
      orb_ref += C[static_cast<size_t>(mu)] * a;
      rho_ref += a * a;
    }
    CHECK(orb[p] == Approx(orb_ref).margin(1e-9));
    CHECK(rho[p] == Approx(rho_ref).margin(1e-9));
    CHECK(orb[p] == Approx(orb_pts[p]).margin(shell_tol));
    CHECK(rho[p] == Approx(rho_pts[p]).margin(shell_tol));
  }

  // Pins the span the batch box is being asked to resolve: the row ends on
  // the molecule and starts far enough away for shells to have died off.
  const size_t row_end = static_cast<size_t>(grid.nz) - 1;
  CHECK(rho[row_end] > 1e-2);
  CHECK(rho[0] < 1e-4 * rho[row_end]);
}

TEST_CASE("OrbitalEvaluator survives a thread-count change after construction",
          "[orbital_evaluator]") {
  // Regression guard: scratch must follow the thread count in force at
  // evaluation time, not at construction. Batch shape is derived from the
  // thread count, so the two runs screen slightly differently and agree to
  // within the shell tolerance rather than bitwise.
  auto mol = make_water();
  auto basis = make_ccpvdz(mol, SphericalType(true));
  constexpr double shell_tol = 1e-12;
  for (auto& sh : basis) sh.set_shell_tolerance(shell_tol);
  const int32_t nbf = basis.nbf();

  auto grid = CubeGrid::from_molecule(mol, 16, 16, 16);
  const int64_t npts = grid.num_points();

  const auto C = make_random_vector(static_cast<size_t>(nbf), 5150u);
  std::vector<double> D(static_cast<size_t>(nbf) * nbf, 0.0);
  for (int32_t i = 0; i < nbf; ++i) D[i * nbf + i] = 1.0;

#ifdef _OPENMP
  const int saved_threads = omp_get_max_threads();
  // Construct while the thread count is 1, then raise it. Per-call scratch
  // must follow the thread count in force at evaluation time.
  omp_set_num_threads(1);
#endif
  auto eval = make_evaluator(basis, shell_tol);

  std::vector<double> orb_serial(static_cast<size_t>(npts));
  std::vector<double> rho_serial(static_cast<size_t>(npts));
  eval.eval_orbital(grid, C.data(), orb_serial.data());
  eval.eval_density(grid, D.data(), nbf, rho_serial.data());

#ifdef _OPENMP
  omp_set_num_threads(saved_threads > 1 ? saved_threads : 2);
#endif

  std::vector<double> orb_par(static_cast<size_t>(npts));
  std::vector<double> rho_par(static_cast<size_t>(npts));
  eval.eval_orbital(grid, C.data(), orb_par.data());
  eval.eval_density(grid, D.data(), nbf, rho_par.data());

#ifdef _OPENMP
  omp_set_num_threads(saved_threads);
#endif

  for (int64_t p = 0; p < npts; ++p) {
    CHECK(orb_par[p] == Approx(orb_serial[p]).margin(shell_tol));
    CHECK(rho_par[p] == Approx(rho_serial[p]).margin(shell_tol));
  }
}

TEST_CASE("OrbitalEvaluator thread-count dependence is bounded by screening",
          "[orbital_evaluator]") {
  // Batch size is derived from the thread count, so when screening is active
  // different thread counts screen against different batch bounding boxes and
  // the results are not bit-identical. The discrepancy is bounded by the shell
  // tolerance, which is what this pins down.
  Molecule mol;
  mol.emplace_back(AtomicNumber(8), 0.0, 0.0, 0.0);
  mol.emplace_back(AtomicNumber(1), 20.0, 0.0, 0.0);
  auto basis = make_ccpvdz(mol, SphericalType(true));
  constexpr double shell_tol = 1e-10;
  for (auto& sh : basis) sh.set_shell_tolerance(shell_tol);
  const int32_t nbf = basis.nbf();
  auto eval = make_evaluator(basis, shell_tol);

  CubeGrid grid;
  grid.origin = {-4.0, -4.0, -4.0};
  grid.spacing = {0.8, 2.0, 2.0};
  grid.nx = 40;
  grid.ny = 4;
  grid.nz = 4;
  const int64_t npts = grid.num_points();

  std::vector<double> D(static_cast<size_t>(nbf) * nbf, 0.0);
  for (int32_t i = 0; i < nbf; ++i) D[i * nbf + i] = 1.0;

  std::vector<double> rho_serial(static_cast<size_t>(npts));
  std::vector<double> rho_par(static_cast<size_t>(npts));

#ifdef _OPENMP
  const int saved_threads = omp_get_max_threads();
  omp_set_num_threads(1);
#endif
  eval.eval_density(grid, D.data(), nbf, rho_serial.data());
#ifdef _OPENMP
  omp_set_num_threads(saved_threads > 1 ? saved_threads : 2);
#endif
  eval.eval_density(grid, D.data(), nbf, rho_par.data());
#ifdef _OPENMP
  omp_set_num_threads(saved_threads);
#endif

  for (int64_t p = 0; p < npts; ++p) {
    CHECK(rho_par[p] == Approx(rho_serial[p]).margin(shell_tol));
  }
}

TEST_CASE("OrbitalEvaluator screening error scales with the shell tolerance",
          "[orbital_evaluator]") {
  // Pins the guidance on the class: orbital error is bounded by the shell
  // tolerance, density error by its square, since dropping a shell with
  // |phi| < t perturbs sum_uv D phi phi by ~t^2.
  Molecule mol;
  mol.emplace_back(AtomicNumber(8), 0.0, 0.0, 0.0);
  mol.emplace_back(AtomicNumber(1), 20.0, 0.0, 0.0);

  constexpr double loose_tol = 1e-6;
  auto basis = make_ccpvdz(mol, SphericalType(true));
  const int32_t nbf = basis.nbf();

  std::vector<double> radii_before;
  for (const auto& sh : basis) radii_before.push_back(sh.cutoff_radius());

  auto eval_tight = make_evaluator(basis, 1e-14);
  auto eval_loose = make_evaluator(basis, loose_tol);

  // Retuning happens on each evaluator's own copy; a basis shared with an SCF
  // setup must come back unchanged.
  for (size_t i = 0; i < basis.size(); ++i)
    CHECK(basis[i].cutoff_radius() == radii_before[i]);

  CubeGrid grid;
  grid.origin = {-4.0, -4.0, -4.0};
  grid.spacing = {0.8, 2.0, 2.0};
  grid.nx = 40;
  grid.ny = 4;
  grid.nz = 4;
  const int64_t npts = grid.num_points();

  const auto C = make_random_vector(static_cast<size_t>(nbf), 31337u);
  std::vector<double> D(static_cast<size_t>(nbf) * nbf, 0.0);
  for (int32_t i = 0; i < nbf; ++i) D[i * nbf + i] = 1.0;

  std::vector<double> orb_tight(npts), orb_loose(npts);
  std::vector<double> rho_tight(npts), rho_loose(npts);
  eval_tight.eval_orbital(grid, C.data(), orb_tight.data());
  eval_loose.eval_orbital(grid, C.data(), orb_loose.data());
  eval_tight.eval_density(grid, D.data(), nbf, rho_tight.data());
  eval_loose.eval_density(grid, D.data(), nbf, rho_loose.data());

  double orb_err = 0.0, rho_err = 0.0;
  for (int64_t p = 0; p < npts; ++p) {
    orb_err = std::max(orb_err, std::fabs(orb_loose[p] - orb_tight[p]));
    rho_err = std::max(rho_err, std::fabs(rho_loose[p] - rho_tight[p]));
  }

  INFO("orbital error " << orb_err << ", density error " << rho_err);
  CHECK(orb_err > 0.0);  // screening is genuinely active at the loose tolerance
  CHECK(orb_err <= 10.0 * loose_tol);
  CHECK(rho_err <= 100.0 * loose_tol * loose_tol);
}

TEST_CASE("CubeGrid construction and grid-points layout", "[cube]") {
  auto mol = make_water();
  CubeGrid g = CubeGrid::from_molecule(mol, /*nx=*/16, /*ny=*/12, /*nz=*/8,
                                       /*margin=*/2.0);
  REQUIRE(g.num_points() == 16 * 12 * 8);

  auto pts = g.points();
  REQUIRE(pts.size() == static_cast<size_t>(g.num_points()) * 3);

  // Check first and last grid points.
  CHECK(pts[0] == Approx(g.origin[0]));
  CHECK(pts[1] == Approx(g.origin[1]));
  CHECK(pts[2] == Approx(g.origin[2]));

  const size_t last = static_cast<size_t>(g.num_points()) - 1;
  CHECK(pts[3 * last + 0] ==
        Approx(g.origin[0] + g.spacing[0] * (g.nx - 1)));
  CHECK(pts[3 * last + 1] ==
        Approx(g.origin[1] + g.spacing[1] * (g.ny - 1)));
  CHECK(pts[3 * last + 2] ==
        Approx(g.origin[2] + g.spacing[2] * (g.nz - 1)));

  // Check ordering: iz varies fastest. Point (1, 0, 0) should be at offset
  // ny*nz, point (0, 1, 0) at nz, point (0, 0, 1) at 1.
  const int64_t off_x = g.ny * g.nz;
  const int64_t off_y = g.nz;
  CHECK(pts[3 * static_cast<size_t>(off_x) + 0] ==
        Approx(g.origin[0] + g.spacing[0]));
  CHECK(pts[3 * static_cast<size_t>(off_y) + 1] ==
        Approx(g.origin[1] + g.spacing[1]));
  CHECK(pts[3 * 1 + 2] == Approx(g.origin[2] + g.spacing[2]));
}

TEST_CASE("write_cube round-trips header and field data", "[cube]") {
#ifdef GAUXC_HAS_MPI
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_rank) return;  // File I/O; only run on root rank
#endif
  auto mol = make_water();
  CubeGrid grid = CubeGrid::from_molecule(mol, /*nx=*/5, /*ny=*/4, /*nz=*/7,
                                          /*margin=*/2.0);
  std::vector<double> field(static_cast<size_t>(grid.num_points()));
  for (int64_t i = 0; i < grid.num_points(); ++i) {
    // Mix of magnitudes to exercise the formatter (negative, near-zero, etc.)
    field[static_cast<size_t>(i)] = std::sin(0.13 * static_cast<double>(i)) *
                                    std::pow(10.0, (i % 5) - 2);
  }

  const std::string path = make_temp_path(".cube");
  write_cube(path, mol, grid, field.data(), "Test cube");

  // Parse back.
  std::ifstream in(path);
  REQUIRE(in.is_open());
  std::string l1, l2;
  std::getline(in, l1);
  std::getline(in, l2);
  CHECK(l1 == "Test cube");
  CHECK(l2 == "Generated by GauXC");

  long long natoms_read;
  double ox, oy, oz;
  in >> natoms_read >> ox >> oy >> oz;
  CHECK(natoms_read == static_cast<long long>(mol.size()));
  CHECK(ox == Approx(grid.origin[0]));
  CHECK(oy == Approx(grid.origin[1]));
  CHECK(oz == Approx(grid.origin[2]));

  // 3 axis lines.
  for (int axis = 0; axis < 3; ++axis) {
    long long n;
    double a, b, c;
    in >> n >> a >> b >> c;
    if (axis == 0) {
      CHECK(n == grid.nx);
      CHECK(a == Approx(grid.spacing[0]));
      CHECK(b == Approx(0.0));
      CHECK(c == Approx(0.0));
    } else if (axis == 1) {
      CHECK(n == grid.ny);
      CHECK(a == Approx(0.0));
      CHECK(b == Approx(grid.spacing[1]));
      CHECK(c == Approx(0.0));
    } else {
      CHECK(n == grid.nz);
      CHECK(a == Approx(0.0));
      CHECK(b == Approx(0.0));
      CHECK(c == Approx(grid.spacing[2]));
    }
  }

  // Atoms.
  for (size_t i = 0; i < mol.size(); ++i) {
    long long Z;
    double q, x, y, z;
    in >> Z >> q >> x >> y >> z;
    CHECK(Z == mol[i].Z.get());
    CHECK(q == Approx(0.0));
    CHECK(x == Approx(mol[i].x));
    CHECK(y == Approx(mol[i].y));
    CHECK(z == Approx(mol[i].z));
  }

  // Field. Read all remaining whitespace-separated doubles and compare.
  std::vector<double> field_read;
  field_read.reserve(static_cast<size_t>(grid.num_points()));
  double v;
  while (in >> v) field_read.push_back(v);

  REQUIRE(field_read.size() == field.size());
  // %13.5E gives 5 significant digits → relative tolerance ~1e-5 for the
  // round-trip.
  for (size_t i = 0; i < field.size(); ++i) {
    CHECK(field_read[i] == Approx(field[i]).epsilon(1e-4).margin(1e-30));
  }
  in.close();

  // Line structure: each (ix,iy) row spans ceil(nz/6) lines, full lines carry
  // six 13-char fields and the row's last line carries the remainder. This
  // pins the exact per-row byte count that write_cube relies on to place rows
  // without a compaction pass.
  {
    std::ifstream lin(path);
    REQUIRE(lin.is_open());
    std::string line;
    for (size_t i = 0; i < 6 + mol.size(); ++i) std::getline(lin, line);

    const int64_t lines_per_row = (grid.nz + 5) / 6;
    for (int64_t row = 0; row < grid.nx * grid.ny; ++row) {
      for (int64_t l = 0; l < lines_per_row; ++l) {
        REQUIRE(static_cast<bool>(std::getline(lin, line)));
        const int64_t nvals = (l + 1 == lines_per_row) ? (grid.nz - l * 6) : 6;
        CHECK(line.size() == static_cast<size_t>(13 * nvals));
      }
    }
    CHECK_FALSE(static_cast<bool>(std::getline(lin, line)));
  }

  std::remove(path.c_str());
}

namespace {

/// Write `field` as a single-row cube file and return the raw data block.
std::string cube_data_block(const Molecule& mol,
                            const std::vector<double>& field) {
  CubeGrid grid;
  grid.origin = {0.0, 0.0, 0.0};
  grid.spacing = {0.1, 0.1, 0.1};
  grid.nx = 1;
  grid.ny = 1;
  grid.nz = static_cast<int64_t>(field.size());

  const std::string path = make_temp_path(".cube");
  write_cube(path, mol, grid, field.data(), "fmt");

  std::ifstream in(path);
  REQUIRE(in.is_open());
  // Skip header (2 comments + 1 natoms line + 3 axis lines + natoms atoms).
  std::string skip;
  for (int i = 0; i < 6; ++i) std::getline(in, skip);
  for (size_t i = 0; i < mol.size(); ++i) std::getline(in, skip);

  std::string block((std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());
  in.close();
  std::remove(path.c_str());
  return block;
}

}  // namespace

TEST_CASE("write_cube agrees with snprintf %13.5E formatting", "[cube]") {
#ifdef GAUXC_HAS_MPI
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_rank) return;  // File I/O; only run on root rank
#endif
  // Spot-check the custom formatter against snprintf for a battery of values
  // by writing a tiny cube file and parsing it back. This is a stronger check
  // than the round-trip above since we compare the byte stream.
  auto mol = make_water();

  const double qnan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  // The last five exercise the snprintf deferral: subnormals (where scaling
  // the mantissa would overflow), a 3-digit exponent either side of zero, and
  // a mantissa that carries from E+99 up into a 3-digit exponent.
  std::vector<double> field = {0.0,
                               -0.0,
                               1.23456e-10,
                               -9.99995e-1,
                               1.0e+99,
                               -1.0e+99,
                               3.14159265358979,
                               -2.71828,
                               1.0,
                               -1.0,
                               1e-300,
                               1.234e+05,
                               qnan,
                               -qnan,
                               inf,
                               -inf,
                               std::numeric_limits<double>::denorm_min(),
                               -1e-310,
                               1.0e+300,
                               -1.0e-305,
                               9.9999999e+99};

  const std::string data_block = cube_data_block(mol, field);

  std::ostringstream expected;
  for (size_t i = 0; i < field.size(); ++i) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%13.5E", field[i]);
    expected << buf;
    if ((i + 1) % 6 == 0 || (i + 1) == field.size()) expected << '\n';
  }

  CHECK(data_block == expected.str());
}

TEST_CASE("write_cube spans multiple output chunks", "[cube]") {
#ifdef GAUXC_HAS_MPI
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_rank) return;  // File I/O; only run on root rank
#endif
  // Sized to exceed write_cube's internal staging-buffer target so the chunk
  // loop runs more than once, with a partial final chunk.
  auto mol = make_water();
  CubeGrid grid;
  grid.origin = {0.0, 0.0, 0.0};
  grid.spacing = {0.1, 0.1, 0.1};
  grid.nx = 2;
  grid.ny = 100;
  grid.nz = 4096;
  const int64_t npts = grid.num_points();

  std::vector<double> field(static_cast<size_t>(npts));
  for (int64_t i = 0; i < npts; ++i)
    field[static_cast<size_t>(i)] = std::sin(1e-3 * static_cast<double>(i));

  const std::string path = make_temp_path(".cube");
  write_cube(path, mol, grid, field.data(), "chunked");

  std::ifstream in(path, std::ios::binary);
  REQUIRE(in.is_open());
  std::string skip;
  for (size_t i = 0; i < 6 + mol.size(); ++i) std::getline(in, skip);

  std::string data_block((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  in.close();

  const int64_t bytes_per_row = grid.nz * 13 + (grid.nz + 5) / 6;
  const int64_t n_rows = grid.nx * grid.ny;
  REQUIRE(data_block.size() == static_cast<size_t>(bytes_per_row * n_rows));

  // Full byte comparison catches any misplacement at a chunk seam.
  std::string expected;
  expected.reserve(data_block.size());
  char buf[32];
  for (int64_t row = 0; row < n_rows; ++row) {
    for (int64_t iz = 0; iz < grid.nz; ++iz) {
      std::snprintf(buf, sizeof(buf), "%13.5E",
                    field[static_cast<size_t>(row * grid.nz + iz)]);
      expected += buf;
      if ((iz + 1) % 6 == 0 || iz + 1 == grid.nz) expected += '\n';
    }
  }
  CHECK(data_block == expected);

  std::remove(path.c_str());
}

TEST_CASE("write_cube formatter stays within one last digit at a rounding "
          "boundary",
          "[cube]") {
#ifdef GAUXC_HAS_MPI
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_rank) return;  // File I/O; only run on root rank
#endif
  // For values sitting within a few ulp of a rounding boundary in the 6th
  // significant digit, scaling the mantissa can tip it across the boundary,
  // so the formatter and glibc may pick different last digits. Both stay
  // within one unit of it; this pins that bound rather than byte equality.
  auto mol = make_water();
  const std::vector<double> field = {123456.5, 1.234575, -123456.5, -1.234575,
                                     9.9999949999999998642e-98, 0.0};

  const std::string data_block = cube_data_block(mol, field);
  REQUIRE(data_block.size() >= field.size() * 13);

  for (size_t i = 0; i < field.size(); ++i) {
    const std::string tok = data_block.substr(i * 13, 13);
    const double got = std::stod(tok);
    const double v = field[i];
    const double last_digit =
        v == 0.0 ? 1.0
                 : std::pow(10.0, std::floor(std::log10(std::fabs(v))) - 5.0);
    INFO("value " << v << " formatted as '" << tok << "'");
    CHECK(std::fabs(got - v) <= 0.5000001 * last_digit);
  }
}

#ifdef GAUXC_HAS_HDF5
TEST_CASE("write_cube_hdf5 round-trip", "[cube]") {
#ifdef GAUXC_HAS_MPI
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_rank) return;  // File I/O; only run on root rank
#endif
  auto mol = make_water();
  auto grid = CubeGrid::from_molecule(mol, 4, 5, 6);

  // Fill a small field with known values.
  const int64_t npts = grid.num_points();
  std::vector<double> field(npts);
  for (int64_t i = 0; i < npts; ++i) field[i] = 0.01 * i - 0.5;

  const std::string path = make_temp_path(".h5");
  write_cube_hdf5(path, mol, grid, field.data(), "test cube");

  // Read back and verify.
  HighFive::File file(path, HighFive::File::ReadOnly);

  // Field shape and values.
  auto ds = file.getDataSet("field");
  auto dims = ds.getDimensions();
  REQUIRE(dims.size() == 3);
  CHECK(dims[0] == static_cast<size_t>(grid.nx));
  CHECK(dims[1] == static_cast<size_t>(grid.ny));
  CHECK(dims[2] == static_cast<size_t>(grid.nz));

  std::vector<double> read_field(npts);
  ds.read(read_field.data());
  for (int64_t i = 0; i < npts; ++i) {
    CHECK(read_field[i] == Approx(field[i]).epsilon(1e-14));
  }

  // Comment attribute.
  std::string cmt;
  ds.getAttribute("comment").read(cmt);
  CHECK(cmt == "test cube");

  // Grid metadata.
  auto grp_grid = file.getGroup("grid");
  std::vector<double> origin, spacing;
  std::vector<int64_t> shape;
  grp_grid.getDataSet("origin").read(origin);
  grp_grid.getDataSet("spacing").read(spacing);
  grp_grid.getDataSet("shape").read(shape);
  REQUIRE(origin.size() == 3);
  REQUIRE(spacing.size() == 3);
  REQUIRE(shape.size() == 3);
  for (int k = 0; k < 3; ++k) {
    CHECK(origin[k] == Approx(grid.origin[k]).epsilon(1e-14));
    CHECK(spacing[k] == Approx(grid.spacing[k]).epsilon(1e-14));
  }
  CHECK(shape[0] == grid.nx);
  CHECK(shape[1] == grid.ny);
  CHECK(shape[2] == grid.nz);

  // Atoms.
  auto grp_atoms = file.getGroup("atoms");
  std::vector<int64_t> Z;
  grp_atoms.getDataSet("Z").read(Z);
  REQUIRE(Z.size() == mol.size());
  for (size_t i = 0; i < mol.size(); ++i) {
    CHECK(Z[i] == static_cast<int64_t>(mol[i].Z.get()));
  }

  std::vector<double> coords(mol.size() * 3);
  grp_atoms.getDataSet("coords").read(coords.data());
  for (size_t i = 0; i < mol.size(); ++i) {
    CHECK(coords[3 * i + 0] == Approx(mol[i].x).epsilon(1e-14));
    CHECK(coords[3 * i + 1] == Approx(mol[i].y).epsilon(1e-14));
    CHECK(coords[3 * i + 2] == Approx(mol[i].z).epsilon(1e-14));
  }

  std::remove(path.c_str());
}
#endif
