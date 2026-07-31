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

OrbitalEvaluator make_evaluator(const BasisSet<double>& basis) {
  return OrbitalEvaluatorFactory::make_orbital_evaluator(ExecutionSpace::Host,
                                                         basis);
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

  auto eval = make_evaluator(basis);
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
  auto eval = make_evaluator(basis);

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
  for (auto& sh : basis) sh.set_shell_tolerance(1e-10);
  const int32_t nbf = basis.nbf();
  auto eval = make_evaluator(basis);

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

    // The grid overload screens against an analytically derived batch bbox
    // while the pointer overload scans the coordinates; with screening active
    // the two must still agree exactly.
    std::vector<double> orb_pts(static_cast<size_t>(npts));
    eval.eval_orbital(npts, pts.data(), C.data(), orb_pts.data());
    std::vector<double> rho_pts(static_cast<size_t>(npts));
    eval.eval_density(npts, pts.data(), D.data(), nbf, rho_pts.data());

    bool any_nonzero = false;
    for (int64_t p = 0; p < npts; ++p) {
      CHECK(orb[p] == orb_pts[p]);
      CHECK(rho[p] == rho_pts[p]);

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

TEST_CASE("OrbitalEvaluator is invariant to the OpenMP thread count",
          "[orbital_evaluator]") {
  // Bit-exact here because the grid encloses the molecule at a 1e-12 shell
  // tolerance, so nothing screens and the batch decomposition cannot change
  // the arithmetic. The screened case is covered separately below.
  auto mol = make_water();
  auto basis = make_ccpvdz(mol, SphericalType(true));
  for (auto& sh : basis) sh.set_shell_tolerance(1e-12);
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
  auto eval = make_evaluator(basis);

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
    CHECK(orb_par[p] == orb_serial[p]);
    CHECK(rho_par[p] == rho_serial[p]);
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
  auto eval = make_evaluator(basis);

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
  std::vector<double> field = {0.0,      -0.0,     1.23456e-10, -9.99995e-1,
                               1.0e+99,  -1.0e+99, 3.14159265358979,
                               -2.71828, 1.0,      -1.0,        1e-300,
                               1.234e+05, qnan,    -qnan,       inf,
                               -inf};

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

TEST_CASE("write_cube formatter rounds half-way ties within one last digit",
          "[cube]") {
#ifdef GAUXC_HAS_MPI
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_rank) return;  // File I/O; only run on root rank
#endif
  // The hand-rolled formatter rounds exact half-way ties in the 6th
  // significant digit half-away-from-zero, whereas glibc rounds the exact
  // binary value half-to-even. Both are within one unit of the last printed
  // digit; this pins that bound rather than byte equality.
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
