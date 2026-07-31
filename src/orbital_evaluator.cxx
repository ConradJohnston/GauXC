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
#include "orbital_evaluator_impl.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
#endif

#include <gauxc/exceptions.hpp>
#include <gauxc/molecule.hpp>

#include "xc_integrator/integrator_util/integrator_common.hpp"
#include "xc_integrator/local_work_driver/host/blas.hpp"

namespace GauXC {

namespace detail {

OrbitalEvaluatorImpl::OrbitalEvaluatorImpl( BasisSet<double> bs ) :
  basis( std::move(bs) ) {

  driver_owner = LocalWorkDriverFactory::make_local_work_driver(
    ExecutionSpace::Host, "Reference" );
  host_driver = dynamic_cast<LocalHostWorkDriver*>( driver_owner.get() );
  if( not host_driver ) {
    GAUXC_GENERIC_EXCEPTION("OrbitalEvaluator: LocalWorkDriverFactory did not "
      "return a LocalHostWorkDriver.");
  }

  nbf_ = basis.nbf();

  // BasisSetMap powers shell -> AO range lookups for the screened submat_map.
  // No Molecule is available here, so an empty one is passed; the only field
  // that needs it (shell_to_center) is unused by this class.
  basis_map = std::make_unique<BasisSetMap>( basis, Molecule{} );

  // Cache squared cutoff radii so the screening loop is a comparison rather
  // than a square root per shell per batch.
  shell_cutoff_r2.resize( basis.size() );
  for( size_t s = 0; s < basis.size(); ++s ) {
    const double r = basis[s].cutoff_radius();
    shell_cutoff_r2[s] = r * r;
  }

}

}  // namespace detail


namespace {

/** @brief Choose the number of points evaluated per batch.
 *
 *  Two constraints, whichever is tighter:
 *    1. Cache footprint: the AO scratch (nbf*batch doubles) should stay
 *       around 1 MiB per thread so it lives comfortably in L2/L3.
 *    2. Load balance: enough batches that each thread gets several, i.e.
 *       batch <= npts / (4*nthreads).
 */
size_t choose_batch_size( int32_t nbf, size_t npts, int nthreads ) {
  constexpr size_t kTargetAOBytesPerThread = 1024 * 1024;
  constexpr size_t kMinBatch = 128;
  constexpr size_t kMaxBatch = 8192;

  size_t batch = kMaxBatch;
  if( nbf > 0 ) {
    batch = kTargetAOBytesPerThread / ( sizeof(double) * static_cast<size_t>(nbf) );
  }
  if( nthreads > 0 ) {
    batch = std::min( batch, npts / ( 4 * static_cast<size_t>(nthreads) ) );
  }
  return std::clamp( batch, kMinBatch, kMaxBatch );
}

/// Axis-aligned bounding box of a point set.
struct PointBbox {
  std::array<double, 3> lo;
  std::array<double, 3> hi;
};

/// Bbox of `npts` AoS points (array of length 3*npts).
PointBbox compute_bbox( const double* points, size_t npts ) {
  PointBbox b{ {points[0], points[1], points[2]},
               {points[0], points[1], points[2]} };
  for( size_t p = 1; p < npts; ++p ) {
    const double* xyz = points + 3 * p;
    for( int k = 0; k < 3; ++k ) {
      if( xyz[k] < b.lo[k] ) b.lo[k] = xyz[k];
      if( xyz[k] > b.hi[k] ) b.hi[k] = xyz[k];
    }
  }
  return b;
}

/** @brief Bbox of the contiguous CubeGrid index range [p0, p0+np).
 *
 *  Exact (not merely conservative): with iz varying fastest, a range that
 *  crosses an ix boundary necessarily contains both iy=0 and iy=ny-1, and a
 *  range that crosses an iy boundary necessarily contains both iz=0 and
 *  iz=nz-1. Computing this analytically avoids rescanning the 3*np
 *  coordinates that were just generated.
 */
PointBbox bbox_for_range( const CubeGrid& g, size_t p0, size_t np ) {
  const int64_t nz = g.nz, ny = g.ny;
  const int64_t k0 = static_cast<int64_t>(p0);
  const int64_t k1 = static_cast<int64_t>(p0 + np) - 1;

  int64_t lo_idx[3], hi_idx[3];
  lo_idx[0] = k0 / (ny * nz);
  hi_idx[0] = k1 / (ny * nz);
  if( hi_idx[0] > lo_idx[0] ) {
    lo_idx[1] = 0;      hi_idx[1] = ny - 1;
    lo_idx[2] = 0;      hi_idx[2] = nz - 1;
  } else {
    lo_idx[1] = (k0 / nz) % ny;
    hi_idx[1] = (k1 / nz) % ny;
    if( hi_idx[1] > lo_idx[1] ) {
      lo_idx[2] = 0;    hi_idx[2] = nz - 1;
    } else {
      lo_idx[2] = k0 % nz;
      hi_idx[2] = k1 % nz;
    }
  }

  PointBbox b;
  for( int k = 0; k < 3; ++k ) {
    const double a = g.origin[k] + g.spacing[k] * static_cast<double>(lo_idx[k]);
    const double c = g.origin[k] + g.spacing[k] * static_cast<double>(hi_idx[k]);
    b.lo[k] = std::min(a, c);
    b.hi[k] = std::max(a, c);
  }
  return b;
}

/// Squared distance from `center` to the nearest point of the bbox. Zero if
/// the center lies inside.
double dist2_center_to_bbox( const double* center, const PointBbox& bbox ) {
  double d2 = 0.0;
  for( int k = 0; k < 3; ++k ) {
    const double c = center[k];
    if( c < bbox.lo[k] ) {
      const double dx = bbox.lo[k] - c;
      d2 += dx * dx;
    } else if( c > bbox.hi[k] ) {
      const double dx = c - bbox.hi[k];
      d2 += dx * dx;
    }
  }
  return d2;
}

/// Point source backed by a caller-supplied AoS coordinate array.
struct RawPointSource {
  static constexpr bool needs_scratch = false;
  const double* points;

  const double* batch( size_t p0, size_t, double* ) const {
    return points + 3 * p0;
  }
  PointBbox bbox( size_t, size_t np, const double* pts ) const {
    return compute_bbox( pts, np );
  }
};

/// Point source that generates CubeGrid coordinates on the fly, avoiding the
/// 3*num_points()*8 byte temporary coordinate array.
struct GridPointSource {
  static constexpr bool needs_scratch = true;
  const CubeGrid* grid;

  const double* batch( size_t p0, size_t np, double* scr ) const {
    // Incremental (ix,iy,iz) counters; coordinates are recomputed from the
    // index (rather than accumulated) to stay bit-identical to
    // CubeGrid::points_into.
    const CubeGrid& g = *grid;
    const int64_t nz = g.nz, ny = g.ny;
    int64_t iz = static_cast<int64_t>(p0) % nz;
    int64_t iy = (static_cast<int64_t>(p0) / nz) % ny;
    int64_t ix = static_cast<int64_t>(p0) / (ny * nz);
    double x = g.origin[0] + g.spacing[0] * static_cast<double>(ix);
    double y = g.origin[1] + g.spacing[1] * static_cast<double>(iy);

    for( size_t i = 0; i < np; ++i ) {
      scr[3 * i + 0] = x;
      scr[3 * i + 1] = y;
      scr[3 * i + 2] = g.origin[2] + g.spacing[2] * static_cast<double>(iz);
      if( ++iz == nz ) {
        iz = 0;
        if( ++iy == ny ) {
          iy = 0;
          ++ix;
          x = g.origin[0] + g.spacing[0] * static_cast<double>(ix);
        }
        y = g.origin[1] + g.spacing[1] * static_cast<double>(iy);
      }
    }
    return scr;
  }

  PointBbox bbox( size_t p0, size_t np, const double* ) const {
    return bbox_for_range( *grid, p0, np );
  }
};

/// Contraction of the AO batch against MO coefficients:
/// out(np,nmo) = ao^T(np,nbe) @ C(nbe,nmo).
class OrbitalContractor {

  const detail::OrbitalEvaluatorImpl& impl_;
  int32_t nmo_;
  const double* C_;
  size_t ldc_;
  double* out_;
  size_t ldo_;

public:

  struct Scratch { std::vector<double> C_compressed; };

  OrbitalContractor( const detail::OrbitalEvaluatorImpl& impl, int32_t nmo,
                     const double* C, size_t ldc, double* out, size_t ldo ) :
    impl_(impl), nmo_(nmo), C_(C), ldc_(ldc), out_(out), ldo_(ldo) {}

  void init( Scratch& scr, size_t ) const {
    scr.C_compressed.resize( static_cast<size_t>(impl_.nbf_) * nmo_ );
  }

  void zero( size_t p0, size_t np ) const {
    for( int32_t j = 0; j < nmo_; ++j ) {
      double* out_col = out_ + static_cast<size_t>(j) * ldo_ + p0;
      std::fill( out_col, out_col + np, 0.0 );
    }
  }

  void apply( Scratch& scr, size_t p0, size_t np, int32_t nbe,
              const std::vector<int32_t>& shells, const double* ao ) const {

    // Gather the rows of C for the surviving shells into a contiguous
    // (nbe, nmo) col-major buffer so the contraction is a dense GEMM.
    const BasisSetMap& basis_map = *impl_.basis_map;
    int32_t row = 0;
    for( int32_t ish : shells ) {
      const auto rng = basis_map.shell_to_ao_range(ish);
      for( int32_t j = 0; j < nmo_; ++j ) {
        const double* C_col = C_ + static_cast<size_t>(j) * ldc_;
        double* dst = scr.C_compressed.data() +
                      static_cast<size_t>(j) * nbe + row;
        std::copy( C_col + rng.first, C_col + rng.second, dst );
      }
      row += rng.second - rng.first;
    }

    blas::gemm<double>( 'T', 'N', static_cast<int>(np), nmo_, nbe, 1.0,
      ao, nbe, scr.C_compressed.data(), nbe, 0.0, out_ + p0,
      static_cast<int>(ldo_) );
  }

};

/// Contraction of the AO batch against the density matrix:
/// rho(r) = sum_{mu,nu} D[mu,nu] phi_mu(r) phi_nu(r).
class DensityContractor {

  const detail::OrbitalEvaluatorImpl& impl_;
  const double* D_;
  size_t ldd_;
  double* out_;

public:

  struct Scratch {
    std::vector<double> dm_ao;
    std::vector<double> xmat_scr;
  };

  DensityContractor( const detail::OrbitalEvaluatorImpl& impl, const double* D,
                     size_t ldd, double* out ) :
    impl_(impl), D_(D), ldd_(ldd), out_(out) {}

  void init( Scratch& scr, size_t batch_size ) const {
    scr.dm_ao.resize( static_cast<size_t>(impl_.nbf_) * batch_size );
  }

  void zero( size_t p0, size_t np ) const {
    std::fill( out_ + p0, out_ + p0 + np, 0.0 );
  }

  void apply( Scratch& scr, size_t p0, size_t np, int32_t nbe,
              const std::vector<int32_t>& shells, const double* ao ) const {

    // eval_xmat needs nbe*nbe scratch. Growing it on demand inside the
    // parallel region keeps the high-water mark at the largest nbe this
    // thread actually saw (not nbf) and first-touches the pages on the
    // owning thread.
    const size_t scr_size = static_cast<size_t>(nbe) * nbe;
    if( scr.xmat_scr.size() < scr_size ) scr.xmat_scr.resize( scr_size );

    const int32_t nbf = impl_.nbf_;
    LocalHostWorkDriver::submat_map_t submat_map;
    std::tie( submat_map, std::ignore ) =
      gen_compressed_submat_map( *impl_.basis_map, shells, nbf, nbf );

    // dm_ao = D_compressed @ ao
    impl_.host_driver->eval_xmat( np, static_cast<size_t>(nbf),
      static_cast<size_t>(nbe), submat_map, 1.0, D_, ldd_,
      ao, static_cast<size_t>(nbe),
      scr.dm_ao.data(), static_cast<size_t>(nbe), scr.xmat_scr.data() );

    // rho[p] = sum_mu ao(mu, p) * dm_ao(mu, p)
    impl_.host_driver->eval_uvvar_lda_rks( np, static_cast<size_t>(nbe), ao,
      scr.dm_ao.data(), static_cast<size_t>(nbe), out_ + p0 );
  }

};

/** @brief The single batched evaluation loop shared by all entry points.
 *
 *  Batches of points are screened against the per-shell cutoff radii, the
 *  surviving shells are collocated into a per-thread AO buffer, and the
 *  result is handed to `contract` for the orbital- or density-specific
 *  reduction. All scratch is per-thread and per-call.
 */
template <typename PointSource, typename Contractor>
void batched_eval( const detail::OrbitalEvaluatorImpl& impl, size_t npts,
                   const PointSource& src, const Contractor& contract ) {

  const BasisSet<double>& basis = impl.basis;
  const auto& shell_cutoff_r2   = impl.shell_cutoff_r2;
  const int32_t nbf             = impl.nbf_;
  const int32_t nshells_total   = basis.nshells();

  const size_t batch_size = choose_batch_size( nbf, npts, omp_get_max_threads() );
  const int64_t n_batches =
    static_cast<int64_t>( (npts + batch_size - 1) / batch_size );

#pragma omp parallel
  {
    std::vector<double> pt_buf;
    if constexpr( PointSource::needs_scratch ) pt_buf.resize( 3 * batch_size );
    std::vector<double> ao_buf( static_cast<size_t>(nbf) * batch_size );
    std::vector<int32_t> screened_shells;
    screened_shells.reserve( nshells_total );
    typename Contractor::Scratch scr;
    contract.init( scr, batch_size );

#pragma omp for schedule(dynamic, 1)
    for( int64_t b = 0; b < n_batches; ++b ) {

      const size_t p0 = static_cast<size_t>(b) * batch_size;
      const size_t np = std::min( batch_size, npts - p0 );

      const double* pts = src.batch( p0, np, pt_buf.data() );

      // Per-batch shell screening: keep shells whose cutoff radius reaches
      // any point of this batch's bounding box.
      const PointBbox bbox = src.bbox( p0, np, pts );
      screened_shells.clear();
      int32_t nbe = 0;
      for( int32_t s = 0; s < nshells_total; ++s )
      if( dist2_center_to_bbox( basis[s].O_data(), bbox ) < shell_cutoff_r2[s] ) {
        screened_shells.push_back(s);
        nbe += basis[s].size();
      }

      // All shells screen out -> the result is identically zero here.
      if( not nbe ) { contract.zero( p0, np ); continue; }

      // Both contractions assume eval_collocation packs AO rows by cumulative
      // shell size in shell_list order. That holds for the gau2grid path; the
      // non-gau2grid fallback in gau2grid_collocation.cxx instead uses the
      // global shell_to_first_ao offset, which is only equivalent when no
      // shell is screened out. This is a pre-existing inconsistency in the
      // reference driver rather than one introduced here.
      impl.host_driver->eval_collocation( np, screened_shells.size(),
        static_cast<size_t>(nbe), pts, basis, screened_shells.data(),
        ao_buf.data() );

      contract.apply( scr, p0, np, nbe, screened_shells, ao_buf.data() );

    }
  }

}

void check_orbital_args( const std::string& ctx, size_t npts, int32_t nbf,
                         const double* C, size_t ldc, const double* out,
                         size_t ldo ) {
  if( not C or not out )
    GAUXC_GENERIC_EXCEPTION( ctx + ": null pointer argument." );
  if( ldc < static_cast<size_t>(nbf) )
    GAUXC_GENERIC_EXCEPTION( ctx + ": ldc must be >= nbf()." );
  if( ldo < npts )
    GAUXC_GENERIC_EXCEPTION( ctx + ": ldo must be >= npts." );
  // ldo is passed to blas::gemm, whose LDC parameter is int.
  if( ldo > static_cast<size_t>(std::numeric_limits<int>::max()) )
    GAUXC_GENERIC_EXCEPTION( ctx + ": ldo exceeds BLAS int range." );
}

void check_density_args( const std::string& ctx, int32_t nbf, const double* D,
                         size_t ldd, const double* out ) {
  if( not D or not out )
    GAUXC_GENERIC_EXCEPTION( ctx + ": null pointer argument." );
  if( ldd < static_cast<size_t>(nbf) )
    GAUXC_GENERIC_EXCEPTION( ctx + ": ldd must be >= nbf()." );
}

}  // namespace


OrbitalEvaluator::OrbitalEvaluator( pimpl_ptr_type&& pimpl ) :
  pimpl_( std::move(pimpl) ) {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
}

OrbitalEvaluator::~OrbitalEvaluator() noexcept = default;
OrbitalEvaluator::OrbitalEvaluator( OrbitalEvaluator&& ) noexcept = default;
OrbitalEvaluator& OrbitalEvaluator::operator=( OrbitalEvaluator&& ) noexcept =
  default;

int32_t OrbitalEvaluator::nbf() const { return pimpl_->nbf_; }

const BasisSet<double>& OrbitalEvaluator::basis() const {
  return pimpl_->basis;
}


OrbitalEvaluator OrbitalEvaluatorFactory::make_orbital_evaluator(
  ExecutionSpace ex, BasisSet<double> basis ) {

  switch(ex) {

  case ExecutionSpace::Host:
    return OrbitalEvaluator(
      std::make_unique<detail::OrbitalEvaluatorImpl>( std::move(basis) )
    );

  default:
    GAUXC_GENERIC_EXCEPTION("OrbitalEvaluator: only ExecutionSpace::Host is "
      "currently supported.");

  }

}


// ---------------------------------------------------------------------------
// Caller-supplied point sets
// ---------------------------------------------------------------------------

void OrbitalEvaluator::eval_orbital( size_t npts, const double* points,
                                     const double* C, double* out ) const {
  eval_orbitals( npts, points, /*nmo=*/1, C,
    /*ldc=*/static_cast<size_t>(pimpl_->nbf_), out, /*ldo=*/npts );
}

void OrbitalEvaluator::eval_orbitals( size_t npts, const double* points,
                                      int32_t nmo, const double* C, size_t ldc,
                                      double* out, size_t ldo ) const {
  if( not npts or nmo < 1 ) return;
  if( not points )
    GAUXC_GENERIC_EXCEPTION(
      "OrbitalEvaluator::eval_orbitals: null pointer argument.");
  check_orbital_args( "OrbitalEvaluator::eval_orbitals", npts, pimpl_->nbf_, C,
    ldc, out, ldo );

  batched_eval( *pimpl_, npts, RawPointSource{points},
    OrbitalContractor( *pimpl_, nmo, C, ldc, out, ldo ) );
}

void OrbitalEvaluator::eval_density( size_t npts, const double* points,
                                     const double* D, size_t ldd,
                                     double* out ) const {
  if( not npts ) return;
  if( not points )
    GAUXC_GENERIC_EXCEPTION(
      "OrbitalEvaluator::eval_density: null pointer argument.");
  check_density_args( "OrbitalEvaluator::eval_density", pimpl_->nbf_, D, ldd,
    out );

  batched_eval( *pimpl_, npts, RawPointSource{points},
    DensityContractor( *pimpl_, D, ldd, out ) );
}


// ---------------------------------------------------------------------------
// CubeGrid overloads: generate per-batch point coordinates on-the-fly,
// avoiding the 3*num_points()*8 byte temporary coordinate array.
// ---------------------------------------------------------------------------

void OrbitalEvaluator::eval_orbital( const CubeGrid& grid, const double* C,
                                     double* out ) const {
  eval_orbitals( grid, /*nmo=*/1, C,
    /*ldc=*/static_cast<size_t>(pimpl_->nbf_), out,
    /*ldo=*/static_cast<size_t>(grid.num_points()) );
}

void OrbitalEvaluator::eval_orbitals( const CubeGrid& grid, int32_t nmo,
                                      const double* C, size_t ldc, double* out,
                                      size_t ldo ) const {
  if( grid.num_points() <= 0 or nmo < 1 ) return;
  const size_t npts = static_cast<size_t>( grid.num_points() );
  check_orbital_args( "OrbitalEvaluator::eval_orbitals(grid)", npts,
    pimpl_->nbf_, C, ldc, out, ldo );

  batched_eval( *pimpl_, npts, GridPointSource{&grid},
    OrbitalContractor( *pimpl_, nmo, C, ldc, out, ldo ) );
}

void OrbitalEvaluator::eval_density( const CubeGrid& grid, const double* D,
                                     size_t ldd, double* out ) const {
  if( grid.num_points() <= 0 ) return;
  const size_t npts = static_cast<size_t>( grid.num_points() );
  check_density_args( "OrbitalEvaluator::eval_density(grid)", pimpl_->nbf_, D,
    ldd, out );

  batched_eval( *pimpl_, npts, GridPointSource{&grid},
    DensityContractor( *pimpl_, D, ldd, out ) );
}

}  // namespace GauXC
