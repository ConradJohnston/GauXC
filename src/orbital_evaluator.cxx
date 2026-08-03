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
#include <cmath>
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

OrbitalEvaluatorImpl::OrbitalEvaluatorImpl( BasisSet<double> bs,
                                            double screening_tolerance ) :
  basis( std::move(bs) ) {

  // Retune our own copy: the caller's basis, which an SCF setup may share,
  // is never touched.
  for( auto& sh : basis ) sh.set_shell_tolerance( screening_tolerance );

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
 *    1. Cache footprint. Each thread holds `nscratch` live blocks of
 *       nbf*batch doubles: the AO block, the collocation kernel's own
 *       transpose staging, and, for density, the D*AO product. They are
 *       sized to share ~1 MiB, a typical per-core L2, rather than allowing
 *       each block that much on its own.
 *    2. Load balance: enough batches that each thread gets several, i.e.
 *       batch <= npts / (4*nthreads).
 */
size_t choose_batch_size( int32_t nbf, size_t npts, int nthreads,
                          int nscratch ) {
  constexpr size_t kTargetScratchBytesPerThread = 1024 * 1024;
  constexpr size_t kMinBatch = 128;
  constexpr size_t kMaxBatch = 8192;

  size_t batch = kMaxBatch;
  if( nbf > 0 ) {
    batch = kTargetScratchBytesPerThread /
            ( sizeof(double) * static_cast<size_t>(nbf) *
              static_cast<size_t>(nscratch) );
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

/// Bbox of the contiguous CubeGrid index range [k0, k1]. Exact: with iz
/// fastest, a range crossing an ix boundary contains both iy=0 and iy=ny-1,
/// and one crossing an iy boundary contains both iz=0 and iz=nz-1.
PointBbox bbox_for_index_range( const CubeGrid& g, int64_t k0, int64_t k1 ) {
  const int64_t nz = g.nz, ny = g.ny;
  int64_t lo_idx[3], hi_idx[3];
  lo_idx[0] = k0 / (ny * nz);
  hi_idx[0] = k1 / (ny * nz);
  if( hi_idx[0] > lo_idx[0] ) {
    lo_idx[1] = 0;  hi_idx[1] = ny - 1;
    lo_idx[2] = 0;  hi_idx[2] = nz - 1;
  } else {
    lo_idx[1] = (k0 / nz) % ny;
    hi_idx[1] = (k1 / nz) % ny;
    if( hi_idx[1] > lo_idx[1] ) {
      lo_idx[2] = 0;  hi_idx[2] = nz - 1;
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

/** @brief The output indices covered by one batch.
 *
 *  `nruns` runs of `run_len` consecutive indices starting at run_off[i]. A
 *  contiguous batch is the single-run case; a grid tile contributes one run
 *  per (ix,iy) pair.
 */
struct BatchSpan {
  const size_t* run_off = nullptr;
  size_t nruns   = 0;
  size_t run_len = 0;
  size_t npts    = 0;
};

/// Batch source over a caller-supplied AoS coordinate array.
struct RawPointSource {
  static constexpr bool needs_scratch = false;
  const double* points;
  size_t npts_total;
  size_t batch_size;

  size_t num_batches() const {
    return ( npts_total + batch_size - 1 ) / batch_size;
  }
  size_t max_points() const { return batch_size; }

  const double* batch( size_t b, double*, std::vector<size_t>& runs,
                       BatchSpan& span, PointBbox& bbox ) const {
    const size_t p0 = b * batch_size;
    const size_t np = std::min( batch_size, npts_total - p0 );
    runs.assign( 1, p0 );
    span = BatchSpan{ runs.data(), 1, np, np };
    const double* pts = points + 3 * p0;
    bbox = compute_bbox( pts, np );
    return pts;
  }
};

/// Batch source that walks a CubeGrid in contiguous index ranges, generating
/// coordinates on the fly. One run per batch, so results land in `out`
/// without a scatter.
struct GridLinearSource {
  static constexpr bool needs_scratch = true;
  const CubeGrid* grid;
  size_t npts_total;
  size_t batch_size;

  size_t num_batches() const {
    return ( npts_total + batch_size - 1 ) / batch_size;
  }
  size_t max_points() const { return batch_size; }

  const double* batch( size_t b, double* scr, std::vector<size_t>& runs,
                       BatchSpan& span, PointBbox& bbox ) const {
    const CubeGrid& g = *grid;
    const size_t p0 = b * batch_size;
    const size_t np = std::min( batch_size, npts_total - p0 );

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

    runs.assign( 1, p0 );
    span = BatchSpan{ runs.data(), 1, np, np };
    bbox = bbox_for_index_range( g, static_cast<int64_t>(p0),
                                 static_cast<int64_t>(p0 + np) - 1 );
    return scr;
  }
};

/** @brief Batch source that walks a CubeGrid in spatially compact tiles.
 *
 *  A contiguous index range is a needle along z: as soon as it crosses a row
 *  boundary its bbox spans the whole z extent, so distant shells survive
 *  screening. A tile of the same point count has a far tighter bbox, which
 *  cuts nbe and therefore both the collocation (~nbe) and the density
 *  contraction (~nbe^2). The cost is that a tile's points are no longer
 *  contiguous in the output.
 */
struct GridTileSource {
  static constexpr bool needs_scratch = true;
  const CubeGrid* grid;
  int64_t tx, ty, tz;      ///< tile extent in grid points
  int64_t ntx, nty, ntz;   ///< tile counts per axis

  size_t num_batches() const {
    return static_cast<size_t>(ntx) * static_cast<size_t>(nty) *
           static_cast<size_t>(ntz);
  }
  size_t max_points() const {
    return static_cast<size_t>(tx) * static_cast<size_t>(ty) *
           static_cast<size_t>(tz);
  }

  const double* batch( size_t b, double* scr, std::vector<size_t>& runs,
                       BatchSpan& span, PointBbox& bbox ) const {
    const CubeGrid& g = *grid;
    const int64_t bi = static_cast<int64_t>(b);
    const int64_t ix0 = ( bi / (nty * ntz) ) * tx;
    const int64_t iy0 = ( ( bi / ntz ) % nty ) * ty;
    const int64_t iz0 = ( bi % ntz ) * tz;
    const int64_t ix1 = std::min( ix0 + tx, g.nx );
    const int64_t iy1 = std::min( iy0 + ty, g.ny );
    const int64_t iz1 = std::min( iz0 + tz, g.nz );
    const int64_t nzr = iz1 - iz0;

    runs.clear();
    size_t n = 0;
    for( int64_t ix = ix0; ix < ix1; ++ix ) {
      const double x = g.origin[0] + g.spacing[0] * static_cast<double>(ix);
      for( int64_t iy = iy0; iy < iy1; ++iy ) {
        const double y = g.origin[1] + g.spacing[1] * static_cast<double>(iy);
        runs.push_back(
          static_cast<size_t>( (ix * g.ny + iy) * g.nz + iz0 ) );
        for( int64_t iz = iz0; iz < iz1; ++iz ) {
          scr[3 * n + 0] = x;
          scr[3 * n + 1] = y;
          scr[3 * n + 2] =
            g.origin[2] + g.spacing[2] * static_cast<double>(iz);
          ++n;
        }
      }
    }
    span = BatchSpan{ runs.data(), runs.size(),
                      static_cast<size_t>(nzr), n };

    const int64_t lo_idx[3] = { ix0, iy0, iz0 };
    const int64_t hi_idx[3] = { ix1 - 1, iy1 - 1, iz1 - 1 };
    for( int k = 0; k < 3; ++k ) {
      const double a =
        g.origin[k] + g.spacing[k] * static_cast<double>(lo_idx[k]);
      const double c =
        g.origin[k] + g.spacing[k] * static_cast<double>(hi_idx[k]);
      bbox.lo[k] = std::min(a, c);
      bbox.hi[k] = std::max(a, c);
    }
    return scr;
  }
};

/// Tile extents holding roughly `target` points and as close to cubic in
/// physical space as the (generally anisotropic) grid spacing allows.
GridTileSource make_tile_source( const CubeGrid& g, size_t target ) {
  const int64_t n[3] = { g.nx, g.ny, g.nz };
  std::array<double, 3> s{};
  for( int k = 0; k < 3; ++k ) {
    s[k] = std::fabs( g.spacing[k] );
    if( !(s[k] > 0.0) ) s[k] = 1.0;  // degenerate axis: treat as unconstrained
  }

  std::array<int64_t, 3> t{ 1, 1, 1 };
  double scale = std::cbrt( static_cast<double>(target) * s[0] * s[1] * s[2] );
  for( int attempt = 0; attempt < 4; ++attempt ) {
    int64_t prod = 1;
    for( int k = 0; k < 3; ++k ) {
      t[k] = std::clamp<int64_t>(
        static_cast<int64_t>( std::llround( scale / s[k] ) ), 1, n[k] );
      prod *= t[k];
    }
    if( static_cast<size_t>(prod) <= target ) break;
    scale *= std::cbrt( static_cast<double>(target) /
                        static_cast<double>(prod) );
  }

  GridTileSource src;
  src.grid = &g;
  src.tx = t[0]; src.ty = t[1]; src.tz = t[2];
  src.ntx = (n[0] + t[0] - 1) / t[0];
  src.nty = (n[1] + t[1] - 1) / t[1];
  src.ntz = (n[2] + t[2] - 1) / t[2];
  return src;
}

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

  /// nbf*batch blocks this contraction keeps live, beyond the AO block.
  static constexpr int scratch_blocks = 0;

  struct Scratch {
    std::vector<double> C_compressed;
    std::vector<double> staging;
  };

  OrbitalContractor( const detail::OrbitalEvaluatorImpl& impl, int32_t nmo,
                     const double* C, size_t ldc, double* out, size_t ldo ) :
    impl_(impl), nmo_(nmo), C_(C), ldc_(ldc), out_(out), ldo_(ldo) {}

  void init( Scratch& scr, size_t ) const {
    scr.C_compressed.resize( static_cast<size_t>(impl_.nbf_) * nmo_ );
  }

  void zero( const BatchSpan& span ) const {
    for( int32_t j = 0; j < nmo_; ++j ) {
      double* out_col = out_ + static_cast<size_t>(j) * ldo_;
      for( size_t r = 0; r < span.nruns; ++r )
        std::fill( out_col + span.run_off[r],
                   out_col + span.run_off[r] + span.run_len, 0.0 );
    }
  }

  void apply( Scratch& scr, const BatchSpan& span, int32_t nbe,
              const std::vector<int32_t>& shells, const double* ao ) const {

    const size_t np = span.npts;

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

    if( span.nruns == 1 ) {
      blas::gemm<double>( 'T', 'N', static_cast<int>(np), nmo_, nbe, 1.0,
        ao, nbe, scr.C_compressed.data(), nbe, 0.0,
        out_ + span.run_off[0], static_cast<int>(ldo_) );
      return;
    }

    const size_t need = np * static_cast<size_t>(nmo_);
    if( scr.staging.size() < need ) scr.staging.resize( need );
    blas::gemm<double>( 'T', 'N', static_cast<int>(np), nmo_, nbe, 1.0,
      ao, nbe, scr.C_compressed.data(), nbe, 0.0,
      scr.staging.data(), static_cast<int>(np) );

    for( int32_t j = 0; j < nmo_; ++j ) {
      const double* src = scr.staging.data() + static_cast<size_t>(j) * np;
      double* out_col = out_ + static_cast<size_t>(j) * ldo_;
      for( size_t r = 0; r < span.nruns; ++r )
        std::copy( src + r * span.run_len, src + (r + 1) * span.run_len,
                   out_col + span.run_off[r] );
    }
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

  /// dm_ao is a second nbf*batch block alongside the AO block.
  static constexpr int scratch_blocks = 1;

  struct Scratch {
    std::vector<double> dm_ao;
    std::vector<double> xmat_scr;
    std::vector<double> staging;
  };

  DensityContractor( const detail::OrbitalEvaluatorImpl& impl, const double* D,
                     size_t ldd, double* out ) :
    impl_(impl), D_(D), ldd_(ldd), out_(out) {}

  void init( Scratch&, size_t ) const {}

  void zero( const BatchSpan& span ) const {
    for( size_t r = 0; r < span.nruns; ++r )
      std::fill( out_ + span.run_off[r],
                 out_ + span.run_off[r] + span.run_len, 0.0 );
  }

  void apply( Scratch& scr, const BatchSpan& span, int32_t nbe,
              const std::vector<int32_t>& shells, const double* ao ) const {

    const size_t np = span.npts;

    // eval_xmat needs nbe*nbe scratch and writes an (nbe,np) block. Growing
    // both on demand inside the parallel region keeps the high-water mark at
    // the largest nbe this thread actually saw (not nbf) and first-touches
    // the pages on the owning thread.
    const size_t scr_size = static_cast<size_t>(nbe) * nbe;
    if( scr.xmat_scr.size() < scr_size ) scr.xmat_scr.resize( scr_size );
    const size_t dm_ao_size = static_cast<size_t>(nbe) * np;
    if( scr.dm_ao.size() < dm_ao_size ) scr.dm_ao.resize( dm_ao_size );

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
    double* den = out_ + span.run_off[0];
    if( span.nruns > 1 ) {
      if( scr.staging.size() < np ) scr.staging.resize( np );
      den = scr.staging.data();
    }
    impl_.host_driver->eval_uvvar_lda_rks( np, static_cast<size_t>(nbe), ao,
      scr.dm_ao.data(), static_cast<size_t>(nbe), den );

    if( span.nruns > 1 ) {
      for( size_t r = 0; r < span.nruns; ++r )
        std::copy( den + r * span.run_len, den + (r + 1) * span.run_len,
                   out_ + span.run_off[r] );
    }
  }

};

/** @brief The single batched evaluation loop shared by all entry points.
 *
 *  Batches of points are screened against the per-shell cutoff radii, the
 *  surviving shells are collocated into a per-thread AO buffer, and the
 *  result is handed to `contract` for the orbital- or density-specific
 *  reduction. All scratch is per-thread and per-call.
 */
template <typename BatchSource, typename Contractor>
void batched_eval( const detail::OrbitalEvaluatorImpl& impl,
                   const BatchSource& src, const Contractor& contract ) {

  const BasisSet<double>& basis = impl.basis;
  const auto& shell_cutoff_r2   = impl.shell_cutoff_r2;
  const int32_t nshells_total   = basis.nshells();

  const size_t max_pts = src.max_points();
  const int64_t n_batches = static_cast<int64_t>( src.num_batches() );

#pragma omp parallel
  {
    std::vector<double> pt_buf;
    if constexpr( BatchSource::needs_scratch ) pt_buf.resize( 3 * max_pts );
    std::vector<size_t> run_storage;
    std::vector<double> ao_buf;
    std::vector<int32_t> screened_shells;
    screened_shells.reserve( nshells_total );
    typename Contractor::Scratch scr;
    contract.init( scr, max_pts );

#pragma omp for schedule(dynamic, 1)
    for( int64_t b = 0; b < n_batches; ++b ) {

      BatchSpan span;
      PointBbox bbox;
      const double* pts =
        src.batch( static_cast<size_t>(b), pt_buf.data(), run_storage, span,
                   bbox );
      const size_t np = span.npts;

      // Per-batch shell screening: keep shells whose cutoff radius reaches
      // any point of this batch's bounding box.
      screened_shells.clear();
      int32_t nbe = 0;
      for( int32_t s = 0; s < nshells_total; ++s )
      if( dist2_center_to_bbox( basis[s].O_data(), bbox ) < shell_cutoff_r2[s] ) {
        screened_shells.push_back(s);
        nbe += basis[s].size();
      }

      // All shells screen out -> the result is identically zero here.
      if( not nbe ) { contract.zero( span ); continue; }

      // Only the surviving nbe rows are written, so the AO block tracks the
      // largest nbe seen rather than the nbf worst case.
      const size_t ao_size = static_cast<size_t>(nbe) * np;
      if( ao_buf.size() < ao_size ) ao_buf.resize( ao_size );

      // Both contractions assume eval_collocation packs AO rows by cumulative
      // shell size in shell_list order. That holds for the gau2grid path; the
      // non-gau2grid fallback in gau2grid_collocation.cxx instead uses the
      // global shell_to_first_ao offset, which is only equivalent when no
      // shell is screened out. This is a pre-existing inconsistency in the
      // reference driver rather than one introduced here.
      impl.host_driver->eval_collocation( np, screened_shells.size(),
        static_cast<size_t>(nbe), pts, basis, screened_shells.data(),
        ao_buf.data() );

      contract.apply( scr, span, nbe, screened_shells, ao_buf.data() );

    }
  }

}

/** @brief Whether spatial tiling can improve screening for this grid.
 *
 *  A contiguous batch's bbox spans the entire z extent as soon as it crosses
 *  a row boundary. If that extent already lies within a shell cutoff radius,
 *  every shell reaching any part of the column reaches all of it, so a
 *  tighter box removes nothing and the scatter it costs is pure overhead.
 *  Tiling is therefore worthwhile only when the grid spans well beyond the
 *  interaction range along z. The factor of two is a margin, not a fit: at a
 *  ratio near one there is nothing to gain.
 */
bool should_tile( const detail::OrbitalEvaluatorImpl& impl, const CubeGrid& g ) {
  if( impl.shell_cutoff_r2.empty() ) return false;
  std::vector<double> r2( impl.shell_cutoff_r2 );
  const auto mid = r2.begin() + r2.size() / 2;
  std::nth_element( r2.begin(), mid, r2.end() );
  const double median_radius = std::sqrt( *mid );
  const double z_extent =
    std::fabs( g.spacing[2] ) * static_cast<double>( g.nz );
  return z_extent > 2.0 * median_radius;
}

/// Target points per batch for a given contraction, accounting for every
/// nbf*batch block it keeps live alongside the AO block.
template <typename Contractor>
size_t batch_target( int32_t nbf, size_t npts ) {
  constexpr int collocation_scratch_blocks = 1;
  const int nscratch =
    1 + collocation_scratch_blocks + Contractor::scratch_blocks;
  return choose_batch_size( nbf, npts, omp_get_max_threads(), nscratch );
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
  ExecutionSpace ex, BasisSet<double> basis, double screening_tolerance ) {

  switch(ex) {

  case ExecutionSpace::Host:
    return OrbitalEvaluator(
      std::make_unique<detail::OrbitalEvaluatorImpl>( std::move(basis),
                                                      screening_tolerance )
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

  batched_eval( *pimpl_,
    RawPointSource{ points, npts,
                    batch_target<OrbitalContractor>( pimpl_->nbf_, npts ) },
    OrbitalContractor( *pimpl_, nmo, C, ldc, out, ldo ) );
}

namespace {

/// Run a grid evaluation with whichever traversal screens better.
template <typename Contractor>
void eval_on_grid( const detail::OrbitalEvaluatorImpl& impl,
                   const CubeGrid& grid, size_t npts, size_t target,
                   const Contractor& contract ) {
  if( should_tile( impl, grid ) )
    batched_eval( impl, make_tile_source( grid, target ), contract );
  else
    batched_eval( impl, GridLinearSource{ &grid, npts, target }, contract );
}

}  // namespace

void OrbitalEvaluator::eval_density( size_t npts, const double* points,
                                     const double* D, size_t ldd,
                                     double* out ) const {
  if( not npts ) return;
  if( not points )
    GAUXC_GENERIC_EXCEPTION(
      "OrbitalEvaluator::eval_density: null pointer argument.");
  check_density_args( "OrbitalEvaluator::eval_density", pimpl_->nbf_, D, ldd,
    out );

  batched_eval( *pimpl_,
    RawPointSource{ points, npts,
                    batch_target<DensityContractor>( pimpl_->nbf_, npts ) },
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

  eval_on_grid( *pimpl_, grid, npts,
    batch_target<OrbitalContractor>( pimpl_->nbf_, npts ),
    OrbitalContractor( *pimpl_, nmo, C, ldc, out, ldo ) );
}

void OrbitalEvaluator::eval_density( const CubeGrid& grid, const double* D,
                                     size_t ldd, double* out ) const {
  if( grid.num_points() <= 0 ) return;
  const size_t npts = static_cast<size_t>( grid.num_points() );
  check_density_args( "OrbitalEvaluator::eval_density(grid)", pimpl_->nbf_, D,
    ldd, out );

  eval_on_grid( *pimpl_, grid, npts,
    batch_target<DensityContractor>( pimpl_->nbf_, npts ),
    DensityContractor( *pimpl_, D, ldd, out ) );
}

}  // namespace GauXC
