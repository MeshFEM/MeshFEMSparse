#include "make_cholesky_factorizer.hh"

#include "CholmodFactorizer.hh"
#include "CatamariFactorizer.hh"
#if MESHFEM_WITH_PARDISO || MESHFEM_WITH_MKL_PARDISO
#include "PardisoFactorizer.hh"
#endif
#if __APPLE__
#include "AccelerateFactorizer.hh"
#endif

namespace MeshFEM {

std::unique_ptr<CholeskyFactorizerBase> make_cholesky_factorizer(CholeskyProvider provider) {
    return make_cholesky_factorizer(provider, false);
}

std::unique_ptr<CholeskyFactorizerBase> make_cholesky_factorizer(CholeskyProvider provider, bool singlePrecision) {
    switch (provider) {
        case CholeskyProvider::CHOLMOD:
            if (singlePrecision) throw std::invalid_argument("CHOLMOD does not support singlePrecision");
            return std::make_unique<CholmodFactorizer>();
        case CholeskyProvider::PARDISO:
            if (singlePrecision) throw std::invalid_argument("Pardiso does not support singlePrecision");
#if MESHFEM_WITH_PARDISO || MESHFEM_WITH_MKL_PARDISO
            return std::make_unique<PardisoFactorizer>();
#else
            throw std::runtime_error("Compiled without Pardiso");
#endif
        case CholeskyProvider::Accelerate:
#if __APPLE__
            return std::make_unique<AccelerateFactorizer>(singlePrecision);
#else
            throw std::runtime_error("Compiled without Accelerate");
#endif
        case CholeskyProvider::Catamari:
        case CholeskyProvider::CatamariNesdis:
        case CholeskyProvider::CatamariNesdisParallel:
        case CholeskyProvider::CatamariNesdisReuse:
        case CholeskyProvider::CatamariMetis:
        case CholeskyProvider::CatamariLegacy:
        case CholeskyProvider::CatamariAMD:
        case CholeskyProvider::CatamariScotch:
        case CholeskyProvider::CatamariAdaptive:
#if MESHFEM_WITH_CATAMARI
            {
                bool legacy = provider == CholeskyProvider::CatamariLegacy;
                auto c = std::make_unique<CatamariFactorizer>(legacy, singlePrecision);
                // c->setUseLeftLooking(true);
                if (provider == CholeskyProvider::Catamari)
                    c->orderingMethod = CatamariFactorizer::OrderingMethod::Catamari;
                else if ((provider == CholeskyProvider::CatamariNesdis) || (provider == CholeskyProvider::CatamariLegacy))
                    c->orderingMethod = CatamariFactorizer::OrderingMethod::CholmodNesdis;
                else if (provider == CholeskyProvider::CatamariNesdisParallel || provider == CholeskyProvider::CatamariNesdisReuse) {
                    c->orderingMethod = CatamariFactorizer::OrderingMethod::CholmodNesdisParallel;
                    if (provider == CholeskyProvider::CatamariNesdisReuse) c->setTemporalReusePeriod(32);
                }
                else if (provider == CholeskyProvider::CatamariMetis)
                    c->orderingMethod = CatamariFactorizer::OrderingMethod::Metis;
                else if (provider == CholeskyProvider::CatamariAMD)
                    c->orderingMethod = CatamariFactorizer::OrderingMethod::AMD;
                else if (provider == CholeskyProvider::CatamariScotch)
                    c->orderingMethod = CatamariFactorizer::OrderingMethod::Scotch;
                else if (provider == CholeskyProvider::CatamariAdaptive)
                    c->orderingMethod = CatamariFactorizer::OrderingMethod::Adaptive;
                return c;
            }
#endif
            throw std::runtime_error("Compiled without Catamari");
        default:
            throw std::runtime_error("Unknown provider");
    }
}

} // namespace MeshFEM
