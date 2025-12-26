#include <iostream>
#include <chrono>
#include <random>
#include <string>
#include <memory>
#include "orderbook.h"
#include "qr_model.h"
#include "simulation.h"
#include "race.h"

using namespace qr;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <ticker> [options]\n";
        std::cerr << "Options: --seed <seed> --mix --race --weibull --gamma\n";
        return 1;
    }

    std::string base_path = "/home/labcmap/saad.souilmi/dev_cpp/qr/data";
    std::string ticker = argv[1];
    std::string data_path = base_path + "/" + ticker;
    std::string output_path = base_path + "/results/result.parquet";

    // Parse flags: --seed [seed], --mix, --race, --weibull, --gamma
    uint64_t master_seed = std::random_device{}();
    bool use_mixture = false;
    bool use_race = false;
    bool use_weibull = true;  // default to weibull

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--seed" && i + 1 < argc) {
            master_seed = std::stoull(argv[++i]);
        } else if (arg == "--mix") {
            use_mixture = true;
        } else if (arg == "--race") {
            use_race = true;
        } else if (arg == "--weibull") {
            use_weibull = true;
        } else if (arg == "--gamma") {
            use_weibull = false;
        }
    }

    std::cout << "Using ticker: " << ticker << "\n";

    // Generate seeds for each component
    std::mt19937_64 seed_rng(master_seed);
    uint64_t lob_seed = seed_rng();
    uint64_t model_seed = seed_rng();
    uint64_t alpha_seed = seed_rng();

    std::cout << "Master seed: " << master_seed << "\n";
    std::cout << "Delta_t: " << (use_mixture ? "mixture" : "exponential") << "\n";
    if (use_race) {
        std::cout << "Race: " << (use_weibull ? "weibull" : "gamma") << "\n";
    }

    // Load queue distributions
    QueueDistributions dists(data_path + "/inv_distributions_qmax30.csv");

    // Load delta_t if using mixture
    std::unique_ptr<MixtureDeltaT> delta_t_ptr;
    if (use_mixture) {
        delta_t_ptr = std::make_unique<MixtureDeltaT>(data_path + "/delta_t_mixtures.csv");
    }

    // Initialize order book
    OrderBook lob(dists, 4, lob_seed);
    lob.init({1516, 1517, 1518, 1519},
              {4, 1, 10, 5},
              {1520, 1521, 1522, 1523},
              {6, 17, 22, 23});

    QRParams params(data_path);
    SizeDistributions size_dists(data_path + "/size_distrib.csv");

    // Create model with or without delta_t
    std::unique_ptr<QRModel> model_ptr;
    if (delta_t_ptr) {
        model_ptr = std::make_unique<QRModel>(&lob, params, size_dists, *delta_t_ptr, model_seed);
    } else {
        model_ptr = std::make_unique<QRModel>(&lob, params, size_dists, model_seed);
    }
    QRModel& model = *model_ptr;

    int64_t duration = 1e9 * 3600 * 1000;  // 5.5 hours

    auto start = std::chrono::high_resolution_clock::now();
    Buffer result;
    if (use_race) {
        // Create race components
        EMAImpact impact(0.01, 0.5);  // alpha=0.01, m=0.5
        OUAlpha alpha(1.0, 0.5, alpha_seed);  // kappa=1/min, s=0.5
        LogisticRace race(data_path, use_weibull);
        result = run_with_race(lob, model, impact, race, alpha, duration);
    } else {
        result = run_simple(lob, model, duration);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Simulation: " << elapsed.count() << " ms\n";
    std::cout << "Events: " << result.num_events() << "\n";

    auto start_save = std::chrono::high_resolution_clock::now();
    result.save_parquet(output_path);
    auto end_save = std::chrono::high_resolution_clock::now();

    auto save_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_save - start_save);
    std::cout << "Parquet save: " << save_time.count() << " ms\n";
}
