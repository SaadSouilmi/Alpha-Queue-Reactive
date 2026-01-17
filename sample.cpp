#include <iostream>
#include <chrono>
#include <random>
#include <string>
#include <memory>
#include <filesystem>
#include "orderbook.h"
#include "qr_model.h"
#include "simulation.h"
#include "race.h"

using namespace qr;

int main(int argc, char* argv[]) {
    auto print_help = [&]() {
        std::cout << "Usage: " << argv[0] << " <ticker> [options]\n";
        std::cout << "Options:\n";
        std::cout << "  --seed <seed>    Random seed\n";
        std::cout << "  --mix            Use mixture delta_t distribution\n";
        std::cout << "  --race           Enable race mechanism\n";
        std::cout << "  --impact <name>  Impact model: ema_impact or no_impact (default: no_impact)\n";
        std::cout << "  --weibull        Use Weibull inter-racer delays (default)\n";
        std::cout << "  --gamma          Use Gamma inter-racer delays\n";
        std::cout << "  --use-total-lvl  Use 3D event probabilities (imb, spread, total_lvl)\n";
        std::cout << "  -h, --help       Show this help message\n";
    };

    if (argc < 2) {
        print_help();
        return 1;
    }

    if (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        print_help();
        return 0;
    }

    std::string base_path = "/home/labcmap/saad.souilmi/dev_cpp/qr/data";
    std::string ticker = argv[1];
    std::string data_path = base_path + "/" + ticker;
    std::string results_path = base_path + "/results/" + ticker + "/";

    // Parse flags
    uint64_t master_seed = std::random_device{}();
    bool use_mixture = false;
    bool use_race = false;
    bool use_weibull = true;  // default to weibull
    bool use_total_lvl = false;
    std::string impact_name = "no_impact";

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--seed" && i + 1 < argc) {
            master_seed = std::stoull(argv[++i]);
        } else if (arg == "--mix") {
            use_mixture = true;
        } else if (arg == "--race") {
            use_race = true;
        } else if (arg == "--impact" && i + 1 < argc) {
            impact_name = argv[++i];
        } else if (arg == "--weibull") {
            use_weibull = true;
        } else if (arg == "--gamma") {
            use_weibull = false;
        } else if (arg == "--use-total-lvl") {
            use_total_lvl = true;
        }
    }

    bool use_ema_impact = (impact_name == "ema_impact");

    std::cout << "Using ticker: " << ticker << "\n";

    // Build output path with appropriate suffixes
    std::string total_lvl_suffix = use_total_lvl ? "_totallvl" : "";
    std::string output_path = results_path + "result" + total_lvl_suffix + ".parquet";

    // Generate seeds for each component
    std::mt19937_64 seed_rng(master_seed);
    uint64_t lob_seed = seed_rng();
    uint64_t model_seed = seed_rng();
    uint64_t alpha_seed = seed_rng();

    std::cout << "Master seed: " << master_seed << "\n";
    std::cout << "Delta_t: " << (use_mixture ? "mixture" : "exponential") << "\n";
    std::cout << "Total lvl: " << (use_total_lvl ? "on" : "off") << "\n";
    if (use_race) {
        std::cout << "Race: " << (use_weibull ? "weibull" : "gamma") << "\n";
    }

    // Load queue distributions
    QueueDistributions dists(data_path + "/invariant_distributions_qmax50.csv");

    // Load delta_t if using mixture (floored version for race, regular for no-race)
    std::unique_ptr<MixtureDeltaT> delta_t_ptr;
    if (use_mixture) {
        std::string delta_t_file = use_race ? "/delta_t_mixtures_floored.csv" : "/delta_t_mixtures.csv";
        delta_t_ptr = std::make_unique<MixtureDeltaT>(data_path + delta_t_file);
    }

    // Initialize order book
    OrderBook lob(dists, 4, lob_seed);
    lob.init({1516, 1517, 1518, 1519},
              {4, 1, 10, 5},
              {1520, 1521, 1522, 1523},
              {6, 17, 22, 23});

    QRParams params(data_path);
    if (use_total_lvl) {
        params.load_total_lvl_quantiles(data_path + "/total_lvl_quantiles.csv");
        params.load_event_probabilities_3d(data_path + "/event_probabilities_3d.csv");
    }
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

    // Create optional components (race requires alpha, impact is independent)
    std::unique_ptr<OUAlpha> alpha_ptr;
    std::unique_ptr<LogisticRace> race_ptr;
    std::unique_ptr<MarketImpact> impact_ptr;

    if (use_race) {
        alpha_ptr = std::make_unique<OUAlpha>(1.0, 0.5, alpha_seed);  // kappa=1/min, s=0.5
        race_ptr = std::make_unique<LogisticRace>(data_path, use_weibull);
    }
    if (use_ema_impact) {
        impact_ptr = std::make_unique<EMAImpact>(0.01, 0.5);
    }

    auto start = std::chrono::high_resolution_clock::now();
    Buffer result = run_simulation(lob, model, duration,
                                    alpha_ptr.get(), impact_ptr.get(), race_ptr.get());
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Simulation: " << elapsed.count() << " ms\n";
    std::cout << "Events: " << result.num_events() << "\n";

    std::filesystem::create_directories(results_path);
    auto start_save = std::chrono::high_resolution_clock::now();
    result.save_parquet(output_path);
    auto end_save = std::chrono::high_resolution_clock::now();

    auto save_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_save - start_save);
    std::cout << "Parquet save: " << save_time.count() << " ms\n";
    std::cout << "Output: " << output_path << "\n";
}
