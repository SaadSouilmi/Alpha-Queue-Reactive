#include <iostream>
#include <chrono>
#include <string>
#include <memory>
#include <random>
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
        std::cout << "  --impact <name>  Impact model: ema_impact or no_impact (default: no_impact)\n";
        std::cout << "  --k <scale>      Alpha scale factor (default: 1.0)\n";
        std::cout << "  --theta <val>    Alpha consumption per race (default: 0.0)\n";
        std::cout << "  --seed <seed>    Random seed\n";
        std::cout << "  --race           Enable race mechanism\n";
        std::cout << "  --weibull        Use Weibull inter-racer delays (default)\n";
        std::cout << "  --gamma          Use Gamma inter-racer delays\n";
        std::cout << "  --use-total-lvl  Use 3D event probabilities (imb, spread, total_lvl)\n";
        std::cout << "  --old-flow       Use old run_with_race() flow (for testing)\n";
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

    // Parse flags: --impact [ema_impact|no_impact] --k [alpha_scale] --theta [theta] --seed [seed] --race
    std::string impact_name = "no_impact";
    double alpha_scale = 1.0;
    double theta = 0.0;  // Alpha consumption per race
    uint64_t master_seed = std::random_device{}();
    bool use_race = false;
    bool use_weibull = true;
    bool use_total_lvl = false;
    bool use_old_flow = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--impact" && i + 1 < argc) {
            impact_name = argv[++i];
        } else if (arg == "--k" && i + 1 < argc) {
            alpha_scale = std::stod(argv[++i]);
        } else if (arg == "--theta" && i + 1 < argc) {
            theta = std::stod(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            master_seed = std::stoull(argv[++i]);
        } else if (arg == "--race") {
            use_race = true;
        } else if (arg == "--weibull") {
            use_weibull = true;
        } else if (arg == "--gamma") {
            use_weibull = false;
        } else if (arg == "--use-total-lvl") {
            use_total_lvl = true;
        } else if (arg == "--old-flow") {
            use_old_flow = true;
        }
    }

    std::cout << "Using ticker: " << ticker << "\n";

    // Generate seeds for each component
    std::mt19937_64 seed_rng(master_seed);
    uint64_t lob_seed = seed_rng();
    uint64_t model_seed = seed_rng();
    uint64_t alpha_seed = seed_rng();

    bool use_ema_impact = (impact_name == "ema_impact");
    std::string k_str = std::to_string(alpha_scale);
    k_str = k_str.substr(0, k_str.find('.') + 2);  // trim to 1 decimal
    std::string theta_str = std::to_string(theta);
    theta_str = theta_str.substr(0, theta_str.find('.') + 3);  // trim to 2 decimals
    std::string race_str = use_race ? "_race" : "_norace";
    std::string theta_suffix = use_race ? "_theta" + theta_str : "";
    std::string total_lvl_suffix = use_total_lvl ? "_totallvl" : "";
    std::string output_path = results_path + "result_alpha_" + impact_name + "_k" + k_str + race_str + theta_suffix + total_lvl_suffix + ".parquet";

    std::cout << "Impact: " << impact_name << "\n";
    std::cout << "Alpha scale (k): " << alpha_scale << "\n";
    std::cout << "Theta: " << theta << "\n";
    std::cout << "Race: " << (use_race ? (use_weibull ? "weibull" : "gamma") : "off") << "\n";
    std::cout << "Total lvl: " << (use_total_lvl ? "on" : "off") << "\n";
    std::cout << "Old flow: " << (use_old_flow ? "yes" : "no") << "\n";
    std::cout << "Master seed: " << master_seed << "\n";

    // Load queue distributions
    QueueDistributions dists(data_path + "/invariant_distributions_qmax50.csv");

    // Initialize order book
    OrderBook lob(dists, 4, lob_seed);
    lob.init({14996, 14997, 14998, 14999},
              {4, 1, 10, 5},
              {15000, 15001, 15002, 15003},
              {6, 17, 22, 23});

    QRParams params(data_path);
    if (use_total_lvl) {
        params.load_total_lvl_quantiles(data_path + "/total_lvl_quantiles.csv");
        params.load_event_probabilities_3d(data_path + "/event_probabilities_3d.csv");
    }
    SizeDistributions size_dists(data_path + "/size_distrib.csv");
    std::string delta_t_file = use_race ? "/delta_t_mixtures_floored.csv" : "/delta_t_mixtures.csv";
    MixtureDeltaT delta_t(data_path + delta_t_file);
    QRModel model(&lob, params, size_dists, delta_t, model_seed);

    // Create OU alpha process
    // kappa = 0.5 min^-1 (~1.4 min half-life, flattens by 5 min)
    // s = 0.5 (stationary std dev)
    OUAlpha alpha(0.5, 0.5, alpha_seed);

    int64_t duration = 1e9 * 3600 * 1000;  // 1000 hours in nanoseconds

    // Create optional race and impact components
    std::unique_ptr<LogisticRace> race_ptr;
    std::unique_ptr<MarketImpact> impact_ptr;

    if (use_race) {
        race_ptr = std::make_unique<LogisticRace>(data_path, use_weibull);
    }
    if (use_ema_impact) {
        impact_ptr = std::make_unique<EMAImpact>(0.01, 4.0);
    }

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    Buffer result;
    if (use_old_flow && use_race) {
        // Old flow: run_with_race() requires non-null impact and race
        EMAImpact default_impact(0.01, 4.0);
        MarketImpact& impact_ref = impact_ptr ? *impact_ptr : default_impact;
        result = run_with_race(lob, model, impact_ref, *race_ptr, alpha, duration, alpha_scale, theta);
    } else {
        result = run_simulation(lob, model, duration,
                                &alpha, impact_ptr.get(), race_ptr.get());
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Simulation: " << elapsed.count() << " ms\n";
    std::cout << "Events: " << result.num_events() << "\n";

    // Compute alpha PnL
    std::vector<int64_t> lags_ns;
    for (int s = 30; s <= 30 * 60; s += 30) {
        lags_ns.push_back(static_cast<int64_t>(s) * 1'000'000'000);
    }

    auto start_pnl = std::chrono::high_resolution_clock::now();
    AlphaPnL pnl = compute_alpha_pnl(result, lags_ns);
    auto end_pnl = std::chrono::high_resolution_clock::now();

    auto pnl_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_pnl - start_pnl);
    std::cout << "Alpha PnL compute: " << pnl_time.count() << " ms\n";

    // Save results
    std::filesystem::create_directories(results_path);
    auto start_save = std::chrono::high_resolution_clock::now();
    result.save_parquet(output_path);

    std::string pnl_path = output_path.substr(0, output_path.find(".parquet")) + "_pnl.csv";
    pnl.save_csv(pnl_path);
    auto end_save = std::chrono::high_resolution_clock::now();

    auto save_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_save - start_save);
    std::cout << "Save: " << save_time.count() << " ms\n";
    std::cout << "Output: " << output_path << "\n";
    std::cout << "PnL: " << pnl_path << "\n";
}
