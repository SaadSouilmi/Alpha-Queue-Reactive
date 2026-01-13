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
    std::string output_path = results_path + "result_alpha_" + impact_name + "_k" + k_str + race_str + theta_suffix + ".parquet";

    std::cout << "Impact: " << impact_name << "\n";
    std::cout << "Alpha scale (k): " << alpha_scale << "\n";
    std::cout << "Theta: " << theta << "\n";
    std::cout << "Race: " << (use_race ? (use_weibull ? "weibull" : "gamma") : "off") << "\n";
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
    SizeDistributions size_dists(data_path + "/size_distrib.csv");
    std::string delta_t_file = use_race ? "/delta_t_mixtures_floored.csv" : "/delta_t_mixtures.csv";
    MixtureDeltaT delta_t(data_path + delta_t_file);
    QRModel model(&lob, params, size_dists, delta_t, model_seed);

    // Create OU alpha process
    // kappa = 0.5 min^-1 (~1.4 min half-life, flattens by 5 min)
    // s = 0.5 (stationary std dev)
    OUAlpha alpha(0.5, 0.5, alpha_seed);

    int64_t duration = 1e9 * 3600 * 1000;  // 1000 hours in nanoseconds

    // Run simulation with chosen impact and race setting
    Buffer result;
    auto start = std::chrono::high_resolution_clock::now();
    if (use_race) {
        LogisticRace race(data_path, use_weibull);
        if (use_ema_impact) {
            EMAImpact impact(0.01, 4.0);
            result = run_with_race(lob, model, impact, race, alpha, duration, alpha_scale, theta);
        } else {
            NoImpact impact;
            result = run_with_race(lob, model, impact, race, alpha, duration, alpha_scale, theta);
        }
    } else {
        if (use_ema_impact) {
            EMAImpact impact(0.01, 4.0);
            result = run_with_alpha(lob, model, impact, alpha, duration, alpha_scale);
        } else {
            NoImpact impact;
            result = run_with_alpha(lob, model, impact, alpha, duration, alpha_scale);
        }
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
