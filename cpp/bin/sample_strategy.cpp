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
#include "strategy.h"

using namespace qr;

int main(int argc, char* argv[]) {
    auto print_help = [&]() {
        std::cout << "Usage: " << argv[0] << " <ticker> [options]\n";
        std::cout << "Options:\n";
        std::cout << "  --seed <seed>              Random seed\n";
        std::cout << "  --alpha-threshold <val>    Min |alpha| to trade (default: 0.5)\n";
        std::cout << "  --q-max <val>              Max inventory (default: 10)\n";
        std::cout << "  --trade-size <val>         Max trade size (default: 5)\n";
        std::cout << "  --cooldown <val>           Min events between trades (default: 10)\n";
        std::cout << "  --race / --no-race         Enable/disable race (default: no-race)\n";
        std::cout << "  --k <scale>                Alpha scale factor (default: 1.0)\n";
        std::cout << "  --theta <val>              Alpha consumption (default: 0.0)\n";
        std::cout << "  --duration <hours>         Simulation length (default: 1000)\n";
        std::cout << "  --mix                      Use mixture delta_t distribution\n";
        std::cout << "  --weibull                  Use Weibull inter-racer delays (default)\n";
        std::cout << "  --gamma                    Use Gamma inter-racer delays\n";
        std::cout << "  --use-total-lvl            Use 3D event probabilities\n";
        std::cout << "  -h, --help                 Show this help message\n";
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
    bool use_weibull = true;
    bool use_total_lvl = false;
    double alpha_threshold = 0.5;
    int32_t q_max = 10;
    int32_t trade_size = 5;
    int32_t cooldown = 10;
    double alpha_scale = 1.0;
    double theta = 0.0;
    int duration_hours = 1000;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--seed" && i + 1 < argc) {
            master_seed = std::stoull(argv[++i]);
        } else if (arg == "--mix") {
            use_mixture = true;
        } else if (arg == "--race") {
            use_race = true;
        } else if (arg == "--no-race") {
            use_race = false;
        } else if (arg == "--weibull") {
            use_weibull = true;
        } else if (arg == "--gamma") {
            use_weibull = false;
        } else if (arg == "--use-total-lvl") {
            use_total_lvl = true;
        } else if (arg == "--alpha-threshold" && i + 1 < argc) {
            alpha_threshold = std::stod(argv[++i]);
        } else if (arg == "--q-max" && i + 1 < argc) {
            q_max = std::stoi(argv[++i]);
        } else if (arg == "--trade-size" && i + 1 < argc) {
            trade_size = std::stoi(argv[++i]);
        } else if (arg == "--cooldown" && i + 1 < argc) {
            cooldown = std::stoi(argv[++i]);
        } else if (arg == "--k" && i + 1 < argc) {
            alpha_scale = std::stod(argv[++i]);
        } else if (arg == "--theta" && i + 1 < argc) {
            theta = std::stod(argv[++i]);
        } else if (arg == "--duration" && i + 1 < argc) {
            duration_hours = std::stoi(argv[++i]);
        }
    }

    std::cout << "Ticker: " << ticker << "\n";
    std::cout << "Alpha threshold: " << alpha_threshold << "\n";
    std::cout << "Q max: " << q_max << "\n";
    std::cout << "Trade size: " << trade_size << "\n";
    std::cout << "Cooldown: " << cooldown << " events\n";
    std::cout << "Race: " << (use_race ? "on" : "off") << "\n";
    std::cout << "Alpha scale: " << alpha_scale << "\n";
    std::cout << "Theta: " << theta << "\n";
    std::cout << "Duration: " << duration_hours << " hours\n";
    std::cout << "Master seed: " << master_seed << "\n";

    // Build output paths
    std::string race_suffix = use_race ? "_race" : "_norace";
    std::string base_name = "aggressive_thresh" + std::to_string(alpha_threshold).substr(0, 4) +
                            "_qmax" + std::to_string(q_max) + race_suffix;
    std::string lob_output = results_path + base_name + "_lob.parquet";
    std::string trades_output = results_path + base_name + "_trades.parquet";

    // Generate seeds for each component
    std::mt19937_64 seed_rng(master_seed);
    uint64_t lob_seed = seed_rng();
    uint64_t model_seed = seed_rng();
    uint64_t alpha_seed = seed_rng();

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

    // Create model
    std::unique_ptr<QRModel> model_ptr;
    if (delta_t_ptr) {
        model_ptr = std::make_unique<QRModel>(&lob, params, size_dists, *delta_t_ptr, model_seed);
    } else {
        model_ptr = std::make_unique<QRModel>(&lob, params, size_dists, model_seed);
    }
    QRModel& model = *model_ptr;

    int64_t duration = static_cast<int64_t>(1e9) * 3600 * duration_hours;

    // Create strategy
    StrategyParams strat_params(alpha_threshold, q_max, trade_size, cooldown);
    AggressiveStrategy strategy(strat_params);

    // Create components
    EMAImpact impact(0.01, 0.5);
    OUAlpha alpha(1.0, 0.5, alpha_seed);

    // Create race only if enabled (nullptr for no race)
    std::unique_ptr<LogisticRace> race_ptr;
    if (use_race) {
        race_ptr = std::make_unique<LogisticRace>(data_path, use_weibull);
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto [lob_buffer, strat_buffer] = run_aggressive(lob, model, impact, race_ptr.get(),
                                                      alpha, strategy, duration, alpha_scale, theta);
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Simulation: " << elapsed.count() << " ms\n";
    std::cout << "LOB events: " << lob_buffer.num_events() << "\n";
    std::cout << "Strategy trades: " << strat_buffer.num_trades() << "\n";
    std::cout << "Strategy fill rate: " << strat_buffer.fill_rate() * 100 << "%\n";
    std::cout << "Final inventory: " << strategy.inventory() << "\n";
    std::cout << "Realized PnL: " << strategy.pnl() << "\n";

    std::filesystem::create_directories(results_path);

    auto start_save = std::chrono::high_resolution_clock::now();
    lob_buffer.save_parquet(lob_output);
    strat_buffer.save_parquet(trades_output);
    auto end_save = std::chrono::high_resolution_clock::now();

    auto save_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_save - start_save);
    std::cout << "Parquet save: " << save_time.count() << " ms\n";
    std::cout << "LOB output: " << lob_output << "\n";
    std::cout << "Trades output: " << trades_output << "\n";
}
