#include <iostream>
#include <chrono>
#include <future>
#include <vector>
#include <array>
#include <filesystem>
#include <thread>
#include <mutex>
#include <fstream>
#include <sstream>
#include <random>
#include <memory>
#include "orderbook.h"
#include "qr_model.h"
#include "simulation.h"

using namespace qr;

// LOB configuration from sample_lob_10k.csv
struct LOBConfig {
    std::array<int32_t, 4> bid_prices;
    std::array<int32_t, 4> bid_vols;
    std::array<int32_t, 4> ask_prices;
    std::array<int32_t, 4> ask_vols;
};

std::vector<LOBConfig> load_lob_configs(const std::string& path) {
    std::vector<LOBConfig> configs;
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open " + path);
    }

    std::string line;
    std::getline(file, line);  // skip header

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        LOBConfig cfg;

        // Q_-4,Q_-3,Q_-2,Q_-1,Q_1,Q_2,Q_3,Q_4,P_-4,P_-3,P_-2,P_-1,P_1,P_2,P_3,P_4
        // Bid vols (Q_-4 to Q_-1) - stored in reverse order for levels -1,-2,-3,-4
        for (int i = 3; i >= 0; i--) {
            std::getline(ss, token, ',');
            cfg.bid_vols[i] = std::stoi(token);
        }
        // Ask vols (Q_1 to Q_4)
        for (int i = 0; i < 4; i++) {
            std::getline(ss, token, ',');
            cfg.ask_vols[i] = std::stoi(token);
        }
        // Bid prices (P_-4 to P_-1) - stored in reverse order
        for (int i = 3; i >= 0; i--) {
            std::getline(ss, token, ',');
            cfg.bid_prices[i] = std::stoi(token);
        }
        // Ask prices (P_1 to P_4)
        for (int i = 0; i < 4; i++) {
            std::getline(ss, token, ',');
            cfg.ask_prices[i] = std::stoi(token);
        }

        configs.push_back(cfg);
    }
    return configs;
}

// Cached trade indices per bin (spread=1 only, one trade per side per bin)
struct TradeIndices {
    std::array<size_t, QRParams::NUM_IMB_BINS> bid_idx;
    std::array<size_t, QRParams::NUM_IMB_BINS> ask_idx;

    void compute(const QRParams& params) {
        for (int bin = 0; bin < QRParams::NUM_IMB_BINS; bin++) {
            const auto& state = params.state_params[bin][0];  // spread=1 -> index 0
            for (size_t i = 0; i < state.events.size(); i++) {
                if (state.events[i].type == OrderType::Trade) {
                    if (state.events[i].side == Side::Bid)
                        bid_idx[bin] = i;
                    else
                        ask_idx[bin] = i;
                }
            }
        }
    }
};

// Record of trade probs at one timestep
struct TradeProbRecord {
    int64_t timestamp;
    std::array<double, QRParams::NUM_IMB_BINS> bid_prob;
    std::array<double, QRParams::NUM_IMB_BINS> ask_prob;
};

struct Accumulator {
    std::vector<int64_t> grid;
    std::vector<double> mid_sum;
    std::vector<double> mid_sum_sq;
    std::vector<double> bias_sum;
    std::vector<double> bias_sum_sq;
    std::vector<double> sign_mean_sum;
    // Trade prob sums per bin: [bin][grid_point]
    std::array<std::vector<double>, QRParams::NUM_IMB_BINS> bid_trade_prob_sum;
    std::array<std::vector<double>, QRParams::NUM_IMB_BINS> ask_trade_prob_sum;
    std::mutex mtx;
    int count = 0;

    Accumulator(int64_t duration, int64_t step) {
        for (int64_t t = 0; t <= duration; t += step) {
            grid.push_back(t);

            mid_sum.push_back(0.0);
            mid_sum_sq.push_back(0.0);
            bias_sum.push_back(0.0);
            bias_sum_sq.push_back(0.0);
            sign_mean_sum.push_back(0.0);
        }
        for (int bin = 0; bin < QRParams::NUM_IMB_BINS; bin++) {
            bid_trade_prob_sum[bin].resize(grid.size(), 0.0);
            ask_trade_prob_sum[bin].resize(grid.size(), 0.0);
        }
    }

    void add(const Buffer& buffer, const std::vector<TradeProbRecord>& prob_records) {
        if (buffer.records.empty()) return;

        double mid0 = (buffer.records[0].best_bid_price +
                       buffer.records[0].best_ask_price) / 2.0;

        std::vector<double> proj_mid(grid.size(), 0.0);
        std::vector<double> proj_bias(grid.size(), 0.0);
        std::vector<double> proj_sign_mean(grid.size(), 0.0);
        std::array<std::vector<double>, QRParams::NUM_IMB_BINS> proj_bid_prob;
        std::array<std::vector<double>, QRParams::NUM_IMB_BINS> proj_ask_prob;
        for (int bin = 0; bin < QRParams::NUM_IMB_BINS; bin++) {
            proj_bid_prob[bin].resize(grid.size(), 0.0);
            proj_ask_prob[bin].resize(grid.size(), 0.0);
        }

        size_t j = 0;
        size_t k = 0;  // index into prob_records
        for (size_t i = 0; i < grid.size(); i++) {
            while (j + 1 < buffer.records.size() &&
                   buffer.records[j + 1].timestamp <= grid[i]) {
                j++;
            }
            while (k + 1 < prob_records.size() &&
                   prob_records[k + 1].timestamp <= grid[i]) {
                k++;
            }
            if (j < buffer.records.size()) {
                double mid = (buffer.records[j].best_bid_price +
                              buffer.records[j].best_ask_price) / 2.0;
                proj_mid[i] = mid - mid0;
                proj_bias[i] = buffer.records[j].bias;
                proj_sign_mean[i] = buffer.records[j].trade_sign_mean;
            }
            if (k < prob_records.size()) {
                for (int bin = 0; bin < QRParams::NUM_IMB_BINS; bin++) {
                    proj_bid_prob[bin][i] = prob_records[k].bid_prob[bin];
                    proj_ask_prob[bin][i] = prob_records[k].ask_prob[bin];
                }
            }
        }

        std::lock_guard<std::mutex> lock(mtx);
        for (size_t i = 0; i < grid.size(); i++) {
            mid_sum[i] += proj_mid[i];
            mid_sum_sq[i] += proj_mid[i] * proj_mid[i];
            bias_sum[i] += proj_bias[i];
            bias_sum_sq[i] += proj_bias[i] * proj_bias[i];
            sign_mean_sum[i] += proj_sign_mean[i];
            for (int bin = 0; bin < QRParams::NUM_IMB_BINS; bin++) {
                bid_trade_prob_sum[bin][i] += proj_bid_prob[bin][i];
                ask_trade_prob_sum[bin][i] += proj_ask_prob[bin][i];
            }
        }
        count++;
    }

    void save_csv(const std::string& path) {
        std::ofstream file(path);
        file << "timestamp,avg_mid_price_change,mid_price_change_se,avg_bias,bias_se,avg_trade_sign_mean";
        for (int bin = 0; bin < QRParams::NUM_IMB_BINS; bin++) {
            file << ",bin_" << bin << "_bid_trade_prob,bin_" << bin << "_ask_trade_prob";
        }
        file << "\n";
        for (size_t i = 0; i < grid.size(); i++) {
            double mid_mean = mid_sum[i] / count;
            double mid_var = mid_sum_sq[i] / count - mid_mean * mid_mean;
            double mid_se = std::sqrt(mid_var / count);

            double bias_mean = bias_sum[i] / count;
            double bias_var = bias_sum_sq[i] / count - bias_mean * bias_mean;
            double bias_se = std::sqrt(bias_var / count);

            file << grid[i] << "," << mid_mean << "," << mid_se << "," << bias_mean << "," << bias_se << "," << (sign_mean_sum[i] / count);
            for (int bin = 0; bin < QRParams::NUM_IMB_BINS; bin++) {
                file << "," << (bid_trade_prob_sum[bin][i] / count) << "," << (ask_trade_prob_sum[bin][i] / count);
            }
            file << "\n";
        }
    }
};

// Constants
constexpr int32_t MAX_ORDER_SIZE = 5;
constexpr int64_t METAORDER_DURATION_NS = 5LL * 60 * 1'000'000'000;  // 5 minutes in nanoseconds

// Build metaorder: total_vol spread over 5 minutes, max size per order = 5
// First order at t=0, last order at t=5min
MetaOrder build_metaorder(int32_t total_vol, Side side) {
    MetaOrder metaorder;
    metaorder.side = side;

    // Calculate number of orders needed (ceil division)
    int num_orders = (total_vol + MAX_ORDER_SIZE - 1) / MAX_ORDER_SIZE;
    if (num_orders == 0) num_orders = 1;

    // Time interval between orders (so last order lands at exactly 5 min)
    // For n orders: t=0, t=interval, ..., t=(n-1)*interval = 5min
    int64_t interval = (num_orders > 1) ? METAORDER_DURATION_NS / (num_orders - 1) : 0;

    int32_t remaining = total_vol;
    for (int i = 0; i < num_orders; i++) {
        int64_t t = static_cast<int64_t>(i) * interval;
        int32_t size = std::min(remaining, MAX_ORDER_SIZE);
        metaorder.timestamps.push_back(t);
        metaorder.sizes.push_back(size);
        remaining -= size;
    }

    return metaorder;
}

void run_and_accumulate(const std::string& data_path, const QueueDistributions& dists,
                        const DeltaT* delta_t, const TradeIndices& trd_idx,
                        const std::vector<LOBConfig>& lob_configs,
                        double ema_alpha, double ema_m, bool no_impact,
                        uint64_t seed, int64_t duration, int32_t metaorder_vol,
                        bool use_total_lvl, Accumulator& acc) {
    // Pick random LOB config based on seed
    std::mt19937_64 rng(seed);
    size_t config_idx = rng() % lob_configs.size();
    const LOBConfig& cfg = lob_configs[config_idx];

    OrderBook lob(dists, 4, seed);
    lob.init(std::vector<int32_t>(cfg.bid_prices.begin(), cfg.bid_prices.end()),
             std::vector<int32_t>(cfg.bid_vols.begin(), cfg.bid_vols.end()),
             std::vector<int32_t>(cfg.ask_prices.begin(), cfg.ask_prices.end()),
             std::vector<int32_t>(cfg.ask_vols.begin(), cfg.ask_vols.end()));

    QRParams params(data_path);
    if (use_total_lvl) {
        params.load_total_lvl_quantiles(data_path + "/total_lvl_quantiles.csv");
        params.load_event_probabilities_3d(data_path + "/event_probabilities_3d.csv");
    }
    SizeDistributions size_dists(data_path + "/size_distrib.csv");

    // Create model with or without delta_t
    std::unique_ptr<QRModel> model_ptr;
    if (delta_t) {
        model_ptr = std::make_unique<QRModel>(&lob, params, size_dists, *delta_t, seed);
    } else {
        model_ptr = std::make_unique<QRModel>(&lob, params, size_dists, seed);
    }
    QRModel& model = *model_ptr;

    // Impact model
    std::unique_ptr<MarketImpact> impact_ptr;
    if (no_impact) {
        impact_ptr = std::make_unique<NoImpact>();
    } else {
        impact_ptr = std::make_unique<EMAImpact>(ema_alpha, ema_m);
    }
    MarketImpact& impact = *impact_ptr;

    // Build metaorder based on volume
    MetaOrder metaorder = build_metaorder(metaorder_vol, Side::Ask);

    // Inline run_metaorder to capture trade probs
    Buffer buffer;
    std::vector<TradeProbRecord> prob_records;
    int64_t time = 0;
    size_t meta_i = 0;
    size_t meta_n = metaorder.timestamps.size();
    double current_bias = 0.0;
    double sign_sum = 0.0;
    int trade_count = 0;
    double current_sign_mean = 0.0;

    Order order;
    while (time < duration) {
        current_bias = impact.bias_factor(time);
        model.bias(current_bias);

        // Capture trade probs for all bins after bias is applied
        TradeProbRecord prob_rec{};
        prob_rec.timestamp = time;
        for (int bin = 0; bin < QRParams::NUM_IMB_BINS; bin++) {
            auto& state = params.state_params[bin][0];  // spread=1 -> index 0
            state.bias(current_bias);
            prob_rec.bid_prob[bin] = state.probs[trd_idx.bid_idx[bin]] / state.total;
            prob_rec.ask_prob[bin] = state.probs[trd_idx.ask_idx[bin]] / state.total;
        }
        prob_records.push_back(prob_rec);

        int64_t dt = model.sample_dt();
        if (meta_i < meta_n && time + dt >= metaorder.timestamps[meta_i]) {
            time = metaorder.timestamps[meta_i];
            int32_t price = (metaorder.side == Side::Bid) ? lob.best_bid() : lob.best_ask();
            order = Order(OrderType::Trade, metaorder.side, price, metaorder.sizes[meta_i], time);
            meta_i++;
        }
        else {
            time += dt;
            order = model.sample_order(time);
        }

        EventRecord record;
        record.record_lob(lob);
        lob.process(order);
        record.record_order(order);
        record.bias = current_bias;

        if (order.type == OrderType::Trade) {
            impact.update(order.side, time);
            double sign = (order.side == Side::Bid) ? -1.0 : 1.0;
            sign_sum += sign;
            trade_count++;
            current_sign_mean = sign_sum / trade_count;
        }
        record.trade_sign_mean = current_sign_mean;

        buffer.records.push_back(record);
    }

    acc.add(buffer, prob_records);
}

int main(int argc, char* argv[]) {
    auto print_help = [&]() {
        std::cout << "Usage: " << argv[0] << " <ticker> [options]\n";
        std::cout << "Options:\n";
        std::cout << "  --mix              Use mixture delta_t distribution\n";
        std::cout << "  --alpha <val>      EMA impact alpha (default: 0.005)\n";
        std::cout << "  --m <val>          EMA impact multiplier (default: 4.5)\n";
        std::cout << "  --duration <min>   Simulation duration in minutes (default: 30)\n";
        std::cout << "  --hourly-vol <val> Hourly volume baseline (default: 1000)\n";
        std::cout << "  --no-impact        Disable EMA impact model\n";
        std::cout << "  --use-total-lvl    Use 3D event probabilities (imb, spread, total_lvl)\n";
        std::cout << "  -h, --help         Show this help message\n";
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
    std::string base_results_path = base_path + "/results/" + ticker;

    // Parse flags: --mix, --alpha, --m, --duration, --hourly-vol, --use-total-lvl
    bool use_mixture = false;
    double ema_alpha = 0.005;
    double ema_m = 4.5;
    int duration_min = 30;
    int32_t hourly_vol = 1000;
    bool use_total_lvl = false;
    bool no_impact = false;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mix") {
            use_mixture = true;
        } else if (arg == "--alpha" && i + 1 < argc) {
            ema_alpha = std::stod(argv[++i]);
        } else if (arg == "--m" && i + 1 < argc) {
            ema_m = std::stod(argv[++i]);
        } else if (arg == "--duration" && i + 1 < argc) {
            duration_min = std::stoi(argv[++i]);
        } else if (arg == "--hourly-vol" && i + 1 < argc) {
            hourly_vol = std::stoi(argv[++i]);
        } else if (arg == "--no-impact") {
            no_impact = true;
        } else if (arg == "--use-total-lvl") {
            use_total_lvl = true;
        }
    }

    std::cout << "Using ticker: " << ticker << "\n";

    // Load queue distributions once
    QueueDistributions dists(data_path + "/invariant_distributions_qmax50.csv");
    std::cout << "Loaded queue distributions\n";

    // Load delta_t based on flag
    std::unique_ptr<MixtureDeltaT> delta_t_ptr;
    if (use_mixture) {
        delta_t_ptr = std::make_unique<MixtureDeltaT>(data_path + "/delta_t_mixtures.csv");
        std::cout << "Using MixtureDeltaT\n";
    } else {
        std::cout << "Using exponential delta_t (intensities)\n";
    }
    const DeltaT* delta_t = delta_t_ptr.get();  // nullptr if exp

    // Load LOB configurations
    std::vector<LOBConfig> lob_configs = load_lob_configs(data_path + "/sample_lob_10k.csv");
    std::cout << "Loaded " << lob_configs.size() << " LOB configurations\n";

    // Compute trade indices once
    QRParams params_for_idx(data_path);
    TradeIndices trd_idx;
    trd_idx.compute(params_for_idx);
    std::cout << "Computed trade indices\n";

    int num_sims = 100'000;
    //int num_sims = 1;
    int num_threads = std::thread::hardware_concurrency();

    int64_t duration = static_cast<int64_t>(duration_min) * 60 * 1'000'000'000;
    int64_t step = 500'000'000;  // 500ms grid

    std::cout << "Using " << num_threads << " threads\n";
    std::cout << "Impact: " << (no_impact ? "off" : "EMA alpha=" + std::to_string(ema_alpha) + ", m=" + std::to_string(ema_m)) << "\n";
    std::cout << "Hourly vol: " << hourly_vol << "\n";
    std::cout << "Total lvl: " << (use_total_lvl ? "on" : "off") << "\n";
    std::cout << "Grid: " << (duration / step + 1) << " points (500ms spacing)\n";
    std::cout << "Duration: " << duration_min << " minutes, Metaorder execution: 5 minutes\n";

    std::filesystem::create_directories(base_results_path);

    // Metaorder sizes: 2.5%, 5%, 10% of hourly_vol
    std::vector<std::pair<double, int32_t>> metaorder_configs = {
        {2.5, static_cast<int32_t>(hourly_vol * 0.025)},
        {5.0, static_cast<int32_t>(hourly_vol * 0.05)},
        {10.0, static_cast<int32_t>(hourly_vol * 0.10)}
    };

    auto total_start = std::chrono::high_resolution_clock::now();

    for (const auto& [pct, metaorder_vol] : metaorder_configs) {
        std::cout << "\n=== Running metaorder " << pct << "% of hourly vol (" << metaorder_vol << " shares) ===\n";

        MetaOrder sample_meta = build_metaorder(metaorder_vol, Side::Ask);
        std::cout << "  Orders: " << sample_meta.timestamps.size()
                  << ", sizes: ";
        for (size_t i = 0; i < std::min(sample_meta.sizes.size(), size_t(5)); i++) {
            std::cout << sample_meta.sizes[i] << " ";
        }
        if (sample_meta.sizes.size() > 5) std::cout << "...";
        std::cout << "\n";

        Accumulator acc(duration, step);

        std::cout << "Starting (" << num_sims << " simulations)\n";
        auto start = std::chrono::high_resolution_clock::now();

        for (int batch_start = 0; batch_start < num_sims; batch_start += num_threads) {
            int batch_end = std::min(batch_start + num_threads, num_sims);

            std::vector<std::future<void>> futures;
            for (int i = batch_start; i < batch_end; i++) {
                futures.push_back(std::async(std::launch::async, run_and_accumulate,
                                              data_path, std::cref(dists), delta_t, std::cref(trd_idx), std::cref(lob_configs),
                                              ema_alpha, ema_m, no_impact, i, duration, metaorder_vol, use_total_lvl, std::ref(acc)));
            }

            for (auto& f : futures) {
                f.get();
            }

            if (batch_start % 10000 == 0) {
                std::cout << "  Progress: " << batch_start << "/" << num_sims << "\n";
            }
        }

        std::string dt_suffix = use_mixture ? "_mix" : "_exp";
        std::string total_lvl_suffix = use_total_lvl ? "_totallvl" : "";
        std::ostringstream oss;
        if (no_impact) {
            oss << base_results_path << "/no_impact_pct_" << static_cast<int>(pct * 10)
                << dt_suffix << total_lvl_suffix << ".csv";
        } else {
            oss << base_results_path << "/ema_impact_pct_" << static_cast<int>(pct * 10)
                << "_a" << ema_alpha << "_m" << ema_m << dt_suffix << total_lvl_suffix << ".csv";
        }
        std::string out_path = oss.str();
        acc.save_csv(out_path);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << "Completed " << pct << "% in " << elapsed.count() << " seconds\n";
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start);
    std::cout << "\nAll done in " << total_elapsed.count() << " seconds\n";
}
