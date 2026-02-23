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
#include <functional>
#include <iomanip>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>
#include "orderbook.h"
#include "qr_model.h"
#include "simulation.h"

using namespace qr;
namespace rj = rapidjson;

// --- JSON helpers (same as sample.cpp) ---

double get_double(const rj::Value& obj, const char* key, double default_val) {
    if (obj.HasMember(key) && obj[key].IsNumber()) return obj[key].GetDouble();
    return default_val;
}

int get_int(const rj::Value& obj, const char* key, int default_val) {
    if (obj.HasMember(key) && obj[key].IsInt()) return obj[key].GetInt();
    return default_val;
}

uint64_t get_uint64(const rj::Value& obj, const char* key, uint64_t default_val) {
    if (obj.HasMember(key) && obj[key].IsUint64()) return obj[key].GetUint64();
    return default_val;
}

bool get_bool(const rj::Value& obj, const char* key, bool default_val) {
    if (obj.HasMember(key) && obj[key].IsBool()) return obj[key].GetBool();
    return default_val;
}

std::string get_string(const rj::Value& obj, const char* key, const std::string& default_val) {
    if (obj.HasMember(key) && obj[key].IsString()) return obj[key].GetString();
    return default_val;
}

std::string hash_config(const std::string& content) {
    size_t h = std::hash<std::string>{}(content);
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(8) << (h & 0xFFFFFFFF);
    return oss.str();
}

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void update_registry(const std::string& registry_path, const std::string& hash,
                     const std::string& config_content) {
    rj::Document registry;
    registry.SetObject();

    std::ifstream ifs(registry_path);
    if (ifs.is_open()) {
        rj::IStreamWrapper isw(ifs);
        registry.ParseStream(isw);
        ifs.close();
        if (registry.HasParseError() || !registry.IsObject()) {
            registry.SetObject();
        }
    }

    rj::Document config_doc;
    config_doc.Parse(config_content.c_str());

    rj::Value entry(rj::kObjectType);
    auto& alloc = registry.GetAllocator();

    rj::Value config_copy;
    config_copy.CopyFrom(config_doc, alloc);
    entry.AddMember("config", config_copy, alloc);

    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ts;
    ts << std::put_time(std::localtime(&t), "%Y-%m-%dT%H:%M:%S");
    entry.AddMember("timestamp", rj::Value(ts.str().c_str(), alloc), alloc);

    rj::Value key(hash.c_str(), alloc);
    if (registry.HasMember(hash.c_str())) {
        registry[hash.c_str()] = entry;
    } else {
        registry.AddMember(key, entry, alloc);
    }

    std::ofstream ofs(registry_path);
    rj::OStreamWrapper osw(ofs);
    rj::PrettyWriter<rj::OStreamWrapper> writer(osw);
    registry.Accept(writer);
    ofs << "\n";
}

// --- LOB configs (volumes only, prices hardcoded) ---

struct LOBConfig {
    std::array<int32_t, 4> bid_vols;
    std::array<int32_t, 4> ask_vols;
};

constexpr std::array<int32_t, 4> BID_PRICES = {99, 98, 97, 96};
constexpr std::array<int32_t, 4> ASK_PRICES = {100, 101, 102, 103};

std::vector<LOBConfig> load_lob_configs(const std::string& path) {
    std::vector<LOBConfig> configs;
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open " + path);
    }

    std::string line;
    std::getline(file, line);  // skip header: q_4,q_3,q_2,q_1,q_-1,q_-2,q_-3,q_-4

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        LOBConfig cfg;

        // Ask vols: q_4, q_3, q_2, q_1 (far to near)
        for (int i = 3; i >= 0; i--) {
            std::getline(ss, token, ',');
            cfg.ask_vols[i] = std::stoi(token);
        }
        // Bid vols: q_-1, q_-2, q_-3, q_-4 (near to far)
        for (int i = 0; i < 4; i++) {
            std::getline(ss, token, ',');
            cfg.bid_vols[i] = std::stoi(token);
        }

        configs.push_back(cfg);
    }
    return configs;
}

// --- Accumulator ---

struct Accumulator {
    std::vector<int64_t> grid;
    std::vector<double> mid_sum;
    std::vector<double> mid_sum_sq;
    std::vector<double> bias_sum;
    std::vector<double> bias_sum_sq;
    std::vector<double> vol_sum;
    std::mutex mtx;
    int count = 0;

    Accumulator(int64_t duration, int64_t step) {
        for (int64_t t = 0; t <= duration; t += step) {
            grid.push_back(t);
            mid_sum.push_back(0.0);
            mid_sum_sq.push_back(0.0);
            bias_sum.push_back(0.0);
            bias_sum_sq.push_back(0.0);
            vol_sum.push_back(0.0);
        }
    }

    void add(const Buffer& buffer, const std::vector<int32_t>& cum_vol) {
        if (buffer.records.empty()) return;

        double mid0 = (buffer.records[0].best_bid_price +
                       buffer.records[0].best_ask_price) / 2.0;

        std::vector<double> proj_mid(grid.size(), 0.0);
        std::vector<double> proj_bias(grid.size(), 0.0);
        std::vector<double> proj_vol(grid.size(), 0.0);

        size_t j = 0;
        for (size_t i = 0; i < grid.size(); i++) {
            while (j + 1 < buffer.records.size() &&
                   buffer.records[j + 1].timestamp <= grid[i]) {
                j++;
            }
            if (j < buffer.records.size()) {
                double mid = (buffer.records[j].best_bid_price +
                              buffer.records[j].best_ask_price) / 2.0;
                proj_mid[i] = mid - mid0;
                proj_bias[i] = buffer.records[j].bias;
                proj_vol[i] = cum_vol[j];
            }
        }

        std::lock_guard<std::mutex> lock(mtx);
        for (size_t i = 0; i < grid.size(); i++) {
            mid_sum[i] += proj_mid[i];
            mid_sum_sq[i] += proj_mid[i] * proj_mid[i];
            bias_sum[i] += proj_bias[i];
            bias_sum_sq[i] += proj_bias[i] * proj_bias[i];
            vol_sum[i] += proj_vol[i];
        }
        count++;
    }

    void save_csv(const std::string& path) {
        std::ofstream file(path);
        file << "timestamp,avg_mid_price_change,mid_price_change_se,avg_bias,bias_se,avg_meta_vol\n";
        for (size_t i = 0; i < grid.size(); i++) {
            double mid_mean = mid_sum[i] / count;
            double mid_var = mid_sum_sq[i] / count - mid_mean * mid_mean;
            double mid_se = std::sqrt(mid_var / count);

            double bias_mean = bias_sum[i] / count;
            double bias_var = bias_sum_sq[i] / count - bias_mean * bias_mean;
            double bias_se = std::sqrt(bias_var / count);

            double vol_mean = vol_sum[i] / count;

            file << grid[i] << "," << mid_mean << "," << mid_se << "," << bias_mean << "," << bias_se << "," << vol_mean << "\n";
        }
    }
};

// --- Metaorder builder ---

MetaOrder build_metaorder(int32_t total_vol, Side side, int32_t max_order_size, int64_t exec_duration_ns) {
    MetaOrder metaorder;
    metaorder.side = side;

    int num_orders = (total_vol + max_order_size - 1) / max_order_size;
    if (num_orders == 0) num_orders = 1;

    int64_t interval = (num_orders > 1) ? exec_duration_ns / (num_orders - 1) : 0;

    int32_t remaining = total_vol;
    for (int i = 0; i < num_orders; i++) {
        int64_t t = static_cast<int64_t>(i) * interval;
        int32_t size = std::min(remaining, max_order_size);
        metaorder.timestamps.push_back(t);
        metaorder.sizes.push_back(size);
        remaining -= size;
    }

    return metaorder;
}

// --- Per-simulation worker ---

struct SimConfig {
    std::string params_path;
    const QueueDistributions* dists;
    const DeltaT* delta_t;
    const std::vector<LOBConfig>* lob_configs;

    // Impact
    std::string impact_type;
    double impact_alpha;
    double impact_m;
    double half_life_sec;
    std::vector<double> pl_half_lives;
    std::vector<double> pl_weights;

    // Metaorder
    int32_t metaorder_vol;
    int32_t max_order_size;
    int64_t exec_duration_ns;

    // Simulation
    int64_t duration;
    bool use_total_lvl;
};

void run_and_accumulate(const SimConfig& cfg, uint64_t seed, Accumulator& acc) {
    std::mt19937_64 rng(seed);
    size_t config_idx = rng() % cfg.lob_configs->size();
    const LOBConfig& lob_cfg = (*cfg.lob_configs)[config_idx];

    OrderBook lob(*cfg.dists, 4, seed);
    lob.init(std::vector<int32_t>(BID_PRICES.begin(), BID_PRICES.end()),
             std::vector<int32_t>(lob_cfg.bid_vols.begin(), lob_cfg.bid_vols.end()),
             std::vector<int32_t>(ASK_PRICES.begin(), ASK_PRICES.end()),
             std::vector<int32_t>(lob_cfg.ask_vols.begin(), lob_cfg.ask_vols.end()));

    QRParams params(cfg.params_path);
    if (cfg.use_total_lvl) {
        params.load_total_lvl_quantiles(cfg.params_path + "/total_lvl_quantiles.csv");
        params.load_event_probabilities_3d(cfg.params_path + "/event_probabilities_3d.csv");
    }
    SizeDistributions size_dists(cfg.params_path + "/size_distrib.csv");

    std::unique_ptr<QRModel> model_ptr;
    if (cfg.delta_t) {
        model_ptr = std::make_unique<QRModel>(&lob, params, size_dists, *cfg.delta_t, seed);
    } else {
        model_ptr = std::make_unique<QRModel>(&lob, params, size_dists, seed);
    }
    QRModel& model = *model_ptr;

    // Impact model
    std::unique_ptr<MarketImpact> impact_ptr;
    if (cfg.impact_type == "time_decay") {
        impact_ptr = std::make_unique<TimeDecayImpact>(cfg.half_life_sec, cfg.impact_m);
    } else if (cfg.impact_type == "ema") {
        impact_ptr = std::make_unique<EMAImpact>(cfg.impact_alpha, cfg.impact_m);
    } else if (cfg.impact_type == "power_law") {
        impact_ptr = std::make_unique<PowerLawImpact>(cfg.pl_half_lives, cfg.pl_weights, cfg.impact_m);
    } else {
        impact_ptr = std::make_unique<NoImpact>();
    }
    MarketImpact& impact = *impact_ptr;

    MetaOrder metaorder = build_metaorder(cfg.metaorder_vol, Side::Ask, cfg.max_order_size, cfg.exec_duration_ns);

    Buffer buffer;
    std::vector<int32_t> cum_vol_vec;
    std::vector<Fill> fills;
    int64_t time = 0;
    size_t meta_i = 0;
    size_t meta_n = metaorder.timestamps.size();
    double current_bias = 0.0;
    int32_t cum_meta_vol = 0;
    Order order;
    bool is_meta = false;

    while (time < cfg.duration) {
        impact.step(time);
        current_bias = impact.bias_factor();
        model.bias(current_bias);

        order = model.sample_order(time);
        int64_t dt = model.sample_dt();
        if (meta_i < meta_n && time + dt >= metaorder.timestamps[meta_i]) {
            time = metaorder.timestamps[meta_i];
            int32_t price = (metaorder.side == Side::Bid)
                ? std::max(1, lob.best_bid() - 4)
                : lob.best_ask() + 4;
            order = Order(OrderType::Trade, metaorder.side, price, metaorder.sizes[meta_i], time);
            meta_i++;
            is_meta = true;
        }
        else {
            time += dt;
            order.ts = time;
            is_meta = false;
        }

        EventRecord record;
        record.record_lob(lob);

        if (order.type == OrderType::Trade) {
            fills.clear();
            lob.process(order, &fills);
            int32_t filled = 0;
            for (const auto& f : fills) filled += f.size;
            if (is_meta) cum_meta_vol += filled;
            if (filled > 0) impact.update(order.side, order.ts, filled);
        } else {
            lob.process(order);
        }

        record.record_order(order);
        record.bias = current_bias;

        buffer.records.push_back(record);
        cum_vol_vec.push_back(cum_meta_vol);
    }

    acc.add(buffer, cum_vol_vec);
}

// --- Main ---

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <config.json>\n";
        std::cout << R"(
Example config.json:
{
  "ticker": "PFE",
  "use_mixture": false,
  "use_total_lvl": false,
  "duration_min": 30,
  "num_sims": 100000,
  "grid_step_ms": 500,

  "impact": {"type": "no_impact"},

  "metaorder_pcts": [10.0, 5.0, 2.5],
  "hourly_vol": 1000,
  "max_order_size": 2
}
)";
        return 1;
    }

    // Read and hash config
    std::string config_content = read_file(argv[1]);
    std::string config_hash = hash_config(config_content);

    rj::Document doc;
    doc.Parse(config_content.c_str());
    if (doc.HasParseError()) {
        std::cerr << "JSON parse error at offset " << doc.GetErrorOffset() << "\n";
        return 1;
    }

    std::string base_path = "/home/labcmap/saad.souilmi/dev_cpp/qr/data";
    std::string ticker = get_string(doc, "ticker", "PFE");
    std::string data_path = base_path + "/" + ticker;
    std::string params_path = data_path + "/qr_params";
    std::string results_path = base_path + "/results/" + ticker + "/metaorder/";

    bool use_mixture = get_bool(doc, "use_mixture", false);
    bool use_total_lvl = get_bool(doc, "use_total_lvl", false);
    int duration_min = get_int(doc, "duration_min", 30);
    int num_sims = get_int(doc, "num_sims", 100'000);
    int64_t grid_step = static_cast<int64_t>(get_int(doc, "grid_step_ms", 500)) * 1'000'000;
    int32_t hourly_vol = get_int(doc, "hourly_vol", 1000);
    int32_t max_order_size = get_int(doc, "max_order_size", 2);
    int exec_min = get_int(doc, "exec_duration_min", 5);
    int64_t exec_duration_ns = static_cast<int64_t>(exec_min) * 60 * 1'000'000'000LL;

    // Impact config
    std::string impact_type = "no_impact";
    double impact_alpha = 0.005;
    double impact_m = 4.5;
    double half_life_sec = 300.0;
    std::vector<double> pl_half_lives;
    std::vector<double> pl_weights;
    if (doc.HasMember("impact") && doc["impact"].IsObject()) {
        const auto& imp = doc["impact"];
        impact_type = get_string(imp, "type", "no_impact");
        impact_alpha = get_double(imp, "alpha", 0.005);
        impact_m = get_double(imp, "m", 4.5);
        half_life_sec = get_double(imp, "half_life_sec", 300.0);
        if (imp.HasMember("half_lives") && imp["half_lives"].IsArray()) {
            for (const auto& v : imp["half_lives"].GetArray())
                pl_half_lives.push_back(v.GetDouble());
        }
        if (imp.HasMember("weights") && imp["weights"].IsArray()) {
            for (const auto& v : imp["weights"].GetArray())
                pl_weights.push_back(v.GetDouble());
        }
    }

    // Metaorder percentages
    std::vector<double> metaorder_pcts = {10.0, 5.0, 2.5};
    if (doc.HasMember("metaorder_pcts") && doc["metaorder_pcts"].IsArray()) {
        metaorder_pcts.clear();
        for (const auto& v : doc["metaorder_pcts"].GetArray()) {
            metaorder_pcts.push_back(v.GetDouble());
        }
    }

    int64_t duration = static_cast<int64_t>(duration_min) * 60 * 1'000'000'000LL;
    int num_threads = std::thread::hardware_concurrency();

    // Print config
    std::cout << "Config: " << argv[1] << "\n";
    std::cout << "Hash: " << config_hash << "\n";
    std::cout << "Ticker: " << ticker << "\n";
    std::cout << "Delta_t: " << (use_mixture ? "mixture" : "exponential") << "\n";
    std::cout << "Total lvl: " << (use_total_lvl ? "on" : "off") << "\n";
    std::cout << "Impact: " << impact_type;
    if (impact_type == "ema") {
        std::cout << " (alpha=" << impact_alpha << ", m=" << impact_m << ")";
    } else if (impact_type == "time_decay") {
        std::cout << " (half_life=" << half_life_sec << "s, m=" << impact_m << ")";
    } else if (impact_type == "power_law") {
        std::cout << " (K=" << pl_half_lives.size() << " components, m=" << impact_m << ")";
    }
    std::cout << "\n";
    std::cout << "Duration: " << duration_min << " min, Metaorder execution: " << exec_min << " min\n";
    std::cout << "Hourly vol: " << hourly_vol << ", Max order size: " << max_order_size << "\n";
    std::cout << "Sims: " << num_sims << ", Threads: " << num_threads << "\n";
    std::cout << "Grid: " << (duration / grid_step + 1) << " points (" << (grid_step / 1'000'000) << "ms spacing)\n";

    // Load shared data
    QueueDistributions dists(params_path + "/invariant_distributions_qmax50.csv");

    std::unique_ptr<MixtureDeltaT> delta_t_ptr;
    if (use_mixture) {
        delta_t_ptr = std::make_unique<MixtureDeltaT>(params_path + "/delta_t_gmm.csv");
    }
    const DeltaT* delta_t = delta_t_ptr.get();

    std::vector<LOBConfig> lob_configs = load_lob_configs(params_path + "/random_lob.csv");
    std::cout << "Loaded " << lob_configs.size() << " LOB configs\n";

    std::filesystem::create_directories(results_path);

    // Build SimConfig (shared across all workers)
    SimConfig sim_cfg;
    sim_cfg.params_path = params_path;
    sim_cfg.dists = &dists;
    sim_cfg.delta_t = delta_t;
    sim_cfg.lob_configs = &lob_configs;
    sim_cfg.impact_type = impact_type;
    sim_cfg.impact_alpha = impact_alpha;
    sim_cfg.impact_m = impact_m;
    sim_cfg.half_life_sec = half_life_sec;
    sim_cfg.pl_half_lives = pl_half_lives;
    sim_cfg.pl_weights = pl_weights;
    sim_cfg.max_order_size = max_order_size;
    sim_cfg.exec_duration_ns = exec_duration_ns;
    sim_cfg.duration = duration;
    sim_cfg.use_total_lvl = use_total_lvl;

    auto total_start = std::chrono::high_resolution_clock::now();

    for (double pct : metaorder_pcts) {
        int32_t metaorder_vol = static_cast<int32_t>(hourly_vol * pct / 100.0);
        sim_cfg.metaorder_vol = metaorder_vol;

        std::cout << "\n=== Metaorder " << pct << "% of hourly vol (" << metaorder_vol << " shares) ===\n";

        MetaOrder sample_meta = build_metaorder(metaorder_vol, Side::Ask, max_order_size, exec_duration_ns);
        std::cout << "  Orders: " << sample_meta.timestamps.size() << ", sizes: ";
        for (size_t i = 0; i < std::min(sample_meta.sizes.size(), size_t(5)); i++) {
            std::cout << sample_meta.sizes[i] << " ";
        }
        if (sample_meta.sizes.size() > 5) std::cout << "...";
        std::cout << "\n";

        Accumulator acc(duration, grid_step);

        auto start = std::chrono::high_resolution_clock::now();

        for (int batch_start = 0; batch_start < num_sims; batch_start += num_threads) {
            int batch_end = std::min(batch_start + num_threads, num_sims);

            std::vector<std::future<void>> futures;
            for (int i = batch_start; i < batch_end; i++) {
                futures.push_back(std::async(std::launch::async,
                    run_and_accumulate, std::cref(sim_cfg), static_cast<uint64_t>(i), std::ref(acc)));
            }
            for (auto& f : futures) f.get();

            if (batch_start % 10000 == 0) {
                std::cout << "  Progress: " << batch_start << "/" << num_sims << "\n";
            }
        }

        std::string out_path = results_path + config_hash + "_pct_" + std::to_string(static_cast<int>(pct * 10)) + ".csv";
        acc.save_csv(out_path);
        std::cout << "  Output: " << out_path << "\n";

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "  Done in " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s\n";
    }

    // Update registry
    std::string registry_path = results_path + "registry.json";
    update_registry(registry_path, config_hash, config_content);

    auto total_end = std::chrono::high_resolution_clock::now();
    std::cout << "\nAll done in " << std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start).count() << "s\n";
    std::cout << "Registry: " << registry_path << "\n";
}
