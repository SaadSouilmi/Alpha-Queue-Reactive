#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <string>
#include <memory>
#include <random>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>
#include "orderbook.h"
#include "qr_model.h"
#include "simulation.h"
#include "race.h"

using namespace qr;
namespace rj = rapidjson;

// Helper to get JSON value with default
double get_double(const rj::Value& obj, const char* key, double default_val) {
    if (obj.HasMember(key) && obj[key].IsNumber()) return obj[key].GetDouble();
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

// Hash config file content → 8-char hex string
std::string hash_config(const std::string& content) {
    size_t h = std::hash<std::string>{}(content);
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(8) << (h & 0xFFFFFFFF);
    return oss.str();
}

// Read entire file as string
std::string read_file(const std::string& path) {
    std::ifstream f(path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Update registry: load existing registry.json, add/overwrite entry, save
void update_registry(const std::string& registry_path, const std::string& hash,
                     const std::string& config_content, uint64_t seed_used) {
    rj::Document registry;
    registry.SetObject();

    // Load existing registry if it exists
    std::ifstream ifs(registry_path);
    if (ifs.is_open()) {
        rj::IStreamWrapper isw(ifs);
        registry.ParseStream(isw);
        ifs.close();
        if (registry.HasParseError() || !registry.IsObject()) {
            registry.SetObject();
        }
    }

    // Parse the config content into a Value
    rj::Document config_doc;
    config_doc.Parse(config_content.c_str());

    // Build entry: { "config": {...}, "seed_used": 12345, "timestamp": "..." }
    rj::Value entry(rj::kObjectType);
    auto& alloc = registry.GetAllocator();

    rj::Value config_copy;
    config_copy.CopyFrom(config_doc, alloc);
    entry.AddMember("config", config_copy, alloc);
    entry.AddMember("seed_used", seed_used, alloc);

    // Timestamp
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ts;
    ts << std::put_time(std::localtime(&t), "%Y-%m-%dT%H:%M:%S");
    entry.AddMember("timestamp", rj::Value(ts.str().c_str(), alloc), alloc);

    // Add/overwrite entry
    rj::Value key(hash.c_str(), alloc);
    if (registry.HasMember(hash.c_str())) {
        registry[hash.c_str()] = entry;
    } else {
        registry.AddMember(key, entry, alloc);
    }

    // Write registry
    std::ofstream ofs(registry_path);
    rj::OStreamWrapper osw(ofs);
    rj::PrettyWriter<rj::OStreamWrapper> writer(osw);
    registry.Accept(writer);
    ofs << "\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <config.json>\n";
        std::cout << R"(
Example config.json:
{
  "ticker": "AAPL",
  "alpha_scale": 1.0,
  "duration_hours": 1000,
  "seed": 12345,
  "use_weibull": true,
  "use_total_lvl": false,
  "impact": {"type": "no_impact"},

  "race": {
    "min_threshold": 0.8,
    "threshold": 0.7,
    "steepness": 8.0,
    "trade_prob": 0.65,
    "cancel_prob": 0.35,
    "base_mean_racers": 4.0,
    "racer_scale": 2.5,
    "mean_size": 3.0
  },

  "ou": {
    "kappa": 0.5,
    "sigma": 0.5
  }
}
)";
        return 1;
    }

    // Read and hash config
    std::string config_content = read_file(argv[1]);
    std::string config_hash = hash_config(config_content);

    // Parse JSON config
    rj::Document doc;
    doc.Parse(config_content.c_str());

    if (doc.HasParseError()) {
        std::cerr << "JSON parse error at offset " << doc.GetErrorOffset() << "\n";
        return 1;
    }

    std::string base_path = "/home/labcmap/saad.souilmi/dev_cpp/qr/data";
    std::string ticker = get_string(doc, "ticker", "AAPL");
    std::string data_path = base_path + "/" + ticker;
    std::string results_path = base_path + "/results/" + ticker + "/hft_alpha_results/";

    // General params
    double alpha_scale = get_double(doc, "alpha_scale", 1.0);
    uint64_t master_seed = get_uint64(doc, "seed", std::random_device{}());
    bool use_weibull = get_bool(doc, "use_weibull", true);
    bool use_total_lvl = get_bool(doc, "use_total_lvl", false);

    // Impact params - always expect {"type": "...", ...}
    std::string impact_type = "no_impact";
    rj::Value impact_cfg(rj::kObjectType);
    if (doc.HasMember("impact") && doc["impact"].IsObject()) {
        impact_cfg.CopyFrom(doc["impact"], doc.GetAllocator());
        impact_type = get_string(impact_cfg, "type", "no_impact");
    }

    // Race params
    RaceParams race_params;
    if (doc.HasMember("race") && doc["race"].IsObject()) {
        const auto& race = doc["race"];
        race_params.min_threshold = get_double(race, "min_threshold", 0.8);
        race_params.threshold = get_double(race, "threshold", 0.7);
        race_params.steepness = get_double(race, "steepness", 8.0);
        race_params.trade_prob = get_double(race, "trade_prob", 0.65);
        race_params.cancel_prob = get_double(race, "cancel_prob", 0.35);
        race_params.base_mean_racers = get_double(race, "base_mean_racers", 4.0);
        race_params.racer_scale = get_double(race, "racer_scale", 2.5);
        race_params.mean_size = get_double(race, "mean_size", 3.0);
    }

    // OU params
    double kappa = 0.5;
    double sigma = 0.5;
    double w_ou = 1.0;
    double w_imb = 0.0;
    if (doc.HasMember("ou") && doc["ou"].IsObject()) {
        const auto& ou = doc["ou"];
        kappa = get_double(ou, "kappa", 0.5);
        sigma = get_double(ou, "sigma", 0.5);
        w_ou = get_double(ou, "w_ou", 1.0);
        w_imb = get_double(ou, "w_imb", 0.0);
    }

    // Simulation duration (hours)
    double duration_hours = get_double(doc, "duration_hours", 1000.0);

    // Output path using hash
    std::string output_path = results_path + config_hash + ".parquet";
    std::string registry_path = results_path + "registry.json";

    // Print config
    std::cout << "Config: " << argv[1] << "\n";
    std::cout << "Hash: " << config_hash << "\n";
    std::cout << "Ticker: " << ticker << "\n";
    std::cout << "Alpha scale (k): " << alpha_scale << "\n";
    std::cout << "Impact: " << impact_type << "\n";
    std::cout << "Race params:\n";
    std::cout << "  min_threshold: " << race_params.min_threshold << "\n";
    std::cout << "  threshold: " << race_params.threshold << "\n";
    std::cout << "  steepness: " << race_params.steepness << "\n";
    std::cout << "  trade/cancel: " << race_params.trade_prob << "/"
              << race_params.cancel_prob << "\n";
    std::cout << "  base_mean_racers: " << race_params.base_mean_racers << "\n";
    std::cout << "  racer_scale: " << race_params.racer_scale << "\n";
    std::cout << "  mean_size: " << race_params.mean_size << "\n";
    std::cout << "OU params: kappa=" << kappa << ", sigma=" << sigma
              << ", w_ou=" << w_ou << ", w_imb=" << w_imb << "\n";
    std::cout << "Duration: " << duration_hours << " hours\n";
    std::cout << "Seed: " << master_seed << "\n";

    // Generate seeds for each component
    std::mt19937_64 seed_rng(master_seed);
    uint64_t lob_seed = seed_rng();
    uint64_t model_seed = seed_rng();
    uint64_t alpha_seed = seed_rng();

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
    std::string delta_t_file = "/delta_t_mixtures_floored.csv";
    MixtureDeltaT delta_t(data_path + delta_t_file);
    QRModel model(&lob, params, size_dists, delta_t, model_seed);

    // Create OU alpha process
    OUAlpha alpha(kappa, sigma, alpha_seed);

    int64_t duration = static_cast<int64_t>(duration_hours * 3600.0 * 1e9);

    // Create race
    LogisticRace race(data_path, use_weibull, race_params);

    // Create optional impact
    std::unique_ptr<MarketImpact> impact_ptr;
    if (impact_type == "time_decay") {
        double half_life = get_double(impact_cfg, "half_life_sec", 30.0);
        double m = get_double(impact_cfg, "m", 4.0);
        impact_ptr = std::make_unique<TimeDecayImpact>(half_life, m);
    } else if (impact_type == "ema") {
        double alpha = get_double(impact_cfg, "alpha", 0.01);
        double m = get_double(impact_cfg, "m", 4.0);
        impact_ptr = std::make_unique<EMAImpact>(alpha, m);
    } else if (impact_type != "no_impact") {
        std::cerr << "Unknown impact type: " << impact_type << "\n";
        return 1;
    }

    // Run HFT alpha simulation
    auto start = std::chrono::high_resolution_clock::now();
    Buffer result = run_hft_alpha(lob, model, duration, alpha, race, w_ou, w_imb, alpha_scale, impact_ptr.get());
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Simulation: " << elapsed.count() << " ms\n";
    std::cout << "Events: " << result.num_events() << "\n";

    // Save results
    std::filesystem::create_directories(results_path);
    auto start_save = std::chrono::high_resolution_clock::now();
    result.save_parquet(output_path);
    auto end_save = std::chrono::high_resolution_clock::now();

    auto save_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_save - start_save);
    std::cout << "Save: " << save_time.count() << " ms\n";
    std::cout << "Output: " << output_path << "\n";

    // Update registry
    update_registry(registry_path, config_hash, config_content, master_seed);
    std::cout << "Registry: " << registry_path << "\n";
}
