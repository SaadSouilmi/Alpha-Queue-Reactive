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
#include <sys/file.h>
#include <unistd.h>
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

std::string hash_config(const std::string& content) {
    size_t h = std::hash<std::string>{}(content);
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << h;
    return oss.str();
}

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void update_registry(const std::string& registry_path, const std::string& hash,
                     const std::string& config_content, uint64_t seed_used) {
    // File lock for concurrent safety
    std::string lock_path = registry_path + ".lock";
    int lock_fd = open(lock_path.c_str(), O_CREAT | O_RDWR, 0666);
    if (lock_fd >= 0) flock(lock_fd, LOCK_EX);

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
    entry.AddMember("seed_used", seed_used, alloc);

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

    if (lock_fd >= 0) {
        flock(lock_fd, LOCK_UN);
        close(lock_fd);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <config.json>\n";
        std::cout << R"(
Example config.json:
{
  "ticker": "AAPL",
  "duration_hours": 1000,
  "seed": 12345,
  "use_mixture": true,
  "use_total_lvl": false,
  "use_race": true,
  "use_alpha": true,

  "impact": {"type": "no_impact"},

  "race": {
    "min_threshold": 0.8,
    "threshold": 0.7,
    "steepness": 8.0,
    "trade_prob": 0.65,
    "cancel_prob": 0.35,
    "base_mean_racers": 4.0,
    "racer_scale": 2.5,
    "mean_size": 3.0,
    "alpha_decay": 0.0,
    "use_weibull": true
  },

  "alpha": {
    "kappa": 0.5,
    "sigma": 0.5,
    "scale": 1.0
  }
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
    std::string ticker = get_string(doc, "ticker", "AAPL");
    std::string data_path = base_path + "/" + ticker;
    std::string results_path = base_path + "/results/" + ticker + "/samples/";

    uint64_t master_seed = get_uint64(doc, "seed", std::random_device{}());
    bool use_mixture = get_bool(doc, "use_mixture", true);
    bool use_total_lvl = get_bool(doc, "use_total_lvl", false);
    double duration_hours = get_double(doc, "duration_hours", 1000.0);

    // Impact config
    std::string impact_type = "no_impact";
    rj::Value impact_cfg(rj::kObjectType);
    if (doc.HasMember("impact") && doc["impact"].IsObject()) {
        impact_cfg.CopyFrom(doc["impact"], doc.GetAllocator());
        impact_type = get_string(impact_cfg, "type", "no_impact");
    }

    // Race config
    bool use_race = get_bool(doc, "use_race", false);
    bool use_alpha = get_bool(doc, "use_alpha", false);
    bool do_alpha_pnl = get_bool(doc, "compute_alpha_pnl", false);
    bool use_weibull = true;
    RaceParams race_params;
    if (use_race) {
        const auto& race = doc["race"];
        race_params.min_threshold = get_double(race, "min_threshold", 0.8);
        race_params.threshold = get_double(race, "threshold", 0.7);
        race_params.steepness = get_double(race, "steepness", 8.0);
        race_params.trade_prob = get_double(race, "trade_prob", 0.65);
        race_params.cancel_prob = get_double(race, "cancel_prob", 0.35);
        race_params.base_mean_racers = get_double(race, "base_mean_racers", 4.0);
        race_params.racer_scale = get_double(race, "racer_scale", 2.5);
        race_params.mean_size = get_double(race, "mean_size", 3.0);
        race_params.alpha_decay = get_double(race, "alpha_decay", 0.0);
        use_weibull = get_bool(race, "use_weibull", true);
    }

    // OU config (only used if race is enabled)
    double kappa = 0.5;
    double sigma = 0.5;
    double alpha_scale = 1.0;
    if (doc.HasMember("alpha") && doc["alpha"].IsObject()) {
        const auto& ou = doc["alpha"];
        kappa = get_double(ou, "kappa", 0.5);
        sigma = get_double(ou, "sigma", 0.5);
        alpha_scale = get_double(ou, "scale", 1.0);
    }

    // Output
    std::string output_path = results_path + config_hash + ".parquet";
    std::string alpha_pnl_path = results_path + config_hash + "_alpha_pnl.csv";
    std::string registry_path = results_path + "registry.json";

    // Print config
    std::cout << "Config: " << argv[1] << "\n";
    std::cout << "Hash: " << config_hash << "\n";
    std::cout << "Ticker: " << ticker << "\n";
    std::cout << "Duration: " << duration_hours << " hours\n";
    std::cout << "Delta_t: " << (use_mixture ? "mixture" : "exponential") << "\n";
    std::cout << "Total lvl: " << (use_total_lvl ? "on" : "off") << "\n";
    std::cout << "Impact: " << impact_type;
    if (impact_type == "ema") {
        std::cout << " (alpha=" << get_double(impact_cfg, "alpha", 0.01)
                  << ", m=" << get_double(impact_cfg, "m", 4.0) << ")";
    } else if (impact_type == "time_decay") {
        std::cout << " (half_life=" << get_double(impact_cfg, "half_life_sec", 30.0)
                  << "s, m=" << get_double(impact_cfg, "m", 4.0) << ")";
    } else if (impact_type == "power_law") {
        int K = impact_cfg.HasMember("half_lives") ? impact_cfg["half_lives"].GetArray().Size() : 0;
        std::cout << " (K=" << K << " components, m=" << get_double(impact_cfg, "m", 4.0) << ")";
    }
    std::cout << "\n";
    if (use_race) {
        std::cout << "Race: " << (use_weibull ? "weibull" : "gamma") << "\n";
        std::cout << "  min_threshold: " << race_params.min_threshold << "\n";
        std::cout << "  threshold: " << race_params.threshold << "\n";
        std::cout << "  steepness: " << race_params.steepness << "\n";
        std::cout << "  trade/cancel: " << race_params.trade_prob << "/"
                  << race_params.cancel_prob << "\n";
        std::cout << "  base_mean_racers: " << race_params.base_mean_racers << "\n";
        std::cout << "  racer_scale: " << race_params.racer_scale << "\n";
        std::cout << "  mean_size: " << race_params.mean_size << "\n";
        std::cout << "  alpha_decay: " << race_params.alpha_decay << "\n";
    } else {
        std::cout << "Race: off\n";
    }
    if (use_alpha) {
        std::cout << "OU: kappa=" << kappa << ", sigma=" << sigma << ", scale=" << alpha_scale << "\n";
    } else {
        std::cout << "Alpha: off\n";
    }
    std::cout << "Seed: " << master_seed << "\n";

    // Generate seeds
    std::mt19937_64 seed_rng(master_seed);
    uint64_t lob_seed = seed_rng();
    uint64_t model_seed = seed_rng();
    uint64_t alpha_seed = seed_rng();

    // Load data
    std::string params_path = data_path + "/qr_params";
    QueueDistributions dists(params_path + "/invariant_distributions_qmax50.csv");

    std::unique_ptr<MixtureDeltaT> delta_t_ptr;
    if (use_mixture) {
        std::string delta_t_file = use_race ? "/delta_t_gmm_floored.csv" : "/delta_t_gmm.csv";
        delta_t_ptr = std::make_unique<MixtureDeltaT>(params_path + delta_t_file);
    }

    // Initialize order book
    OrderBook lob(dists, 4, lob_seed);
    lob.init({1516, 1517, 1518, 1519},
              {4, 1, 10, 5},
              {1520, 1521, 1522, 1523},
              {6, 17, 22, 23});

    QRParams params(params_path);
    if (use_total_lvl) {
        params.load_total_lvl_quantiles(params_path + "/total_lvl_quantiles.csv");
        params.load_event_probabilities_3d(params_path + "/event_probabilities_3D.csv");
    }
    SizeDistributions size_dists(params_path + "/size_distrib.csv");

    // Create model
    std::unique_ptr<QRModel> model_ptr;
    if (delta_t_ptr) {
        model_ptr = std::make_unique<QRModel>(&lob, params, size_dists, *delta_t_ptr, model_seed);
    } else {
        model_ptr = std::make_unique<QRModel>(&lob, params, size_dists, model_seed);
    }
    QRModel& model = *model_ptr;

    int64_t duration = static_cast<int64_t>(duration_hours * 3600.0 * 1e9);

    // Create optional components
    std::unique_ptr<OUAlpha> alpha_ptr;
    std::unique_ptr<LogisticRace> race_ptr;
    std::unique_ptr<MarketImpact> impact_ptr;

    if (use_alpha) {
        alpha_ptr = std::make_unique<OUAlpha>(kappa, sigma, alpha_seed, alpha_scale);
    }
    if (use_race) {
        race_ptr = std::make_unique<LogisticRace>(data_path, use_weibull, race_params);
    }

    if (impact_type == "time_decay") {
        double half_life = get_double(impact_cfg, "half_life_sec", 30.0);
        double m = get_double(impact_cfg, "m", 4.0);
        impact_ptr = std::make_unique<TimeDecayImpact>(half_life, m);
    } else if (impact_type == "ema") {
        double a = get_double(impact_cfg, "alpha", 0.01);
        double m = get_double(impact_cfg, "m", 4.0);
        impact_ptr = std::make_unique<EMAImpact>(a, m);
    } else if (impact_type == "power_law") {
        double m = get_double(impact_cfg, "m", 4.0);
        std::vector<double> half_lives, weights;
        if (impact_cfg.HasMember("half_lives") && impact_cfg["half_lives"].IsArray()) {
            for (const auto& v : impact_cfg["half_lives"].GetArray())
                half_lives.push_back(v.GetDouble());
        }
        if (impact_cfg.HasMember("weights") && impact_cfg["weights"].IsArray()) {
            for (const auto& v : impact_cfg["weights"].GetArray())
                weights.push_back(v.GetDouble());
        }
        impact_ptr = std::make_unique<PowerLawImpact>(half_lives, weights, m);
    } else if (impact_type != "no_impact") {
        std::cerr << "Unknown impact type: " << impact_type << "\n";
        return 1;
    }

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    Buffer result = run_simulation(lob, model, duration,
                                    alpha_ptr.get(), impact_ptr.get(), race_ptr.get());
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << "Simulation: " << elapsed.count() << " s\n";
    std::cout << "Events: " << result.num_events() << "\n";

    // Save
    std::filesystem::create_directories(results_path);
    auto start_save = std::chrono::high_resolution_clock::now();
    result.save_parquet(output_path);
    auto end_save = std::chrono::high_resolution_clock::now();

    auto save_time = std::chrono::duration_cast<std::chrono::seconds>(end_save - start_save);
    std::cout << "Save: " << save_time.count() << " s\n";
    std::cout << "Output: " << output_path << "\n";

    if (do_alpha_pnl){
        std::vector<int64_t> lags_ns;
        for (int s = 30; s <= 30 * 60; s += 30) {
            lags_ns.push_back(static_cast<int64_t>(s) * 1'000'000'000);
        }
        std::vector<double> quantiles;
        for (int i = 1; i <= 9; i++){
            quantiles.push_back((double)i / 10.0);
        }
        auto start = std::chrono::high_resolution_clock::now();
        AlphaPnL result_alpha = compute_alpha_pnl(result, lags_ns, quantiles);
        result_alpha.save_csv(alpha_pnl_path);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << "Alpha P&L Computation: " << elapsed.count() << "s\n";
        
    }

    // Update registry
    update_registry(registry_path, config_hash, config_content, master_seed);
    std::cout << "Registry: " << registry_path << "\n";
}
