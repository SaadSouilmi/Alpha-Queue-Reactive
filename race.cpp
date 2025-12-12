#include "race.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace qr {

// ============================================================================
// DeltaDistrib - Truncated normal in log10 space
// ============================================================================

DeltaDistrib::DeltaDistrib(const std::string& csv_path) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open delta_distrib.csv: " + csv_path);
    }

    std::string line;
    std::getline(file, line);  // skip header

    if (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;

        std::getline(ss, token, ',');
        mu = std::stod(token);

        std::getline(ss, token, ',');
        sigma = std::stod(token);

        std::getline(ss, token, ',');
        lower = std::stod(token);

        std::getline(ss, token, ',');
        upper = std::stod(token);
    }
}

int64_t DeltaDistrib::sample(std::mt19937_64& rng) const {
    // Sample from truncated normal using rejection sampling
    std::normal_distribution<double> normal(mu, sigma);
    double x;
    do {
        x = normal(rng);
    } while (x < lower || x > upper);

    // Convert log10 to nanoseconds
    return static_cast<int64_t>(std::pow(10.0, x));
}

// ============================================================================
// WeibullDistrib - Weibull in log10 space
// ============================================================================

WeibullDistrib::WeibullDistrib(const std::string& csv_path) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open weibull_distrib.csv: " + csv_path);
    }

    std::string line;
    std::getline(file, line);  // skip header

    if (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;

        std::getline(ss, token, ',');
        k = std::stod(token);

        std::getline(ss, token, ',');
        scale = std::stod(token);

        std::getline(ss, token, ',');
        shift = std::stod(token);

        std::getline(ss, token, ',');
        max_log10 = std::stod(token);
    }
}

int64_t WeibullDistrib::sample(std::mt19937_64& rng) const {
    std::weibull_distribution<double> weibull(k, scale);
    double x;
    double log10_dt;
    do {
        x = weibull(rng);
        log10_dt = x + shift;
    } while (log10_dt > max_log10);

    return static_cast<int64_t>(std::pow(10.0, log10_dt));
}

// ============================================================================
// GammaDistrib - Gamma in log10 space
// ============================================================================

GammaDistrib::GammaDistrib(const std::string& csv_path) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open gamma_distrib.csv: " + csv_path);
    }

    std::string line;
    std::getline(file, line);  // skip header

    if (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;

        std::getline(ss, token, ',');
        k = std::stod(token);

        std::getline(ss, token, ',');
        scale = std::stod(token);

        std::getline(ss, token, ',');
        shift = std::stod(token);

        std::getline(ss, token, ',');
        max_log10 = std::stod(token);
    }
}

int64_t GammaDistrib::sample(std::mt19937_64& rng) const {
    std::gamma_distribution<double> gamma(k, scale);
    double x;
    double log10_dt;
    do {
        x = gamma(rng);
        log10_dt = x + shift;
    } while (log10_dt > max_log10);

    return static_cast<int64_t>(std::pow(10.0, log10_dt));
}

// ============================================================================
// LogisticRace
// ============================================================================

LogisticRace::LogisticRace(const std::string& data_path, bool use_weibull)
    : delta_(data_path + "/delta_distrib.csv"),
      use_weibull_(use_weibull)
{
    if (use_weibull_) {
        weibull_ = std::make_unique<WeibullDistrib>(data_path + "/weibull_distrib.csv");
    } else {
        gamma_ = std::make_unique<GammaDistrib>(data_path + "/gamma_distrib.csv");
    }
}

bool LogisticRace::should_race(double alpha, std::mt19937_64& rng) const {
    double p = params_.race_probability(alpha);
    if (p <= 0.0) return false;

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    return uniform(rng) < p;
}

std::vector<Order> LogisticRace::generate_racers(double alpha, int64_t base_time,
                                                  int32_t best_bid, int32_t best_ask,
                                                  std::mt19937_64& rng) {
    std::vector<Order> racers;

    // Determine side based on alpha direction
    // α > 0 → racers target ask side (sells)
    // α < 0 → racers target bid side (buys)
    Side side = (alpha > 0) ? Side::Ask : Side::Bid;
    int32_t price = (side == Side::Ask) ? best_ask : best_bid;

    // Sample number of racers (scales with |α|)
    int num_racers = params_.sample_num_racers(alpha, rng);

    // First racer arrives at base_time + delta
    int64_t time = base_time + delta_.sample(rng);

    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    for (int i = 0; i < num_racers; ++i) {
        // Determine order type: 70% trades, 30% cancels
        OrderType type = (uniform(rng) < params_.trade_prob) ? OrderType::Trade : OrderType::Cancel;

        // Sample racer size from geometric distribution
        int32_t size = params_.sample_racer_size(rng);
        racers.emplace_back(type, side, price, size, time);

        // Next racer arrives after gamma delay
        if (i < num_racers - 1) {
            time += sample_gamma(rng);
        }
    }

    return racers;
}

} // namespace qr
