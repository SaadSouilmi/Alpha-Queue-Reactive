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
    // Legacy method: use new split functions
    auto orders = generate_racer_orders(alpha, best_bid, best_ask, rng);
    assign_timestamps(orders, base_time, rng);
    return orders;
}

std::vector<Order> LogisticRace::generate_racer_orders(double alpha,
                                                        int32_t best_bid, int32_t best_ask,
                                                        std::mt19937_64& rng) {
    std::vector<Order> orders;

    // Sample number of racers (scales with |α|)
    int num_racers = params_.sample_num_racers(alpha, rng);

    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    // Compute dynamic probabilities based on alpha
    double trade_prob, limit_prob, cancel_prob;
    params_.compute_probs(alpha, trade_prob, limit_prob, cancel_prob);

    // Cumulative thresholds for order type sampling
    double thresh_trade = trade_prob;
    double thresh_limit = thresh_trade + limit_prob;

    for (int i = 0; i < num_racers; ++i) {
        double r = uniform(rng);
        OrderType type;
        Side side;
        int32_t price;

        if (r < thresh_trade) {
            // TRADE: marketable limit at best_ask/best_bid at race trigger
            type = OrderType::Trade;
            side = (alpha > 0) ? Side::Bid : Side::Ask;
            price = (side == Side::Bid) ? best_ask : best_bid;

        } else if (r < thresh_limit) {
            // LIMIT: passive positioning on SAME side as alpha direction
            type = OrderType::Add;
            side = (alpha > 0) ? Side::Bid : Side::Ask;
            price = (side == Side::Bid) ? best_bid : best_ask;

        } else {
            // CANCEL: pull quotes to avoid adverse selection
            type = OrderType::Cancel;
            side = (alpha > 0) ? Side::Ask : Side::Bid;
            price = (side == Side::Ask) ? best_ask : best_bid;
        }

        int32_t size = params_.sample_racer_size(rng);
        // Create order with timestamp=0 (will be assigned later)
        orders.emplace_back(type, side, price, size, 0);
    }

    return orders;
}

void LogisticRace::assign_timestamps(std::vector<Order>& orders, int64_t base_time,
                                     std::mt19937_64& rng) {
    if (orders.empty()) return;

    // First order: base_time + delta
    int64_t time = base_time + delta_.sample(rng);
    orders[0].ts = time;

    // Subsequent orders: previous + gamma
    for (size_t i = 1; i < orders.size(); ++i) {
        time += sample_gamma(rng);
        orders[i].ts = time;
    }
}

} // namespace qr
