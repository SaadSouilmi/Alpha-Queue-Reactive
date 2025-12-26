#pragma once
#include "orderbook.h"
#include <random>
#include <vector>
#include <string>
#include <memory>
#include <cmath>

namespace qr {

// First racer arrival distribution (truncated normal in log10 space)
struct DeltaDistrib {
    double mu, sigma, lower, upper;

    DeltaDistrib() = default;
    DeltaDistrib(const std::string& csv_path);

    // Sample and return nanoseconds
    int64_t sample(std::mt19937_64& rng) const;
};

// Inter-racer delay distribution (Weibull in log10 space)
struct WeibullDistrib {
    double k, scale, shift, max_log10;

    WeibullDistrib() = default;
    WeibullDistrib(const std::string& csv_path);

    // Sample and return nanoseconds
    int64_t sample(std::mt19937_64& rng) const;
};

// Inter-racer delay distribution (Gamma in log10 space)
struct GammaDistrib {
    double k, scale, shift, max_log10;

    GammaDistrib() = default;
    GammaDistrib(const std::string& csv_path);

    // Sample and return nanoseconds
    int64_t sample(std::mt19937_64& rng) const;
};

// Race parameters (hardcoded defaults, can be made configurable later)
struct RaceParams {
    double min_threshold = 0.8;   // no race if |α| < this
    double threshold = 0.7;       // logistic inflection point
    double steepness = 8.0;       // logistic steepness

    // Order type probabilities (must sum to <= 1, remainder is cancels)
    // Races only trigger at spread = 1
    double trade_prob = 0.45;       // 45% trades (if late, naturally rest as limits)
    double limit_next_prob = 0.30;  // 30% MM repositioning at Q_2
    // cancel_prob = 0.25           // 25% cancels (implicit)

    double base_mean_racers = 4.0;  // base geometric mean for number of racers
    double racer_scale = 2.5;       // scale factor: mean = base + scale * (|α| - threshold)
    double mean_size = 3.0;       // geometric mean for racer size

    // P(race|α) = 0 if |α| < min_threshold, else logistic
    double race_probability(double alpha) const {
        double abs_alpha = std::abs(alpha);
        if (abs_alpha < min_threshold) return 0.0;
        return 1.0 / (1.0 + std::exp(-steepness * (abs_alpha - threshold)));
    }

    // Sample number of racers from geometric distribution (mean scales with |α|)
    int sample_num_racers(double alpha, std::mt19937_64& rng) const {
        double abs_alpha = std::abs(alpha);
        // mean = base + scale * (|α| - threshold), clamped to at least base
        double mean = base_mean_racers + racer_scale * std::max(0.0, abs_alpha - threshold);
        double p = 1.0 / mean;
        std::geometric_distribution<int> dist(p);
        return dist(rng) + 1;  // at least 1 racer
    }

    // Sample racer size from geometric distribution
    int32_t sample_racer_size(std::mt19937_64& rng) const {
        double p = 1.0 / mean_size;
        std::geometric_distribution<int> dist(p);
        return static_cast<int32_t>(dist(rng) + 1);  // at least size 1
    }
};

// Abstract race interface
class Race {
public:
    virtual ~Race() = default;
    virtual bool should_race(double alpha, std::mt19937_64& rng) const = 0;
    virtual std::vector<Order> generate_racers(double alpha, int64_t base_time,
                                               int32_t best_bid, int32_t best_ask,
                                               std::mt19937_64& rng) = 0;
};

// No-op race (for running without race mechanism)
class NoRace : public Race {
public:
    bool should_race(double /*alpha*/, std::mt19937_64& /*rng*/) const override {
        return false;
    }
    std::vector<Order> generate_racers(double /*alpha*/, int64_t /*base_time*/,
                                       int32_t /*best_bid*/, int32_t /*best_ask*/,
                                       std::mt19937_64& /*rng*/) override {
        return {};
    }
};

// Logistic race model
class LogisticRace : public Race {
public:
    LogisticRace(const std::string& data_path, bool use_weibull);

    bool should_race(double alpha, std::mt19937_64& rng) const override;

    std::vector<Order> generate_racers(double alpha, int64_t base_time,
                                       int32_t best_bid, int32_t best_ask,
                                       std::mt19937_64& rng) override;

private:
    RaceParams params_;
    DeltaDistrib delta_;
    std::unique_ptr<WeibullDistrib> weibull_;
    std::unique_ptr<GammaDistrib> gamma_;
    bool use_weibull_;

    // Sample inter-racer delay
    int64_t sample_gamma(std::mt19937_64& rng) const {
        if (use_weibull_) {
            return weibull_->sample(rng);
        } else {
            return gamma_->sample(rng);
        }
    }
};

} // namespace qr
