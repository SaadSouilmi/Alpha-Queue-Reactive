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

    // Order type probabilities (constant, no limits in races)
    double trade_prob = 0.65;   // 65% trades
    double cancel_prob = 0.35;  // 35% cancels

    double base_mean_racers = 4.0;  // base geometric mean for number of racers
    double racer_scale = 2.5;       // scale factor: mean = base + scale * (|α| - threshold)
    double mean_size = 3.0;       // geometric mean for racer size
    double alpha_decay = 0.0;     // fraction of alpha consumed after race (0 = no decay)

    // P(race|signal) = 0 if |signal| < min_threshold, else logistic
    // signal = alpha + imbalance (or just alpha if no imbalance)
    double race_probability(double signal) const {
        double abs_signal = std::abs(signal);
        if (abs_signal < min_threshold) return 0.0;
        return 1.0 / (1.0 + std::exp(-steepness * (abs_signal - threshold)));
    }

    // Probabilities are constant (no limits, no dynamic adjustment)

    // Sample number of racers from geometric distribution (mean scales with |signal|)
    int sample_num_racers(double signal, std::mt19937_64& rng) const {
        double abs_signal = std::abs(signal);
        // mean = base + scale * (|signal| - threshold), clamped to at least base
        double mean = base_mean_racers + racer_scale * std::max(0.0, abs_signal - threshold);
        double p = 1.0 / mean;
        std::geometric_distribution<int> dist(p);
        return std::max(3, dist(rng) + 1);  // at least 3 racers
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
    virtual bool should_race(double signal, std::mt19937_64& rng) const = 0;

    // Legacy: generate racers with timestamps (for backward compatibility)
    virtual std::vector<Order> generate_racers(double alpha, int64_t base_time,
                                               int32_t best_bid, int32_t best_ask,
                                               std::mt19937_64& rng) = 0;

    // New: generate orders WITHOUT timestamps
    virtual std::vector<Order> generate_racer_orders(double alpha,
                                                      int32_t best_bid, int32_t best_ask,
                                                      std::mt19937_64& rng) = 0;

    // New: assign timestamps to orders (delta for first, gamma for rest)
    virtual void assign_timestamps(std::vector<Order>& orders, int64_t base_time,
                                   std::mt19937_64& rng) = 0;

    // Sample round-trip delay (time for market to see strategy order)
    virtual int64_t sample_roundtrip(std::mt19937_64& rng) const = 0;

    // Alpha decay fraction after race
    virtual double alpha_decay() const = 0;
};

// No-op race (for running without race mechanism)
class NoRace : public Race {
public:
    bool should_race(double /*signal*/, std::mt19937_64& /*rng*/) const override {
        return false;
    }
    std::vector<Order> generate_racers(double /*alpha*/, int64_t /*base_time*/,
                                       int32_t /*best_bid*/, int32_t /*best_ask*/,
                                       std::mt19937_64& /*rng*/) override {
        return {};
    }
    std::vector<Order> generate_racer_orders(double /*alpha*/,
                                              int32_t /*best_bid*/, int32_t /*best_ask*/,
                                              std::mt19937_64& /*rng*/) override {
        return {};
    }
    void assign_timestamps(std::vector<Order>& /*orders*/, int64_t /*base_time*/,
                          std::mt19937_64& /*rng*/) override {
        // No-op
    }
    int64_t sample_roundtrip(std::mt19937_64& /*rng*/) const override {
        return 0;  // No delay
    }
    double alpha_decay() const override { return 0.0; }
};

// Logistic race model
class LogisticRace : public Race {
public:
    LogisticRace(const std::string& data_path, bool use_weibull);
    LogisticRace(const std::string& data_path, bool use_weibull, const RaceParams& params);

    bool should_race(double signal, std::mt19937_64& rng) const override;

    // Legacy: generate racers with timestamps
    std::vector<Order> generate_racers(double alpha, int64_t base_time,
                                       int32_t best_bid, int32_t best_ask,
                                       std::mt19937_64& rng) override;

    // New: generate orders without timestamps
    std::vector<Order> generate_racer_orders(double alpha,
                                              int32_t best_bid, int32_t best_ask,
                                              std::mt19937_64& rng) override;

    // New: assign timestamps (delta for first, gamma for rest)
    void assign_timestamps(std::vector<Order>& orders, int64_t base_time,
                          std::mt19937_64& rng) override;

    int64_t sample_roundtrip(std::mt19937_64& rng) const override {
        return delta_.sample(rng);
    }
    double alpha_decay() const override { return params_.alpha_decay; }

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
