#pragma once
#include "orderbook.h"
#include <random>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace qr {

    // Size distribution parameters for geometric distribution
    // Key: (imb_bin_idx, event_type, queue_level) -> p parameter
    struct SizeDistributions {
        static constexpr int NUM_IMB_BINS = 21;
        static constexpr int32_t MAX_SIZE = 50;

        // Spread=1: [imb_bin][event_type: 0=Add, 1=Can, 2=Trd][queue: 0=q1, 1=q2]
        // Trd only has queue 1, so [imb][2][1] is unused
        std::array<std::array<std::array<double, 2>, 3>, NUM_IMB_BINS> p_params{};

        // Spread>=2: [imb_bin][event_type: 0=Create_Bid, 1=Create_Ask]
        std::array<std::array<double, 2>, NUM_IMB_BINS> p_create{};

        SizeDistributions() = default;
        SizeDistributions(const std::string& csv_path);

        // Sample size from geometric distribution capped at MAX_SIZE
        int32_t sample_size(int imb_bin, OrderType type, int queue, std::mt19937_64& rng) const {
            double p = 0.0;

            if (type == OrderType::CreateBid) {
                p = p_create[imb_bin][0];
            } else if (type == OrderType::CreateAsk) {
                p = p_create[imb_bin][1];
            } else {
                int type_idx = (type == OrderType::Add) ? 0 : (type == OrderType::Cancel) ? 1 : 2;
                int queue_idx = queue - 1;  // queue 1 -> 0, queue 2 -> 1
                p = p_params[imb_bin][type_idx][queue_idx];
            }

            if (p <= 0.0 || p >= 1.0) return 1;  // fallback

            std::geometric_distribution<int32_t> dist(p);
            int32_t size = dist(rng) + 1;  // geometric_distribution gives 0,1,2,... we want 1,2,3,...
            return std::min(size, MAX_SIZE);
        }
    };

    struct Event {
        OrderType type;
        Side side;
        int queue_nbr; // -2, -1, 1, 2, 0
    };

    struct StateParams {
        std::vector<Event> events;
        std::vector<double> base_probs; // immutable do not modify

        // Sampling state
        std::vector<double> probs;
        std::vector<double> cum_probs;
        double total;
        double lambda;

        void bias(double b) {
            // Exponential bias: always positive, no clamping needed
            // bid trades *= exp(b), ask trades *= exp(-b)
            total = 0.0;
            for (size_t i = 0; i < events.size(); i++) {
                if (events[i].type == OrderType::Trade) {
                    double factor = (events[i].side == Side::Bid) ? std::exp(b) : std::exp(-b);
                    probs[i] = base_probs[i] * factor;
                }
                else
                    probs[i] = base_probs[i];
                total += probs[i];
            }
            cum_probs[0] = probs[0];
            for (size_t i = 1; i < probs.size(); i++)
                cum_probs[i] = cum_probs[i-1] + probs[i];
        }

        const Event& sample_event(std::mt19937_64& rng) const {
            double u = std::uniform_real_distribution<>(0, total)(rng);
            auto it = std::lower_bound(cum_probs.begin(), cum_probs.end(), u);
            return events[it - cum_probs.begin()];
        }

        long long sample_dt(std::mt19937_64& rng) const {
            double dt = std::exponential_distribution<>(lambda)(rng);
            return static_cast<long long>(std::ceil(dt));
        }
    };


    struct QRParams {
        // 21 imbalance bins: -1.0, -0.9, ..., 0.0, ..., 0.9, 1.0
        // 2 spread states: spread=1 (index 0), spread>=2 (index 1)
        static constexpr int NUM_IMB_BINS = 21;
        std::array<std::array<StateParams, 2>, NUM_IMB_BINS> state_params;

        QRParams(const std::string& data_path);

        StateParams& get(uint8_t imbalance_bin, int32_t spread) {
            return state_params[imbalance_bin][spread];
        }
    };

    inline const char* order_type_to_string(OrderType type) {
        switch (type) {
            case OrderType::Add: return "Add";
            case OrderType::Cancel: return "Cancel";
            case OrderType::Trade: return "Trade";
            case OrderType::CreateBid: return "CreateBid";
            case OrderType::CreateAsk: return "CreateAsk";
            default: return "Unknown";
        }
    }

    class MarketImpact {
    public:
        virtual ~MarketImpact() = default;
        virtual void update(Side side, int64_t time) = 0;
        virtual double bias_factor(int64_t current_time) const = 0;
    };

    class NoImpact : public MarketImpact {
    public:
        void update(Side /*side*/, int64_t /*time*/) override {}
        double bias_factor(int64_t /*current_time*/) const override { return 0.0; }
    };

    class EMAImpact : public MarketImpact {
    public:
        EMAImpact(double alpha, double m) : alpha_(alpha), m_(m), ewma_(0.0) {}

        void update(Side side, int64_t /*time*/) override {
            double side_value = (side == Side::Bid) ? -1.0 : 1.0;
            ewma_ = alpha_ * side_value + (1 - alpha_) * ewma_;
        }

        double bias_factor(int64_t /*current_time*/) const override {
            return ewma_ * m_;
        }

        double ewma() const { return ewma_; }

    private:
        double alpha_;
        double m_;
        double ewma_;
    };

    class PowerLawImpact : public MarketImpact {
    public:
        PowerLawImpact(double alpha, double A, double m, double eps = 1.5e9)
            : alpha_(alpha), A_(A), m_(m), eps_(eps) {}

        void update(Side side, int64_t time) override {
            double sign = (side == Side::Bid) ? -1.0 : 1.0;
            timestamps_.push_back(time);
            signs_.push_back(sign);
        }

        double bias_factor(int64_t current_time) const override {
            double sum = 0.0;
            for (size_t i = 0; i < timestamps_.size(); i++) {
                double dt = static_cast<double>(current_time - timestamps_[i]) + eps_;
                double kernel;
                if (alpha_ == 0.5)       kernel = 1.0 / std::sqrt(dt);
                else if (alpha_ == 1.0)  kernel = 1.0 / dt;
                else if (alpha_ == 1.5)  kernel = 1.0 / (dt * std::sqrt(dt));
                else                     kernel = std::pow(dt, -alpha_);
                sum += signs_[i] * kernel;
            }
            return std::clamp(A_ * sum, -m_, m_);
        }

        void clear() {
            timestamps_.clear();
            signs_.clear();
        }

        size_t size() const { return timestamps_.size(); }

    private:
        double alpha_;
        double A_;
        double m_;
        double eps_;
        std::vector<int64_t> timestamps_;
        std::vector<double> signs_;
    };

    class LinearImpact : public MarketImpact {
    public:
        LinearImpact(double B, double A, double t0)
            : B_(B), A_(A), t0_(t0) {}

        void update(Side side, int64_t time) override {
            if (time == t0_)
                on_ = true;
        }
        double bias_factor(int64_t current_time) const override {
            if (!on_)
                return 0.0;
            double b = B_ + A_ * static_cast<double>(current_time - t0_);
            return std::max(b, 0.0);
        }


    private:
        double B_;
        double A_;
        int64_t t0_;
        bool on_ = false;
    };

    // ============================================================================
    // DeltaT (Inter-arrival time) Interface
    // ============================================================================

    class DeltaT {
    public:
        virtual ~DeltaT() = default;
        virtual int64_t sample(int imb_bin, int spread, std::mt19937_64& rng) const = 0;
    };

    // Exponential dt using lambda from QRParams (the original approach)
    class ExponentialDeltaT : public DeltaT {
    public:
        ExponentialDeltaT(const QRParams& params) : params_(params) {}

        int64_t sample(int imb_bin, int spread, std::mt19937_64& rng) const override {
            double lambda = params_.state_params[std::clamp(imb_bin, 0, 20)][std::clamp(spread, 0, 1)].lambda;
            double dt = std::exponential_distribution<>(lambda)(rng);
            return static_cast<int64_t>(std::ceil(dt));
        }

    private:
        const QRParams& params_;
    };

    // Gaussian mixture model for dt (in log10 space), conditional on imbalance bin and spread
    class MixtureDeltaT : public DeltaT {
    public:
        MixtureDeltaT(const std::string& csv_path) {
            std::ifstream file(csv_path);
            if (!file.is_open()) {
                throw std::runtime_error("Cannot open mixture CSV: " + csv_path);
            }

            std::string line;
            std::getline(file, line); // skip header

            while (std::getline(file, line)) {
                std::istringstream ss(line);
                std::string token;

                // imb_bin
                std::getline(ss, token, ',');
                double imb_bin_val = std::stod(token);
                int imb_idx = static_cast<int>(std::round(imb_bin_val * 10.0)) + 10;
                imb_idx = std::clamp(imb_idx, 0, 20);

                // spread (0 or 1)
                std::getline(ss, token, ',');
                int spread_idx = std::clamp(std::stoi(token), 0, 1);

                // w1, mu1, sigma1
                std::getline(ss, token, ',');
                params_[imb_idx][spread_idx].w1 = std::stod(token);
                std::getline(ss, token, ',');
                params_[imb_idx][spread_idx].mu1 = std::stod(token);
                std::getline(ss, token, ',');
                params_[imb_idx][spread_idx].sigma1 = std::stod(token);

                // w2, mu2, sigma2
                std::getline(ss, token, ',');
                params_[imb_idx][spread_idx].w2 = std::stod(token);
                std::getline(ss, token, ',');
                params_[imb_idx][spread_idx].mu2 = std::stod(token);
                std::getline(ss, token, ',');
                params_[imb_idx][spread_idx].sigma2 = std::stod(token);

                // w3, mu3, sigma3
                std::getline(ss, token, ',');
                params_[imb_idx][spread_idx].w3 = std::stod(token);
                std::getline(ss, token, ',');
                params_[imb_idx][spread_idx].mu3 = std::stod(token);
                std::getline(ss, token, ',');
                params_[imb_idx][spread_idx].sigma3 = std::stod(token);

                params_[imb_idx][spread_idx].loaded = true;
            }

            // Validate all entries loaded
            for (int i = 0; i < 21; ++i) {
                for (int s = 0; s < 2; ++s) {
                    if (!params_[i][s].loaded) {
                        double imb = (i - 10) / 10.0;
                        throw std::runtime_error("Missing mixture params for imb_bin=" +
                            std::to_string(imb) + ", spread=" + std::to_string(s));
                    }
                }
            }
        }

        int64_t sample(int imb_bin, int spread, std::mt19937_64& rng) const override {
            const auto& p = params_[std::clamp(imb_bin, 0, 20)][std::clamp(spread, 0, 1)];

            // Choose component based on cumulative weights
            double u = std::uniform_real_distribution<>(0.0, 1.0)(rng);
            double mu, sigma;
            if (u < p.w1) {
                mu = p.mu1;
                sigma = p.sigma1;
            } else if (u < p.w1 + p.w2) {
                mu = p.mu2;
                sigma = p.sigma2;
            } else {
                mu = p.mu3;
                sigma = p.sigma3;
            }

            // Sample from normal in log10 space, clamp to min 1 (so dt >= 10 ns)
            double log_dt = std::max(1.0, std::normal_distribution<>(mu, sigma)(rng));

            // Convert back: dt = 10^log_dt (in nanoseconds)
            return static_cast<int64_t>(std::pow(10.0, log_dt));
        }

    private:
        struct MixtureParams {
            double w1, mu1, sigma1;
            double w2, mu2, sigma2;
            double w3, mu3, sigma3;
            bool loaded = false;
        };
        std::array<std::array<MixtureParams, 2>, 21> params_;  // [imb_bin][spread]
    };

    // ============================================================================
    // Alpha Process Interface
    // ============================================================================

    class Alpha {
    public:
        virtual ~Alpha() = default;
        virtual void step(int64_t dt_ns) = 0;
        virtual double value() const = 0;
        virtual void reset() = 0;
        virtual void consume(double fraction) = 0;  // Reduce alpha by fraction (info acted upon)
    };

    class NoAlpha : public Alpha {
    public:
        void step(int64_t) override {}
        double value() const override { return 0.0; }
        void reset() override {}
        void consume(double) override {}  // No-op for NoAlpha
    };

    class OUAlpha : public Alpha {
    public:
        // kappa_per_min: mean reversion rate in min^-1
        // s: stationary standard deviation
        OUAlpha(double kappa_per_min, double s, uint64_t seed)
            : kappa_(kappa_per_min / (60.0 * 1e9)),
              sigma_(s * std::sqrt(2.0 * kappa_)),
              alpha_(0.0),
              rng_(seed) {}

        void step(int64_t dt_ns) override {
            double dt = static_cast<double>(dt_ns);
            double decay = std::exp(-kappa_ * dt);
            double var = (sigma_ * sigma_) * (1.0 - decay * decay) / (2.0 * kappa_);
            alpha_ = alpha_ * decay + std::sqrt(var) * normal_(rng_);
        }

        double value() const override { return alpha_; }
        void reset() override { alpha_ = 0.0; }
        void consume(double fraction) override {
            // Reduce alpha toward 0 by fraction (info acted upon)
            alpha_ *= (1.0 - fraction);
        }

    private:
        double kappa_;    // in ns^-1
        double sigma_;    // σ = s·√(2κ)
        double alpha_;
        std::mt19937_64 rng_;
        std::normal_distribution<> normal_{0.0, 1.0};
    };

    class QRModel {
    public:
        // Constructor without size distributions (sizes default to 1)
        QRModel(OrderBook* lob, const QRParams& params, uint64_t seed = 42) :
            lob_(lob), params_(params), size_dists_(nullptr), delta_t_(nullptr), rng_(seed) {}

        // Constructor with size distributions
        QRModel(OrderBook* lob, const QRParams& params, const SizeDistributions& size_dists, uint64_t seed = 42) :
            lob_(lob), params_(params), size_dists_(&size_dists), delta_t_(nullptr), rng_(seed) {}

        // Constructor with size distributions and delta_t
        QRModel(OrderBook* lob, const QRParams& params, const SizeDistributions& size_dists, const DeltaT& delta_t, uint64_t seed = 42) :
            lob_(lob), params_(params), size_dists_(&size_dists), delta_t_(&delta_t), rng_(seed) {}

        Order sample_order(int64_t current_time) {
            StateParams& state_params = get_state_params();
            const Event& event = state_params.sample_event(rng_);

            int32_t size = 1;
            if (size_dists_) {
                uint8_t imb_bin = get_imbalance_bin(lob_->imbalance());
                if (lob_->spread() == 1) {
                    // Spread=1: Add, Can, Trd at queue 1 or 2
                    int queue = std::abs(event.queue_nbr);
                    if (queue == 0) queue = 1;  // Trd uses queue 1
                    size = size_dists_->sample_size(imb_bin, event.type, queue, rng_);
                } else if (event.type == OrderType::CreateBid || event.type == OrderType::CreateAsk) {
                    // Spread>=2: Create events
                    size = size_dists_->sample_size(imb_bin, event.type, 0, rng_);
                }
            }

            Order order(event.type, event.side, get_price(event), size, current_time);
            return order;
        }

        int64_t sample_dt() {
            if (delta_t_) {
                uint8_t imb_bin = get_imbalance_bin(lob_->imbalance());
                int spread = std::min(lob_->spread() - 1, 1);
                return delta_t_->sample(imb_bin, spread, rng_);
            }
            StateParams& state_params = get_state_params();
            return state_params.sample_dt(rng_);
        }

        void bias(double b) {
            StateParams& state_params = get_state_params();
            state_params.bias(b);
        }

    private:
        OrderBook* lob_;
        QRParams params_;
        const SizeDistributions* size_dists_;
        const DeltaT* delta_t_;
        std::mt19937_64 rng_;

        // 21 bins: -1.0 (idx 0), -0.9 (idx 1), ..., 0.0 (idx 10), ..., 0.9 (idx 19), 1.0 (idx 20)
        // Bin boundaries:
        //   -1.0: [-1, -0.9)    -> idx 0
        //   -0.9: [-0.9, -0.8)  -> idx 1
        //   ...
        //   -0.1: [-0.1, 0)     -> idx 9
        //   0.0:  {0}           -> idx 10
        //   0.1:  (0, 0.1]      -> idx 11
        //   ...
        //   1.0:  (0.9, 1]      -> idx 20
        uint8_t get_imbalance_bin(double imbalance) {
            imbalance = std::clamp(imbalance, -1.0, 1.0);

            if (imbalance == 0.0) {
                return 10;
            }
            if (imbalance < 0.0) {
                // [-1, -0.9) -> 0, [-0.9, -0.8) -> 1, ..., [-0.1, 0) -> 9
                int bin = static_cast<int>(std::floor(imbalance * 10.0)) + 10;
                return static_cast<uint8_t>(std::clamp(bin, 0, 9));
            } else {
                // (0, 0.1] -> 11, (0.1, 0.2] -> 12, ..., (0.9, 1] -> 20
                int bin = static_cast<int>(std::ceil(imbalance * 10.0)) + 10;
                return static_cast<uint8_t>(std::clamp(bin, 11, 20));
            }
        }
        StateParams& get_state_params() {
            uint8_t imbalance_bin = get_imbalance_bin(lob_->imbalance());
            int32_t spread = std::min(lob_->spread()-1, 1);
            return params_.get(imbalance_bin, spread);
        }
        int32_t get_price(const Event& event) const {
            if (event.type == OrderType::CreateBid)
                return lob_->best_bid() + 1;
            if (event.type == OrderType::CreateAsk)
                return lob_->best_ask() - 1;
            if (event.side == Side::Bid)
                return lob_->best_bid() + 1 + event.queue_nbr;  // queue_nbr is -1 or -2
            return lob_->best_ask() - 1 + event.queue_nbr;      // queue_nbr is 1 or 2
        }
    };
};
