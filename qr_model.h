#pragma once
#include "orderbook.h"
#include <random>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>

namespace qr {
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
            // Multiply trade bid probs by 1 + b and trade ask probs by 1 - b
            total = 0.0;
            for (size_t i = 0; i < events.size(); i++) {
                if (events[i].type == OrderType::Trade) {
                    double factor = (events[i].side == Side::Bid) ? 1+b : 1-b;
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

    class QRModel {
    public:
        QRModel(OrderBook* lob, const QRParams& params, uint64_t seed = 42) :
            lob_(lob), params_(params), rng_(seed) {}

        Order sample_order(int64_t current_time) {
            StateParams& state_params = get_state_params();
            const Event& event = state_params.sample_event(rng_);
            Order order(event.type, event.side, get_price(event), 1, current_time);
            return order;
        }

        int64_t sample_dt() {
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