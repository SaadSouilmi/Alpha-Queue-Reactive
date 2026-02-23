#pragma once
#include "orderbook.h"
#include "qr_model.h"
#include "strategy.h"
#include <vector>
#include <string>

namespace qr {

    // Forward declarations
    class Race;
    struct DeltaDistrib;

    // Event source constants
    constexpr int8_t SOURCE_QR = 0;
    constexpr int8_t SOURCE_RACE = 1;

    struct EventRecord {
        // Sequence number for ordering
        int64_t sequence;

        // LOB state before event
        int32_t best_bid_price;
        int32_t best_bid_vol;
        int32_t best_ask_price;
        int32_t best_ask_vol;
        int32_t second_bid_price;
        int32_t second_bid_vol;
        int32_t second_ask_price;
        int32_t second_ask_vol;
        double imbalance;
        double mid;

        // Order metadata (recorded after processing)
        int64_t timestamp;
        std::string type;
        Side side;
        int32_t price;
        int32_t volume;
        int32_t filled_size = 0;
        bool rejected;
        bool partial;
        double bias;
        double alpha;
        int8_t source;  // SOURCE_QR=0, SOURCE_RACE=1

        void record_lob(const OrderBook& lob) {
            best_bid_price = lob.best_bid();
            best_bid_vol = lob.best_bid_vol();
            best_ask_price = lob.best_ask();
            best_ask_vol = lob.best_ask_vol();
            second_bid_price = lob.best_bid() - 1;
            second_bid_vol = 0;  // Fill with your logic
            second_ask_price = lob.best_ask() + 1;
            second_ask_vol = 0;  // Fill with your logic
            imbalance = static_cast<double>(best_bid_vol - best_ask_vol) /
                        static_cast<double>(best_bid_vol + best_ask_vol);
            mid = (best_bid_price + best_ask_price) / 2.0;
        }

        void record_order(const Order& order) {
            timestamp = order.ts;
            type = order_type_to_string(order.type);
            side = order.side;
            price = order.price;
            volume = order.size;
            rejected = order.rejected;
            partial = order.partial;
        }
    };

    struct Buffer {
        std::vector<EventRecord> records;

        int64_t total_time() const { return records.empty() ? 0 : records.back().timestamp; }
        int64_t num_events() const { return records.size(); }

        void save_parquet(const std::string& path) const;
    };

    struct MetaOrder {
        std::vector<int64_t> timestamps;
        std::vector<int32_t> sizes;
        Side side;
    };

    struct AlphaPnL {
        std::vector<double> lag_sec;
        std::vector<double> quantiles;
        std::vector<double> thresholds;
        std::vector<double> alpha_tickreturn_cov;
        std::vector<double> alpha_tickreturn_cov_ci;  // 95% CI half-width

        void save_csv(const std::string& path) const;
    };

    AlphaPnL compute_alpha_pnl(const Buffer& buffer, const std::vector<int64_t>& lags_ns, const std::vector<double>& quantiles);

    class P2Quantile {
        std::array<double, 5> q;   // marker heights (values)
        std::array<double, 5> n;   // marker positions (actual)
        std::array<double, 5> np;  // desired marker positions
        std::array<double, 5> dn;  // desired position increments
        int count = 0;
        double p;                   // target quantile
    
    public:
        explicit P2Quantile(double quantile) : p(quantile) {
            dn = {0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0};
        }
    
        void add(double x) {
            if (count < 5) {
                q[count] = x;
                count++;
                if (count == 5) {
                    std::sort(q.begin(), q.end());
                    for (int i = 0; i < 5; i++) n[i] = i;
                    np = {0.0, 2.0 * p, 4.0 * p, 2.0 + 2.0 * p, 4.0};
                }
                return;
            }
    
            // Find cell k
            int k;
            if (x < q[0]) { q[0] = x; k = 0; }
            else if (x < q[1]) k = 0;
            else if (x < q[2]) k = 1;
            else if (x < q[3]) k = 2;
            else if (x <= q[4]) k = 3;
            else { q[4] = x; k = 3; }
    
            // Increment positions of markers k+1 through 4
            for (int i = k + 1; i < 5; i++) n[i]++;
    
            // Update desired positions
            for (int i = 0; i < 5; i++) np[i] += dn[i];
    
            // Adjust markers 1, 2, 3
            for (int i = 1; i <= 3; i++) {
                double d = np[i] - n[i];
                if ((d >= 1.0 && n[i + 1] - n[i] > 1) ||
                    (d <= -1.0 && n[i - 1] - n[i] < -1)) {
                    int sign = (d > 0) ? 1 : -1;
                    double qp = parabolic(i, sign);
                    if (q[i - 1] < qp && qp < q[i + 1]) {
                        q[i] = qp;
                    } else {
                        q[i] = linear(i, sign);
                    }
                    n[i] += sign;
                }
            }
        }
    
        double estimate() const { return q[2]; }
    
        bool ready() const { return count >= 5; }
    
    private:
        double parabolic(int i, int sign) const {
            return q[i] + (sign / (n[i + 1] - n[i - 1])) * (
                (n[i] - n[i - 1] + sign) * (q[i + 1] - q[i]) / (n[i + 1] - n[i]) +
                (n[i + 1] - n[i] - sign) * (q[i] - q[i - 1]) / (n[i] - n[i - 1])
            );
        }
    
        double linear(int i, int sign) const {
            return q[i] + sign * (q[i + sign] - q[i]) / (n[i + sign] - n[i]);
        }
    };

    // Unified simulation function
    Buffer run_simulation(OrderBook& lob, QRModel& model, int64_t duration,
                          Alpha* alpha = nullptr, MarketImpact* impact = nullptr,
                          Race* race = nullptr);

    // Strategy trader simulation (accumulates stats in strategy trader)
    void run_simulation_with_strategy(
        OrderBook& lob, QRModel& model, int64_t duration,
        Alpha* alpha, MarketImpact* impact, Race* race,
        StrategyTrader& strategy,
        DeltaDistrib* roundtrip_distrib = nullptr);

}
