#pragma once
#include "orderbook.h"
#include "qr_model.h"
#include <vector>
#include <string>

namespace qr {

    // Forward declaration
    class Race;

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
        bool rejected;
        bool partial;
        double bias;
        double alpha;
        double trade_sign_mean;
        bool is_race;

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


    // Simple loop - runs until duration elapsed
    Buffer run_simple(OrderBook& lob, QRModel& model, int64_t duration);
    Buffer run_metaorder(OrderBook& lob, QRModel& model, MarketImpact& impact, MetaOrder& metaorder, int64_t duration);
    Buffer run_with_alpha(OrderBook& lob, QRModel& model, MarketImpact& impact, Alpha& alpha, int64_t duration, double alpha_scale = 1.0);
    Buffer run_with_race(OrderBook& lob, QRModel& model, MarketImpact& impact, Race& race, Alpha& alpha, int64_t duration, double alpha_scale = 1.0);

    // Alpha PnL computation
    struct AlphaPnL {
        std::vector<double> lag_sec;
        std::vector<double> alpha_tickreturn_cov;

        void save_csv(const std::string& path) const;
    };

    AlphaPnL compute_alpha_pnl(const Buffer& buffer, const std::vector<int64_t>& lags_ns);

}
