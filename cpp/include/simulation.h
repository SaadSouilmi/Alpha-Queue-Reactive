#pragma once
#include "orderbook.h"
#include "qr_model.h"
#include <vector>
#include <string>

namespace qr {

    // Forward declaration
    class Race;

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

    // Unified simulation function
    Buffer run_simulation(OrderBook& lob, QRModel& model, int64_t duration,
                          Alpha* alpha = nullptr, MarketImpact* impact = nullptr,
                          Race* race = nullptr);

}
