#pragma once
#include "types.h"
#include "orderbook.h"
#include <vector>
#include <optional>
#include <cmath>
#include <algorithm>

namespace qr {

    struct StrategyTradeRecord {
        int64_t timestamp;
        int32_t inventory;   // Current inventory (after trade if filled)
        Side side;           // Buy or Sell
        int32_t size;        // Trade size
        int32_t price;       // Trade price
        bool rejected;
        double pnl;          // Cumulative realized PnL
    };

    struct StrategyBuffer {
        std::vector<StrategyTradeRecord> trades;

        void save_parquet(const std::string& path) const;

        size_t num_trades() const { return trades.size(); }

        size_t num_rejected() const {
            size_t count = 0;
            for (const auto& t : trades) {
                if (t.rejected) count++;
            }
            return count;
        }

        double fill_rate() const {
            if (trades.empty()) return 0.0;
            return 1.0 - static_cast<double>(num_rejected()) / trades.size();
        }
    };

    struct StrategyParams {
        double alpha_threshold;
        int32_t Q_max;
        int32_t max_trade_size;
        int32_t cooldown;  // Min non-rejected events between strategy trades

        StrategyParams(double threshold = 0.5, int32_t q_max = 10, int32_t trade_size = 5, int32_t cd = 10)
            : alpha_threshold(threshold), Q_max(q_max), max_trade_size(trade_size), cooldown(cd) {}
    };

    class AggressiveStrategy {
    public:
        AggressiveStrategy(const StrategyParams& params)
            : params_(params), inventory_(0), avg_cost_(0.0), realized_pnl_(0.0) {}

        // Returns order if should trade, nullopt otherwise
        std::optional<Order> decide(double alpha, int64_t time, const OrderBook& lob) {
            if (std::abs(alpha) < params_.alpha_threshold) {
                return std::nullopt;
            }

            Side side;
            if (alpha > params_.alpha_threshold && inventory_ < params_.Q_max) {
                side = Side::Bid;  // BUY: hit the ask
            } else if (alpha < -params_.alpha_threshold && inventory_ > -params_.Q_max) {
                side = Side::Ask;  // SELL: hit the bid
            } else {
                return std::nullopt;
            }

            // Size = min(max_trade_size, available_at_best, inventory_room)
            int32_t available = (side == Side::Bid) ? lob.best_ask_vol() : lob.best_bid_vol();
            int32_t inv_room = (side == Side::Bid)
                ? (params_.Q_max - inventory_)
                : (inventory_ + params_.Q_max);
            int32_t size = std::min({params_.max_trade_size, available, inv_room});

            if (size <= 0) {
                return std::nullopt;
            }

            // Price at current best (from strategy's view)
            int32_t price = (side == Side::Bid) ? lob.best_ask() : lob.best_bid();

            return Order(OrderType::Trade, side, price, size, time);
        }

        // Update inventory and PnL after fill
        void on_fill(const Order& order, int32_t filled_size, bool rejected) {
            if (rejected || filled_size == 0) return;

            int32_t price = order.price;

            if (order.side == Side::Bid) {
                // BUY: increase inventory
                if (inventory_ >= 0) {
                    // Adding to long position - update average cost
                    double total_cost = avg_cost_ * inventory_ + price * filled_size;
                    inventory_ += filled_size;
                    avg_cost_ = total_cost / inventory_;
                } else {
                    // Covering short position
                    int32_t cover_size = std::min(filled_size, -inventory_);
                    // Realize PnL: sold at avg_cost_, buying back at price
                    // Short PnL = (sell_price - buy_price) * size = (avg_cost_ - price) * cover_size
                    realized_pnl_ += (avg_cost_ - price) * cover_size;

                    inventory_ += filled_size;
                    if (inventory_ > 0) {
                        // Flipped to long, new cost basis is current price
                        avg_cost_ = price;
                    } else if (inventory_ == 0) {
                        avg_cost_ = 0.0;
                    }
                    // If still short, avg_cost_ stays the same
                }
            } else {
                // SELL: decrease inventory
                if (inventory_ <= 0) {
                    // Adding to short position - update average cost (sale price)
                    double total_cost = avg_cost_ * (-inventory_) + price * filled_size;
                    inventory_ -= filled_size;
                    avg_cost_ = total_cost / (-inventory_);
                } else {
                    // Closing long position
                    int32_t close_size = std::min(filled_size, inventory_);
                    // Realize PnL: bought at avg_cost_, selling at price
                    // Long PnL = (sell_price - buy_price) * size = (price - avg_cost_) * close_size
                    realized_pnl_ += (price - avg_cost_) * close_size;

                    inventory_ -= filled_size;
                    if (inventory_ < 0) {
                        // Flipped to short, new cost basis is current price
                        avg_cost_ = price;
                    } else if (inventory_ == 0) {
                        avg_cost_ = 0.0;
                    }
                    // If still long, avg_cost_ stays the same
                }
            }
        }

        int32_t inventory() const { return inventory_; }
        double pnl() const { return realized_pnl_; }
        double avg_cost() const { return avg_cost_; }
        int32_t cooldown() const { return params_.cooldown; }

        void reset() {
            inventory_ = 0;
            avg_cost_ = 0.0;
            realized_pnl_ = 0.0;
        }

    private:
        StrategyParams params_;
        int32_t inventory_;
        double avg_cost_;        // Average cost basis
        double realized_pnl_;    // Cumulative realized PnL
    };

}  // namespace qr
