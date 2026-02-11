#include "simulation.h"
#include "race.h"
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>
#include <fstream>

namespace qr {

void Buffer::save_parquet(const std::string& path) const {
    // Build columns
    arrow::Int64Builder sequence_builder;
    arrow::Int32Builder best_bid_price_builder;
    arrow::Int32Builder best_bid_vol_builder;
    arrow::Int32Builder best_ask_price_builder;
    arrow::Int32Builder best_ask_vol_builder;
    arrow::Int32Builder second_bid_price_builder;
    arrow::Int32Builder second_bid_vol_builder;
    arrow::Int32Builder second_ask_price_builder;
    arrow::Int32Builder second_ask_vol_builder;
    arrow::DoubleBuilder imbalance_builder;
    arrow::DoubleBuilder mid_builder;
    arrow::Int64Builder timestamp_builder;
    arrow::StringBuilder type_builder;
    arrow::Int8Builder side_builder;
    arrow::Int32Builder price_builder;
    arrow::Int32Builder volume_builder;
    arrow::BooleanBuilder rejected_builder;
    arrow::BooleanBuilder partial_builder;
    arrow::DoubleBuilder bias_builder;
    arrow::DoubleBuilder alpha_builder;
    arrow::Int8Builder source_builder;

    for (const auto& r : records) {
        (void)sequence_builder.Append(r.sequence);
        (void)best_bid_price_builder.Append(r.best_bid_price);
        (void)best_bid_vol_builder.Append(r.best_bid_vol);
        (void)best_ask_price_builder.Append(r.best_ask_price);
        (void)best_ask_vol_builder.Append(r.best_ask_vol);
        (void)second_bid_price_builder.Append(r.second_bid_price);
        (void)second_bid_vol_builder.Append(r.second_bid_vol);
        (void)second_ask_price_builder.Append(r.second_ask_price);
        (void)second_ask_vol_builder.Append(r.second_ask_vol);
        (void)imbalance_builder.Append(r.imbalance);
        (void)mid_builder.Append(r.mid);
        (void)timestamp_builder.Append(r.timestamp);
        (void)type_builder.Append(r.type);
        (void)side_builder.Append(static_cast<int8_t>(r.side));
        (void)price_builder.Append(r.price);
        (void)volume_builder.Append(r.volume);
        (void)rejected_builder.Append(r.rejected);
        (void)partial_builder.Append(r.partial);
        (void)bias_builder.Append(r.bias);
        (void)alpha_builder.Append(r.alpha);
        (void)source_builder.Append(r.source);
    }

    std::shared_ptr<arrow::Array> sequence_arr;
    std::shared_ptr<arrow::Array> best_bid_price_arr, best_bid_vol_arr;
    std::shared_ptr<arrow::Array> best_ask_price_arr, best_ask_vol_arr;
    std::shared_ptr<arrow::Array> second_bid_price_arr, second_bid_vol_arr;
    std::shared_ptr<arrow::Array> second_ask_price_arr, second_ask_vol_arr;
    std::shared_ptr<arrow::Array> imbalance_arr;
    std::shared_ptr<arrow::Array> mid_arr;
    std::shared_ptr<arrow::Array> timestamp_arr, type_arr, side_arr;
    std::shared_ptr<arrow::Array> price_arr, volume_arr, rejected_arr, partial_arr;
    std::shared_ptr<arrow::Array> bias_arr, alpha_arr;
    std::shared_ptr<arrow::Array> source_arr;

    (void)sequence_builder.Finish(&sequence_arr);
    (void)best_bid_price_builder.Finish(&best_bid_price_arr);
    (void)best_bid_vol_builder.Finish(&best_bid_vol_arr);
    (void)best_ask_price_builder.Finish(&best_ask_price_arr);
    (void)best_ask_vol_builder.Finish(&best_ask_vol_arr);
    (void)second_bid_price_builder.Finish(&second_bid_price_arr);
    (void)second_bid_vol_builder.Finish(&second_bid_vol_arr);
    (void)second_ask_price_builder.Finish(&second_ask_price_arr);
    (void)second_ask_vol_builder.Finish(&second_ask_vol_arr);
    (void)imbalance_builder.Finish(&imbalance_arr);
    (void)mid_builder.Finish(&mid_arr);
    (void)timestamp_builder.Finish(&timestamp_arr);
    (void)type_builder.Finish(&type_arr);
    (void)side_builder.Finish(&side_arr);
    (void)price_builder.Finish(&price_arr);
    (void)volume_builder.Finish(&volume_arr);
    (void)rejected_builder.Finish(&rejected_arr);
    (void)partial_builder.Finish(&partial_arr);
    (void)bias_builder.Finish(&bias_arr);
    (void)alpha_builder.Finish(&alpha_arr);
    (void)source_builder.Finish(&source_arr);

    auto schema = arrow::schema({
        arrow::field("sequence", arrow::int64()),
        arrow::field("p_-1", arrow::int32()),
        arrow::field("q_-1", arrow::int32()),
        arrow::field("p_1", arrow::int32()),
        arrow::field("q_1", arrow::int32()),
        arrow::field("p_-2", arrow::int32()),
        arrow::field("q_-2", arrow::int32()),
        arrow::field("p_2", arrow::int32()),
        arrow::field("q_2", arrow::int32()),
        arrow::field("imbalance", arrow::float64()),
        arrow::field("mid", arrow::float64()),
        arrow::field("ts_event", arrow::int64()),
        arrow::field("event", arrow::utf8()),
        arrow::field("side", arrow::int8()),
        arrow::field("price", arrow::int32()),
        arrow::field("size", arrow::int32()),
        arrow::field("rejected", arrow::boolean()),
        arrow::field("partial", arrow::boolean()),
        arrow::field("bias", arrow::float64()),
        arrow::field("alpha", arrow::float64()),
        arrow::field("source", arrow::int8())
    });

    auto table = arrow::Table::Make(schema, {
        sequence_arr,
        best_bid_price_arr, best_bid_vol_arr,
        best_ask_price_arr, best_ask_vol_arr,
        second_bid_price_arr, second_bid_vol_arr,
        second_ask_price_arr, second_ask_vol_arr,
        imbalance_arr,
        mid_arr,
        timestamp_arr, type_arr, side_arr,
        price_arr, volume_arr, rejected_arr, partial_arr,
        bias_arr, alpha_arr, source_arr
    });

    auto outfile = arrow::io::FileOutputStream::Open(path).ValueOrDie();
    (void)parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, records.size());
}

Buffer run_simulation(OrderBook& lob, QRModel& model, int64_t duration,
                      Alpha* alpha, MarketImpact* impact,
                      Race* race) {
    Buffer buffer;
    int64_t time = 0;
    int64_t seq = 0;
    std::vector<Fill> fills;

    // RNG for race decisions (only used if race != nullptr)
    std::mt19937_64 race_rng(std::random_device{}());

    while (time < duration) {
        // Get current alpha value for race decision
        double alpha_val = alpha ? alpha->value() : 0.0;

        // Check if race should trigger (only if race mechanism is enabled)
        if (race && race->should_race(alpha_val, race_rng)) {
            // === RACE PATH ===
            int64_t race_start = time;

            // Generate racing orders with timestamps
            std::vector<Order> racers = race->generate_racer_orders(
                alpha_val, lob.best_bid(), lob.best_ask(), race_rng);
            race->assign_timestamps(racers, race_start, race_rng);

            // Process each racer
            for (auto& racer : racers) {
                EventRecord base_record;
                base_record.record_lob(lob);
                base_record.alpha = alpha_val;

                // Update time to racer timestamp
                time = racer.ts;
                if (impact) {
                    impact->step(time);
                }

                if (racer.type == OrderType::Trade) {
                    fills.clear();
                    lob.process(racer, &fills);

                    int64_t order_seq = seq++;
                    int32_t filled_size = 0;
                    for (const auto& fill : fills) filled_size += fill.size;

                    // One trade record with intended size
                    EventRecord trade_record = base_record;
                    trade_record.sequence = order_seq;
                    trade_record.timestamp = racer.ts;
                    trade_record.type = "Trade";
                    trade_record.side = racer.side;
                    trade_record.price = racer.price;
                    trade_record.volume = racer.size;
                    trade_record.rejected = racer.rejected;
                    trade_record.partial = (filled_size < racer.size);
                    trade_record.source = SOURCE_RACE;
                    buffer.records.push_back(trade_record);

                    // Update impact with total filled size
                    if (impact && filled_size > 0) {
                        impact->add_trade(racer.side, filled_size);
                    }

                    // If partial, record resting limit order (same LOB snapshot)
                    if (racer.partial) {
                        EventRecord limit_record = base_record;
                        limit_record.sequence = order_seq;
                        limit_record.timestamp = racer.ts;
                        limit_record.type = "Add";
                        limit_record.side = (racer.side == Side::Bid) ? Side::Ask : Side::Bid;
                        limit_record.price = racer.price;
                        limit_record.volume = racer.size - filled_size;
                        limit_record.rejected = false;
                        limit_record.partial = true;
                        limit_record.source = SOURCE_RACE;
                        buffer.records.push_back(limit_record);
                    }
                } else {
                    // Cancel or Add order
                    lob.process(racer);

                    EventRecord record = base_record;
                    record.sequence = seq++;
                    record.record_order(racer);
                    record.source = SOURCE_RACE;
                    buffer.records.push_back(record);
                }
            }

            // Advance alpha for race duration
            if (alpha) {
                alpha->step(time - race_start);
            }

            // Consume alpha after race (information acted upon)
            if (alpha && race->alpha_decay() > 0.0) {
                alpha->consume(race->alpha_decay());
            }

        }

        else {
            alpha_val = alpha ? alpha->value() : 0.0;
            double impact_val = impact ? impact->bias_factor() : 0.0;
            double total_bias = -alpha_val + impact_val;
            model.bias(total_bias);
            Order order = model.sample_order(time);

            int64_t dt = model.sample_dt();
            time += dt;
            order.ts = time;
            if (time >= duration) break;

            if (alpha) {
                alpha->step(dt);
            }
            if (impact) {
                impact->step(time);
            }

            // Capture LOB state before processing
            EventRecord base_record;
            base_record.record_lob(lob);
            base_record.bias = total_bias;
            base_record.alpha = alpha_val;

            if (order.type == OrderType::Trade) {
                fills.clear();
                lob.process(order, &fills);

                int64_t order_seq = seq++;
                int32_t filled_size = 0;
                for (const auto& fill : fills) filled_size += fill.size;

                // One trade record with intended size, LOB before
                EventRecord trade_record = base_record;
                trade_record.sequence = order_seq;
                trade_record.timestamp = order.ts;
                trade_record.type = "Trade";
                trade_record.side = order.side;
                trade_record.price = order.price;
                trade_record.volume = order.size;
                trade_record.rejected = false;
                trade_record.partial = (filled_size < order.size);
                trade_record.source = SOURCE_QR;
                buffer.records.push_back(trade_record);

                if (impact && filled_size > 0) {
                    impact->add_trade(order.side, filled_size);
                }

                // If partial, record resting limit order (same LOB snapshot)
                if (order.partial) {
                    EventRecord limit_record = base_record;
                    limit_record.sequence = order_seq;
                    limit_record.timestamp = order.ts;
                    limit_record.type = "Add";
                    limit_record.side = (order.side == Side::Bid) ? Side::Ask : Side::Bid;
                    limit_record.price = order.price;
                    limit_record.volume = order.size - filled_size;
                    limit_record.rejected = false;
                    limit_record.partial = true;
                    limit_record.source = SOURCE_QR;
                    buffer.records.push_back(limit_record);
                }
            } else {
                lob.process(order);

                EventRecord record = base_record;
                record.sequence = seq++;
                record.record_order(order);
                record.source = SOURCE_QR;
                buffer.records.push_back(record);
            }
        }
    }

    return buffer;
}

}
